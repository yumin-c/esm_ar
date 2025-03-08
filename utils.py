# Original work Copyright (c) 2025 Moritz Gollasch
# Modified work Copyright (c) 2025 Yumin Cheong

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
import esm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from collections import defaultdict


class ProteinDatasetESMEffect(Dataset):
    """Dataset class for protein sequences with configurable features."""
    def __init__(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Args:
            df: DataFrame containing protein data
            feature_columns: Optional list of additional feature column names to include
        """
        self.df = df
        self.feature_columns = feature_columns or []
        self._setup_esm()

    def _setup_esm(self):
        """Initialize ESM model components."""
        _, self.esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

    def __getitem__(self, idx):
        """Get item with configurable features."""
        # Base features
        _, _, esm_batch_tokens1 = self.esm_batch_converter(
            [('', ''.join(self.df.iloc[idx]['wt'])[:1022])])
        _, _, esm_batch_tokens2 = self.esm_batch_converter(
            [('', ''.join(self.df.iloc[idx]['sequence'])[:1022])])

        # Standard features
        features = [
            esm_batch_tokens1,
            esm_batch_tokens2,
            self.df.iloc[idx]['pos'],
            len(self.df.iloc[idx]["wt"]),
        ]

        # Additional configurable features
        for col in self.feature_columns:
            features.append(self.df.iloc[idx][col])

        # Target
        features.append(
            torch.unsqueeze(torch.FloatTensor([self.df.iloc[idx]['label']]), 0)
        )

        return tuple(features)

    def __len__(self):
        return len(self.df)

def collate_fn_esmeffect(batch):
    """
    Custom collate function for ProteinDatasetESMEffect that handles variable number of features.

    Args:
        batch: List of tuples from dataset __getitem__

    Returns:
        Tuple of tensors ready for model input
    """
    # Unzip the batch into separate lists
    # The last element is always the label, everything before that are features
    features = list(zip(*batch))
    labels = features[-1]
    features = features[:-1]

    # Process ESM tokens (always the first two elements)
    esm_batch_tokens1 = pad_sequence(
        [tokens[0].clone().detach() for tokens in features[0]],
        batch_first=True,
        padding_value=1
    )
    esm_batch_tokens2 = pad_sequence(
        [tokens[0].clone().detach() for tokens in features[1]],
        batch_first=True,
        padding_value=1
    )

    # Position and length (always elements 2 and 3)
    pos = torch.tensor(features[2], dtype=torch.long)
    length = torch.tensor(features[3], dtype=torch.long)

    # Process any additional features (elements 4 onwards)
    additional_features = []
    for feature_list in features[4:]:
        additional_features.append(torch.tensor(feature_list))

    # Stack labels
    labels = torch.stack(labels)

    # Combine all elements
    output = [esm_batch_tokens1, esm_batch_tokens2, pos, length]
    output.extend(additional_features)
    output.append(labels)

    return tuple(output)

class ESMEffectFull(nn.Module):
    '''
    ESM-Effect full implementation with Speedup using cache.
    '''
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        self.freeze_up_to = 10
        self.esm2mut, _ = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm2wt, _ = esm.pretrained.esm2_t12_35M_UR50D()

        # Freeze first 10 layers
        for model in [self.esm2mut, self.esm2wt]:
            for i in range(self.freeze_up_to):
                for param in model.layers[i].parameters():
                    param.requires_grad = False
        
        self.embedding_cache = defaultdict(torch.Tensor)  # Cache for embeddings
        
        embedding_dim = 480
        self.n_layers = 12

        # Regression head parameters
        self.const1 = nn.Parameter(torch.ones((1, embedding_dim)))
        self.const2 = nn.Parameter(-1 * torch.ones((1, embedding_dim)))
        self.const3 = nn.Parameter(torch.ones((1, embedding_dim)))
        self.const4 = nn.Parameter(-1 * torch.ones((1, embedding_dim)))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifierbig = nn.Linear(2 * embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, tokens_wt, tokens_mut, pos, lengths):
        batch_size = tokens_wt.shape[0]
        
        cached_embeddings_wt = []
        cached_embeddings = []
        
        for i in range(batch_size):
            seq_id = tokens_wt[i].tolist()  # Convert tensor to a unique key
            if tuple(seq_id) in self.embedding_cache:
                # Use cached embedding
                cached_embeddings_wt.append(self.embedding_cache[tuple(seq_id)])
            else:
                with torch.no_grad():
                    x = self.esm2wt(tokens_wt[i].unsqueeze(0), repr_layers=list(range(0,self.freeze_up_to+1)))
                embedding = x['representations'][self.freeze_up_to]
                self.embedding_cache[tuple(seq_id)] = embedding.detach()
                cached_embeddings_wt.append(embedding)

        for i in range(batch_size):
            seq_id = tokens_mut[i].tolist()  # Convert tensor to a unique key
            if tuple(seq_id) in self.embedding_cache:
                # Use cached embedding
                cached_embeddings.append(self.embedding_cache[tuple(seq_id)])
            else:
                # Generate new embedding and cache it
                with torch.no_grad():
                    x = self.esm2mut(tokens_mut[i].unsqueeze(0), repr_layers=list(range(0,self.freeze_up_to+1)))
                embedding = x['representations'][self.freeze_up_to]
                self.embedding_cache[tuple(seq_id)] = embedding.detach()
                cached_embeddings.append(embedding)

        # Stack cached embeddings for the batch
        wt  = torch.cat(cached_embeddings_wt, dim=0)
        mut = torch.cat(cached_embeddings, dim=0)

        wt, _ = self.esm2wt.layers[self.freeze_up_to](wt)
        wt, _ = self.esm2wt.layers[self.freeze_up_to+1](wt)
        wt = self.esm2wt.emb_layer_norm_after(wt)
        
        mut, _ = self.esm2mut.layers[self.freeze_up_to](mut)
        mut, _ = self.esm2mut.layers[self.freeze_up_to+1](mut)
        mut = self.esm2mut.emb_layer_norm_after(mut)

        position = self.const1 * wt[torch.arange(batch_size), pos, :] + self.const2 * mut[torch.arange(batch_size), pos, :]
        mean = self.const3 * wt[:, 1:].mean(dim=1) + self.const4 * mut[:, 1:].mean(dim=1)

        x = torch.cat((position, mean), dim=1)
        x = self.dropout(self.relu(self.classifierbig(self.dropout(x))))
        predictions = self.classifier(x)
        
        return predictions

def setup_optimizer_and_scheduler_esmeffectfull(model, train_loader, epochs, batch_size, lr_for_esm, lr_for_head):

    # Define learning rates
    lr_esm =  batch_size * lr_for_esm  # Lower learning rate for pre-trained ESM2 layers
    lr_new =  batch_size * lr_for_head  # Higher learning rate for new layers (classifier)

    # Group parameters
    esm_params = list(model.esm2mut.parameters()) + list(model.esm2wt.parameters()) #<- those are frozen
    new_params = (
        [model.const1] +
        [model.const2] +
        [model.const3] +
        [model.const4] +
       list(model.classifierbig.parameters()) + list(model.classifier.parameters())
    )

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': esm_params, 'lr': lr_esm},
        {'params': new_params, 'lr': lr_new}
    ]

    # Initialize the optimizer with parameter groups
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)


    # Create the scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[lr_esm, lr_new],  # Specify max_lr for each group
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    return optimizer, scheduler

class ExperimentManager:
    """Manages training experiments with logging and visualization."""
    def __init__(
        self,
        config: Dict,
        experiment_name: str,
        base_dir: str = "./experiments"
    ):
        self.config = config
        self.base_dir = Path(base_dir)
        self.experiment_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device(self.config['device'])
        self.reset_model()
        self.criterion = getattr(nn, self.config['criterion'])()
        self.scaler = torch.cuda.amp.GradScaler() # device=self.device)

        self.setup_directories()
        self.writer = SummaryWriter(self.log_dir)
        
    def reset_model(self):
        """reset model for multi-fold training."""
        self.model = ESMEffectFull().to(self.device)

    def setup_directories(self):
        """Create experiment directories."""
        self.exp_dir = self.base_dir / self.experiment_name
        self.log_dir = self.exp_dir / "logs"
        self.model_dir = self.exp_dir / "models"
        self.results_dir = self.exp_dir / "results"

        for dir_path in [self.exp_dir, self.log_dir, self.model_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_training_components(self, train_loader: DataLoader):
        """Initialize training components based on config."""
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler_esmeffectfull(
            self.model,
            train_loader,
            self.config['epochs'],
            self.config['batch_size'],
            self.config['lr_esm'],
            self.config['lr_head']
        )

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_count = 0

        for batch in train_loader:
            loss = self._process_batch(batch, training=True)
            total_loss += loss
            batch_count += 1

            # Log batch metrics
            self.writer.add_scalar('Loss/train_batch', loss,
                                 epoch * len(train_loader) + batch_count)

        return total_loss / batch_count

    def evaluate(self, data_loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """Evaluate model on given dataset."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_classifications = []

        with torch.no_grad():
            for batch in data_loader:
                loss = self._process_batch(batch, training=False)
                total_loss += loss

                # Store predictions and labels for correlation metrics
                all_preds.extend(self.predictions.cpu().numpy())
                all_labels.extend(self.labels.cpu().numpy())
                all_classifications.extend(self.classifications.cpu().numpy())

        # Calculate metrics
        all_preds = np.squeeze(all_preds)
        all_labels = np.squeeze(all_labels)
        all_classifications = np.squeeze(all_classifications)

        df = pd.DataFrame({'label': all_labels, 'prediction': all_preds})
        df['error'] = abs(df["label"] - df["prediction"])
        df['bin'] = pd.cut(df["label"], bins=100, labels=False, include_lowest=True)
        bin_stats = df.groupby('bin').agg(
            mean_error=('error', 'mean'),
            n_datapoints=('error', 'size')
        ).reset_index()
        bme = bin_stats['mean_error'].mean()

        # Calculate auROC for classification
        auroc = roc_auc_score(all_classifications[all_classifications != 0], all_preds[all_classifications != 0])

        metrics = {
            f"{prefix}_loss": total_loss / len(data_loader),
            f"{prefix}_pearson": pearsonr(all_labels, all_preds)[0],
            f"{prefix}_spearman": spearmanr(all_labels, all_preds)[0],
            f"{prefix}_bme": bme,
            f"{prefix}_auroc": auroc
        }

        return metrics

    def _process_batch(self, batch: Tuple, training: bool = True) -> float:
        """Process a single batch."""
        # Move all tensors to device
        batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        self.labels = batch[-1].float()  # Last element is always the target
        self.classifications = batch[-2].float()

        #with autocast(device_type='cuda', dtype=torch.float16):
        with torch.amp.autocast('cuda'):
            self.predictions = self.model(*batch[:-2])  # All but last element are features
            loss = self.criterion(self.predictions, self.labels.squeeze(2))

            if training:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

                self.scaler.step(self.optimizer)  # Perform optimizer step
                self.scaler.update()  # Update scaler for mixed precision

                self.scheduler.step()  # Adjust learning rate AFTER optimizer step
                self.optimizer.zero_grad()  # Reset gradients

        return loss.item()

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint including model state, optimizer state, and config.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_metric': self.best_val_metric if hasattr(self, 'best_val_metric') else None,
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else None
        }

        save_path = self.model_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, filename: str):
        """
        Load a model checkpoint and restore training state.

        Args:
            filename: Name of the checkpoint file to load
        """
        checkpoint_path = self.model_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler states if they exist
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other training states
        if 'best_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_metric']
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']

        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['config']  # Return loaded config for reference

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, fold: int = 1):
        """Full training loop with logging."""
        # Setup optimizer and scheduler here, now that we have the train_loader
        self.setup_training_components(train_loader)

        self.best_val_metric = - float('inf')

        for self.current_epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            # Training
            train_loss = self.train_epoch(self.current_epoch, train_loader)
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)

            print(f"Fold {fold} Epoch {self.current_epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            # Validation
            if val_loader:
                metrics = self.evaluate(val_loader)
                for name, value in metrics.items():
                    self.writer.add_scalar(f'Metrics/{name}', value, self.current_epoch)

                # Save best model according to Spearman R
                if metrics['val_spearman'] > self.best_val_metric:
                    self.best_val_metric = metrics['val_spearman']
                    self.save_checkpoint(f"best_model_fold{fold}.pt")

                # Also save periodic checkpoints
                if (self.current_epoch + 1) % 5 == 0:  # Save every 5 epochs
                    self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch+1}_fold{fold}.pt")

            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Time/epoch', epoch_time, self.current_epoch)

            # Print progress
            if val_loader:
                print(f"Val Pearson: {metrics['val_pearson']:.4f}")
                print(f"Val Spearman: {metrics['val_spearman']:.4f}")
                print(f"Val BME: {metrics['val_bme']:.4f}")
                print(f"Val ROC: {metrics['val_auroc']:.4f}")
            print(f"Time: {epoch_time:.2f}s")


    def predict(self, dataframe: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:

        # Preprocess the dataframe into a dataset and dataloader
        dataset = ProteinDatasetESMEffect(dataframe, feature_columns)
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=collate_fn_esmeffect
        )

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                preds = self.model(*batch[:-2])  # Exclude classifications and target scores
                all_preds.extend(preds.cpu().numpy())

        # Add predictions to the original dataframe
        dataframe['prediction'] = np.squeeze(all_preds)
        return dataframe

def plot_correlation(df, x_col, y_col, title, filename, figsize=(8, 8)):
    """
    Creates a scatter plot with a diagonal reference line and saves the plot.

    Args:
        df: pandas DataFrame containing the data
        x_col: name of column for x-axis
        y_col: name of column for y-axis
        title: title for the plot
        filename: file name to save the plot in Google Drive
        figsize: tuple of figure dimensions
    """
    ev = pd.DataFrame({'label': df[x_col], 'prediction': df[y_col]})
    ev['error'] = abs(ev["label"] - ev["prediction"])
    ev['bin'] = pd.cut(ev["label"], bins=100, labels=False, include_lowest=True)
    bin_stats = ev.groupby('bin').agg(
        mean_error=('error', 'mean'),
        n_datapoints=('error', 'size')
    ).reset_index()
    bme = bin_stats['mean_error'].mean()

    # Calculate correlations
    spearman_corr, _ = spearmanr(df[x_col], df[y_col])
    pearson_corr, _ = pearsonr(df[x_col], df[y_col])

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.scatter(df[x_col], df[y_col], alpha=0.5)

    # Add diagonal reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8)
    ax.set_aspect('equal')

    # Add correlation text
    corr_text = f"R = {spearman_corr:.2f}\nr = {pearson_corr:.2f}"
    ax.text(0.1, 0.8, corr_text, transform=ax.transAxes, fontsize=12)

    # Labels and title
    ax.set_xlabel('Score/10')
    ax.set_ylabel('Prediction')
    ax.set_title(f"{title} ({len(df)} test mutations)")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close(fig)