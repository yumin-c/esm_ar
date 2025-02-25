import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
import esm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random_seed = 216
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
    
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
            [('', ''.join(self.df.iloc[idx]['wt_seq'])[:1022])])
        _, _, esm_batch_tokens2 = self.esm_batch_converter(
            [('', ''.join(self.df.iloc[idx]['mut_seq'])[:1022])])

        # Standard features
        features = [
            esm_batch_tokens1,
            esm_batch_tokens2,
            self.df.iloc[idx]['pos'],
            len(self.df.iloc[idx]["wt_seq"]),
        ]

        # Additional configurable features
        for col in self.feature_columns:
            features.append(self.df.iloc[idx][col])

        # Target
        features.append(
            torch.unsqueeze(torch.FloatTensor([self.df.iloc[idx]['score']]), 0)
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

class ESMEffect(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        freeze_up_to = 10
        self.esm2mut, _ = esm.pretrained.esm2_t30_150M_UR50D() # .esm2_t12_35M_UR50D()    # alternatively: esm2_t30_150M_UR50D etc.

        for i in range(freeze_up_to):
            for param in self.esm2mut.layers[i].parameters():
                param.requires_grad = False

        embedding_dim = 640 # 640 for 150M model
        self.n_layers = 12

        # Regression head
        self.const = torch.nn.Parameter(-1 * torch.ones((1,embedding_dim)))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear((embedding_dim ), 1)

    def forward(self, tokens_wt, tokens_mut, pos, lengths):

        batch_size = tokens_mut.shape[0]
        mut = self.esm2mut(tokens_mut, repr_layers=[self.n_layers])['representations'][self.n_layers]

        x = []
        for i in range(batch_size):
            position = self.const * mut[i, pos[i], :]
            x.append(position)

        x = torch.stack(x).squeeze(1)
        predictions = self.classifier(self.dropout(x))

        return predictions

class ESMEffectSpeed(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        self.freeze_up_to = 10

        self.esm2mut, _ = esm.pretrained.esm2_t30_150M_UR50D()

        # Freeze layers up to the 10th layer
        for i in range(self.freeze_up_to):
            for param in self.esm2mut.layers[i].parameters():
                param.requires_grad = False

        self.n_layers = 12
        embedding_dim = 640

        # Cache for embeddings from the first 10 layers
        self.embedding_cache = defaultdict(torch.Tensor)

        # Weight constant for regression head
        self.const = nn.Parameter(-1 * torch.ones((1, embedding_dim)))

        # Regression head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, token_ids1, token_ids2, pos, lengths):
        batch_size = token_ids2.size(0)
        cached_embeddings = []

        for i in range(batch_size):
            seq_id = token_ids2[i].tolist()  # Convert tensor to a unique key
            if tuple(seq_id) in self.embedding_cache:
                # Use cached embedding
                cached_embeddings.append(self.embedding_cache[tuple(seq_id)])
            else:
                # Generate new embedding and cache it
                with torch.no_grad():
                    x = self.esm2mut(token_ids2[i].unsqueeze(0), repr_layers=list(range(0,11)))
                embedding = x['representations'][10]  # Take the output of the 10th layer
                self.embedding_cache[tuple(seq_id)] = embedding.detach()  # Detach from computation graph
                cached_embeddings.append(embedding)

        # Stack cached embeddings for the batch
        x = torch.cat(cached_embeddings, dim=0)

        # Pass through 11th and 12th layers (trainable)
        x, _ = self.esm2mut.layers[10](x)
        x, _ = self.esm2mut.layers[11](x)
        x = self.esm2mut.emb_layer_norm_after(x)

        # Apply element-wise multiplication with constant
        x = self.const * x[torch.arange(batch_size), pos, :]

        # Feed through the regression head
        predictions = self.classifier(self.dropout(x))
        return predictions

class ESMEffectFull(nn.Module):
    '''
    ESM-Effect full implementation with Speedup using cache.
    '''
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        self.freeze_up_to = 10
        self.esm2mut, _ = esm.pretrained.esm2_t30_150M_UR50D()
        self.esm2wt, _ = esm.pretrained.esm2_t30_150M_UR50D()

        # Freeze first 10 layers
        for model in [self.esm2mut, self.esm2wt]:
            for i in range(self.freeze_up_to):
                for param in model.layers[i].parameters():
                    param.requires_grad = False
        
        self.embedding_cache = defaultdict(torch.Tensor)  # Cache for embeddings
        
        embedding_dim = 640
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
                    x = self.esm2wt(tokens_wt[i].unsqueeze(0), repr_layers=list(range(0,11)))
                embedding = x['representations'][10]  # Take the output of the 10th layer
                self.embedding_cache[tuple(seq_id)] = embedding.detach()  # Detach from computation graph
                cached_embeddings_wt.append(embedding)

        for i in range(batch_size):
            seq_id = tokens_mut[i].tolist()  # Convert tensor to a unique key
            if tuple(seq_id) in self.embedding_cache:
                # Use cached embedding
                cached_embeddings.append(self.embedding_cache[tuple(seq_id)])
            else:
                # Generate new embedding and cache it
                with torch.no_grad():
                    x = self.esm2mut(tokens_mut[i].unsqueeze(0), repr_layers=list(range(0,11)))
                embedding = x['representations'][10]  # Take the output of the 10th layer
                self.embedding_cache[tuple(seq_id)] = embedding.detach()  # Detach from computation graph
                cached_embeddings.append(embedding)

        # Stack cached embeddings for the batch
        wt  = torch.cat(cached_embeddings_wt, dim=0)
        mut = torch.cat(cached_embeddings, dim=0)

        # Pass through 11th and 12th layers (trainable)
        wt, _ = self.esm2wt.layers[10](wt)
        wt, _ = self.esm2wt.layers[11](wt)
        wt = self.esm2wt.emb_layer_norm_after(wt)
        
        mut, _ = self.esm2mut.layers[10](mut)
        mut, _ = self.esm2mut.layers[11](mut)
        mut = self.esm2mut.emb_layer_norm_after(mut)

        position = self.const1 * wt[torch.arange(batch_size), pos, :] + self.const2 * mut[torch.arange(batch_size), pos, :]
        mean = self.const3 * wt[:, 1:].mean(dim=1) + self.const4 * mut[:, 1:].mean(dim=1)

        x = torch.cat((position, mean), dim=1)
        x = self.dropout(self.relu(self.classifierbig(self.dropout(x))))
        predictions = self.classifier(x)
        
        return predictions

def setup_optimizer_and_scheduler_esmeffect(model, train_loader, epochs, batch_size, lr_for_esm, lr_for_head):
    # Define learning rates
    lr_esm =  batch_size * lr_for_esm  # Lower learning rate for pre-trained ESM2 layers
    lr_new =  batch_size * lr_for_head  # Higher learning rate for new layers (regression head)

    # Group parameters
    esm_params = list(model.esm2mut.parameters())
    new_params = (
        [model.const] +
        list(model.classifier.parameters())
    )

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': esm_params, 'lr': lr_esm},
        {'params': new_params, 'lr': lr_new}
    ]

    # Initialize the optimizer with parameter groups
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[lr_esm, lr_new],  # Specify max_lr for each group
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    return optimizer, scheduler

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
        model: nn.Module,
        config: Dict,
        experiment_name: str,
        base_dir: str = "./experiments"
    ):
        self.model = model
        self.config = config
        self.base_dir = Path(base_dir)
        self.experiment_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device(self.config['device'])
        self.model = self.model.to(self.device)
        self.criterion = getattr(nn, self.config['criterion'])()
        self.scaler = torch.cuda.amp.GradScaler() # device=self.device)

        self.setup_directories()
        self.writer = SummaryWriter(self.log_dir)

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
        # self.optimizer, self.scheduler = setup_optimizer_and_scheduler_esmeffect(
        #     self.model,
        #     train_loader,
        #     self.config['epochs'],
        #     self.config['batch_size'],
        #     self.config['lr_esm'],
        #     self.config['lr_head']
        # )
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

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

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop with logging."""
        # Setup optimizer and scheduler here, now that we have the train_loader
        self.setup_training_components(train_loader)

        self.best_val_metric = float('inf')

        for self.current_epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            # Training
            train_loss = self.train_epoch(self.current_epoch, train_loader)
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)

            print(f"Epoch {self.current_epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            # Validation
            if val_loader:
                metrics = self.evaluate(val_loader)
                for name, value in metrics.items():
                    self.writer.add_scalar(f'Metrics/{name}', value, self.current_epoch)

                # Save best model according to BME
                if metrics['val_bme'] < self.best_val_metric:
                    self.best_val_metric = metrics['val_bme']
                    self.save_checkpoint(f"best_model.pt")

                # Also save periodic checkpoints
                if (self.current_epoch + 1) % 5 == 0:  # Save every 5 epochs
                    self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch+1}.pt")

            if small_train_loader:
                train_subset_metrics = self.evaluate(small_train_loader, prefix="small_train")
                for name, value in train_subset_metrics.items():
                    self.writer.add_scalar(f'Metrics/{name}', value, self.current_epoch)
                print(f"Small Train Pearson: {train_subset_metrics['small_train_pearson']:.4f}")
                print(f"Small Train Spearman: {train_subset_metrics['small_train_spearman']:.4f}")
                print(f"Small Train BME: {train_subset_metrics['small_train_bme']:.4f}")
                print(f"Small Train ROC: {train_subset_metrics['small_train_auroc']:.4f}")


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
                preds = self.model(*batch[:-2])  # Exclude labels
                all_preds.extend(preds.cpu().numpy())

        # Add predictions to the original dataframe
        dataframe['prediction'] = np.squeeze(all_preds)
        return dataframe

def plot_correlation(df, x_col, y_col, title, figsize=(8, 8)):
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
    plt.savefig('ESM_experiment_test.jpg')
    plt.show()
    plt.close(fig)

# Load the DataFrame
ar = pd.read_csv("data/filtered_dependency.csv")
ar = ar.loc[ar['Fold']=='Train']

# Position based split
train_pos = np.random.choice(
    ar["pos"].unique(), size=int(len(ar["pos"].unique()) * 0.95), replace=False
)

# Split data
train = ar[ar["pos"].isin(train_pos)].reset_index()
valid = ar[~ar["pos"].isin(train_pos)].reset_index()

# Set up a Training Run

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# Configuration
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 4,
    'epochs': 10,
    'criterion': 'MSELoss',
    'lr_esm': 5e-5,
    'lr_head': 1e-3,
    'num_workers': 2,
    'feature_columns': ['classification']  # Add or remove features as needed (experimental feature, column must be present in dataset)
}

# Create model and experiment manager
model = ESMEffectFull() # change setup function in ExperimentManager as well. 
experiment = ExperimentManager(model, config, "protein_prediction")

# Prepare data
train.loc[:, "wt_seq"] = train["wt"]
train.loc[:, "mut_seq"] = train["sequence"]
train.loc[:, "score"] = train["label"]
train.loc[:, "classification"] = train["classification"]

valid.loc[:, "wt_seq"] = valid["wt"]
valid.loc[:, "mut_seq"] = valid["sequence"]
valid.loc[:, "score"] = valid["label"]
valid.loc[:, "classification"] = valid["classification"]


train_dataset = ProteinDatasetESMEffect(train, config['feature_columns'])
val_dataset = ProteinDatasetESMEffect(valid, config['feature_columns'])

small_train = train.sample(frac=0.1, random_state=random_seed)  # 10% of the training data
small_train_dataset = ProteinDatasetESMEffect(small_train, config['feature_columns'])

small_train_loader = DataLoader(
    small_train_dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    shuffle=False,
    collate_fn=collate_fn_esmeffect
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    shuffle=True,
    collate_fn=collate_fn_esmeffect
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    shuffle=False,
    collate_fn=collate_fn_esmeffect
)

# Train model
experiment.train(train_loader, val_loader)

# Load the best model
# experiment.load_checkpoint("best_model.pt")
experiment.load_checkpoint("checkpoint_epoch_10.pt")

predicted_data = experiment.predict(valid, config['feature_columns'])
fig = plot_correlation(
    predicted_data,
    'score',
    'prediction',
    "Optimized ESM-Effect on AR Dependency Test set\n(20%, non-overlapping positions)"
)