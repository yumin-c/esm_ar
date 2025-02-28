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
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

random_seed = 216
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
    
from collections import defaultdict

# Modify the ProteinDatasetESMEffect class to include alphamissense as a target

class ProteinDatasetESMEffect(Dataset):
    """Dataset class for protein sequences with configurable features and multiple targets."""
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
        """Get item with configurable features and multiple targets."""
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

        # Target 1: score (may be NaN)
        score = self.df.iloc[idx].get('score', float('nan'))
        score_tensor = torch.unsqueeze(torch.FloatTensor([score]), 0)
        # Create score mask (1 if valid, 0 if NaN)
        score_mask = torch.unsqueeze(torch.FloatTensor([0 if np.isnan(score) else 1]), 0)
        score_tensor = torch.nan_to_num(score_tensor, nan=0.0)
        
        # Target 2: alphamissense
        alphamissense = self.df.iloc[idx]['alphamissense']
        alphamissense_tensor = torch.unsqueeze(torch.FloatTensor([alphamissense]), 0)
        
        # Add targets to features
        features.append(score_tensor)
        features.append(score_mask)
        features.append(alphamissense_tensor)

        return tuple(features)

    def __len__(self):
        return len(self.df)

# Update the collate function to handle multiple targets
def collate_fn_esmeffect(batch):
    """
    Custom collate function for ProteinDatasetESMEffect that handles variable number of features
    and multiple targets with masking.

    Args:
        batch: List of tuples from dataset __getitem__

    Returns:
        Tuple of tensors ready for model input
    """
    # Unzip the batch into separate lists
    # The last three elements are score, score_mask, and alphamissense
    features = list(zip(*batch))
    score_tensors = features[-3]
    score_masks = features[-2]
    alphamissense_tensors = features[-1]
    features = features[:-3]  # Remove targets from features

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
    pos = torch.tensor(np.array(features[2], dtype=int), dtype=torch.long)
    length = torch.tensor(np.array(features[3], dtype=int), dtype=torch.long)

    # Process any additional features (elements 4 onwards)
    additional_features = []
    for feature_list in features[4:]:
        additional_features.append(torch.tensor(feature_list))

    # Stack targets
    score_tensors = torch.stack(score_tensors)
    score_masks = torch.stack(score_masks)
    alphamissense_tensors = torch.stack(alphamissense_tensors)

    # Combine all elements
    output = [esm_batch_tokens1, esm_batch_tokens2, pos, length]
    output.extend(additional_features)
    output.append(score_tensors)
    output.append(score_masks)
    output.append(alphamissense_tensors)

    return tuple(output)

# Modify the ESMEffectFull model to support multiple outputs
class ESMEffectFull(nn.Module):
    '''
    ESM-Effect full implementation with Speedup using cache and multitask learning.
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
        
        # self.embedding_cache = defaultdict(torch.Tensor)  # Cache for embeddings
        
        embedding_dim = 640
        self.n_layers = 12

        # Regression head parameters
        self.const1 = nn.Parameter(torch.ones((1, embedding_dim)))
        self.const2 = nn.Parameter(-1 * torch.ones((1, embedding_dim)))
        self.const3 = nn.Parameter(torch.ones((1, embedding_dim)))
        self.const4 = nn.Parameter(-1 * torch.ones((1, embedding_dim)))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.shared_layer = nn.Linear(2 * embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        # Separate heads for different tasks
        self.score_head = nn.Linear(embedding_dim, 1)
        self.alphamissense_head = nn.Linear(embedding_dim, 1)

    def forward(self, tokens_wt, tokens_mut, pos, lengths, *args):
        batch_size = tokens_wt.shape[0]
        
        # cached_embeddings_wt = []
        # cached_embeddings = []
        
        # for i in range(batch_size):
        #     seq_id = tokens_wt[i].tolist()  # Convert tensor to a unique key
        #     if tuple(seq_id) in self.embedding_cache:
        #         # Use cached embedding
        #         cached_embeddings_wt.append(self.embedding_cache[tuple(seq_id)])
        #     else:
        #         with torch.no_grad():
        #             x = self.esm2wt(tokens_wt[i].unsqueeze(0), repr_layers=list(range(0,11)))
        #         embedding = x['representations'][10]  # Take the output of the 10th layer
        #         self.embedding_cache[tuple(seq_id)] = embedding.detach()  # Detach from computation graph
        #         cached_embeddings_wt.append(embedding)

        # for i in range(batch_size):
        #     seq_id = tokens_mut[i].tolist()  # Convert tensor to a unique key
        #     if tuple(seq_id) in self.embedding_cache:
        #         # Use cached embedding
        #         cached_embeddings.append(self.embedding_cache[tuple(seq_id)])
        #     else:
        #         # Generate new embedding and cache it
        #         with torch.no_grad():
        #             x = self.esm2mut(tokens_mut[i].unsqueeze(0), repr_layers=list(range(0,11)))
        #         embedding = x['representations'][10]  # Take the output of the 10th layer
        #         self.embedding_cache[tuple(seq_id)] = embedding.detach()  # Detach from computation graph
        #         cached_embeddings.append(embedding)

        # Stack cached embeddings for the batch
        wt  = self.esm2wt(tokens_wt, repr_layers=list(range(0,11)))['representations'][10]
        mut = self.esm2mut(tokens_mut, repr_layers=list(range(0,11)))['representations'][10]
        # wt = torch.cat(cached_embeddings_wt, dim=0)
        # mut = torch.cat(cached_embeddings, dim=0)

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
        shared_features = self.dropout(self.relu(self.shared_layer(self.dropout(x))))
        
        # Task-specific predictions
        score_predictions = self.score_head(shared_features)
        alphamissense_predictions = self.alphamissense_head(shared_features)
        
        return score_predictions, alphamissense_predictions


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
       list(model.shared_layer.parameters()) + list(model.score_head.parameters()) + list(model.alphamissense_head.parameters())
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


# Multitask loss function with masking and weight balancing
class MultitaskLoss:
    def __init__(self, score_weight=1.0, alphamissense_weight=1.0):
        self.criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply masking
        self.score_weight = score_weight
        self.alphamissense_weight = alphamissense_weight
        
    def __call__(self, score_preds, score_targets, score_masks, alphamisssense_preds, alphamisssense_targets):
        # Calculate masked loss for score predictions
        score_loss_per_sample = self.criterion(score_preds, score_targets)
        
        masked_score_loss = (score_loss_per_sample * score_masks).sum() / (score_masks.sum() + 1e-8)
        
        # Calculate loss for alphamissense predictions
        alphamissense_loss = self.criterion(alphamisssense_preds, alphamisssense_targets).mean()
        
        # Combine losses with weights
        total_loss = self.score_weight * masked_score_loss + self.alphamissense_weight * alphamissense_loss
        
        return total_loss, masked_score_loss, alphamissense_loss

# Update the ExperimentManager to handle multitask learning
class ExperimentManager:
    """Manages training experiments with logging and visualization for multitask models."""
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
        
        # Create multitask loss function
        self.multitask_loss = MultitaskLoss(
            score_weight=self.config.get('score_weight', 1.0),
            alphamissense_weight=self.config.get('alphamissense_weight', 1.0)
        )
        
        self.scaler = torch.cuda.amp.GradScaler() # device=self.device)

        self.setup_directories()
        self.writer = SummaryWriter(self.log_dir)
        
    def reset_model(self):
        """reset model for multi-fold training."""
        # self.model = ESMEffectFull().to(self.device)
        
        self.model = ESMEffectFull()

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
        self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model = self.model.to(self.device)

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_score_loss = 0
        total_alphamissense_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader):
            loss_dict = self._process_batch(batch, training=True)
            total_loss += loss_dict['total_loss']
            total_score_loss += loss_dict['score_loss']
            total_alphamissense_loss += loss_dict['alphamissense_loss']
            batch_count += 1

            # Log batch metrics
            self.writer.add_scalar('Loss/train_batch', loss_dict['total_loss'],
                                 epoch * len(train_loader) + batch_count)

        return {
            'train_loss': total_loss / batch_count,
            'train_score_loss': total_score_loss / batch_count,
            'train_alphamissense_loss': total_alphamissense_loss / batch_count
        }

    def evaluate(self, data_loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """Evaluate model on given dataset."""
        self.model.eval()
        total_loss = 0
        total_score_loss = 0
        total_alphamissense_loss = 0
        
        all_score_preds = []
        all_score_labels = []
        all_score_masks = []
        
        all_alphamissense_preds = []
        all_alphamissense_labels = []
        all_classifications = []  # Keep for compatibility

        with torch.no_grad():
            for batch in data_loader:
                loss_dict = self._process_batch(batch, training=False)
                total_loss += loss_dict['total_loss']
                total_score_loss += loss_dict['score_loss']
                total_alphamissense_loss += loss_dict['alphamissense_loss']

                # Store predictions and labels for correlation metrics
                all_score_preds.extend(self.score_preds.cpu().numpy())
                all_score_labels.extend(self.score_targets.cpu().numpy())
                all_score_masks.extend(self.score_masks.cpu().numpy())
                
                all_alphamissense_preds.extend(self.alphamissense_preds.cpu().numpy())
                all_alphamissense_labels.extend(self.alphamissense_targets.cpu().numpy())
                
                if hasattr(self, 'classifications'):
                    all_classifications.extend(self.classifications.cpu().numpy())

        # Process score metrics (with masking)
        all_score_preds = np.squeeze(all_score_preds)
        all_score_labels = np.squeeze(all_score_labels)
        all_score_masks = np.squeeze(all_score_masks)
        
        # Filter out samples where score is not available
        valid_score_indices = np.where(all_score_masks > 0)[0]
        valid_score_preds = all_score_preds[valid_score_indices]
        valid_score_labels = all_score_labels[valid_score_indices]
        
        # Calculate score metrics if we have valid samples
        score_pearson = 0
        score_spearman = 0
        score_bme = 0
        
        if len(valid_score_indices) > 0:
            # Score metrics
            score_pearson = pearsonr(valid_score_labels, valid_score_preds)[0]
            score_spearman = spearmanr(valid_score_labels, valid_score_preds)[0]
            
            # Calculate BME for score
            df_score = pd.DataFrame({'label': valid_score_labels, 'prediction': valid_score_preds})
            df_score['error'] = abs(df_score["label"] - df_score["prediction"])
            df_score['bin'] = pd.cut(df_score["label"], bins=100, labels=False, include_lowest=True)
            bin_stats = df_score.groupby('bin').agg(
                mean_error=('error', 'mean'),
                n_datapoints=('error', 'size')
            ).reset_index()
            score_bme = bin_stats['mean_error'].mean()
        
        # Process alphamissense metrics
        all_alphamissense_preds = np.squeeze(all_alphamissense_preds)
        all_alphamissense_labels = np.squeeze(all_alphamissense_labels)
        
        # Calculate alphamissense metrics
        alphamissense_pearson = pearsonr(all_alphamissense_labels, all_alphamissense_preds)[0]
        alphamissense_spearman = spearmanr(all_alphamissense_labels, all_alphamissense_preds)[0]
        
        # Calculate BME for alphamissense
        df_alpha = pd.DataFrame({'label': all_alphamissense_labels, 'prediction': all_alphamissense_preds})
        df_alpha['error'] = abs(df_alpha["label"] - df_alpha["prediction"])
        df_alpha['bin'] = pd.cut(df_alpha["label"], bins=100, labels=False, include_lowest=True)
        alpha_bin_stats = df_alpha.groupby('bin').agg(
            mean_error=('error', 'mean'),
            n_datapoints=('error', 'size')
        ).reset_index()
        alphamissense_bme = alpha_bin_stats['mean_error'].mean()
        
        # Calculate auROC for classification if we have the data
        auroc = 0
        if len(all_classifications) > 0:
            valid_indices = np.where(np.array(all_classifications) != 0)[0]
            if len(valid_indices) > 0:
                auroc = roc_auc_score(
                    np.array(all_classifications)[valid_indices], 
                    all_score_preds[valid_indices]
                )

        metrics = {
            f"{prefix}_loss": total_loss / len(data_loader),
            f"{prefix}_score_loss": total_score_loss / len(data_loader),
            f"{prefix}_alphamissense_loss": total_alphamissense_loss / len(data_loader),
            
            f"{prefix}_score_pearson": score_pearson,
            f"{prefix}_score_spearman": score_spearman,
            f"{prefix}_score_bme": score_bme,
            
            f"{prefix}_alphamissense_pearson": alphamissense_pearson,
            f"{prefix}_alphamissense_spearman": alphamissense_spearman,
            f"{prefix}_alphamissense_bme": alphamissense_bme,
            
            f"{prefix}_auroc": auroc
        }

        return metrics

    def _process_batch(self, batch: Tuple, training: bool = True) -> Dict[str, float]:
        """Process a single batch with multitask outputs."""
        # Move all tensors to device
        batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        
        # Extract targets from batch
        # Last three elements are score_targets, score_masks, and alphamissense_targets
        self.alphamissense_targets = batch[-1].float().squeeze(2)
        self.score_masks = batch[-2].float().squeeze(2)
        self.score_targets = batch[-3].float().squeeze(2)
        
        # Store classifications if available (for backward compatibility)
        if len(batch) > 6:  # We have at least one feature (classification)
            self.classifications = batch[-4].float()

        with torch.amp.autocast('cuda'):
            # Get predictions from model
            self.score_preds, self.alphamissense_preds = self.model(*batch[:-3])
            
            # Calculate loss with masking
            total_loss, score_loss, alphamissense_loss = self.multitask_loss(
                self.score_preds,
                self.score_targets,
                self.score_masks,
                self.alphamissense_preds,
                self.alphamissense_targets
            )

            if training:
                self.scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

                self.scaler.step(self.optimizer)  # Perform optimizer step
                self.scaler.update()  # Update scaler for mixed precision

                self.scheduler.step()  # Adjust learning rate AFTER optimizer step
                self.optimizer.zero_grad()  # Reset gradients

        return {
            'total_loss': total_loss.item(),
            'score_loss': score_loss.item(),
            'alphamissense_loss': alphamissense_loss.item()
        }

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
        self.metric_to_monitor = self.config.get('metric_to_monitor', 'val_alphamissense_spearman')

        for self.current_epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            # Training
            train_metrics = self.train_epoch(self.current_epoch, train_loader)
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'Loss/{name}', value, self.current_epoch)

            print(f"Fold {fold} Epoch {self.current_epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Train Score Loss: {train_metrics['train_score_loss']:.4f}")
            print(f"Train AlphaMissense Loss: {train_metrics['train_alphamissense_loss']:.4f}")
            
            # Validation
            if val_loader:
                metrics = self.evaluate(val_loader)
                for name, value in metrics.items():
                    self.writer.add_scalar(f'Metrics/{name}', value, self.current_epoch)

                # Save best model according to specified metric
                if metrics[self.metric_to_monitor] > self.best_val_metric:
                    self.best_val_metric = metrics[self.metric_to_monitor]
                    self.save_checkpoint(f"best_model_fold{fold}.pt")

                # Also save periodic checkpoints
                if (self.current_epoch + 1) % 5 == 0:  # Save every 5 epochs
                    self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch+1}_fold{fold}.pt")

            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Time/epoch', epoch_time, self.current_epoch)

            # Print progress
            if val_loader:
                print(f"Val Score Pearson: {metrics['val_score_pearson']:.4f}")
                print(f"Val Score Spearman: {metrics['val_score_spearman']:.4f}")
                print(f"Val Score BME: {metrics['val_score_bme']:.4f}")
                print(f"Val AlphaMissense Pearson: {metrics['val_alphamissense_pearson']:.4f}")
                print(f"Val AlphaMissense Spearman: {metrics['val_alphamissense_spearman']:.4f}")
                print(f"Val AlphaMissense BME: {metrics['val_alphamissense_bme']:.4f}")
                print(f"Val ROC: {metrics['val_auroc']:.4f}")
            print(f"Time: {epoch_time:.2f}s")

    def predict(self, dataframe: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Generate predictions for both score and alphamissense."""
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
        all_score_preds = []
        all_alphamissense_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                # Exclude score_targets, score_masks, and alphamissense_targets
                score_preds, alphamissense_preds = self.model(*batch[:-3])
                all_score_preds.extend(score_preds.cpu().numpy())
                all_alphamissense_preds.extend(alphamissense_preds.cpu().numpy())

        # Add predictions to the original dataframe
        dataframe['score_prediction'] = np.squeeze(all_score_preds)
        dataframe['alphamissense_prediction'] = np.squeeze(all_alphamissense_preds)
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


# Updated configuration with multitask weights
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'epochs': 20,
    'lr_esm': 2e-5,
    'lr_head': 1e-3,
    'num_workers': 2,
    'feature_columns': ['classification'],
    'score_weight': 1.0,             # Weight for score prediction loss
    'alphamissense_weight': 0.5,     # Weight for alphamissense prediction loss
    'metric_to_monitor': 'val_score_spearman'  # Metric to use for best model selection (consider val_alphamissense_spearman)
}

# Load the DataFrame
ar = pd.read_csv("data/all_dependency_merged.csv")

ar.loc[:, "wt_seq"] = ar["wt"]
ar.loc[:, "mut_seq"] = ar["sequence"]
ar.loc[:, "score"] = ar["label"]

ar_test_internal = ar.loc[ar['Fold']=='Test_internal']
ar_test_clinvar = ar.loc[ar['Fold']=='Test_ClinVar']
ar_train = ar.loc[~ar['Fold'].isin(['Test_ClinVar', 'Test_internal'])]

experiment = ExperimentManager(config, "ESM_multitask_ar_dependency_5cv")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.simplefilter('ignore', category=pd.errors.SettingWithCopyWarning)

from torch.utils.data import Sampler
class SemiSupervisedSampler(Sampler):
    def __init__(self, dataset, labeled_indices, unlabeled_indices, labeled_batch_size, batch_size, shuffle=True):
        self.dataset = dataset
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_batches = len(self.labeled_indices) // self.labeled_batch_size
        
    def __iter__(self):
        if self.shuffle:
            labeled_indices = np.random.permutation(self.labeled_indices).tolist()
            unlabeled_indices = np.random.permutation(self.unlabeled_indices).tolist()
        else:
            labeled_indices = self.labeled_indices.copy()
            unlabeled_indices = self.unlabeled_indices.copy()
            
        for i in range(self.num_batches):
            labeled_batch = labeled_indices[i * self.labeled_batch_size:(i + 1) * self.labeled_batch_size]
            
            unlabeled_start = (i * self.unlabeled_batch_size) % len(unlabeled_indices)
            unlabeled_end = unlabeled_start + self.unlabeled_batch_size
            
            if unlabeled_end <= len(unlabeled_indices):
                unlabeled_batch = unlabeled_indices[unlabeled_start:unlabeled_end]
            else:
                part1 = unlabeled_indices[unlabeled_start:]
                part2 = unlabeled_indices[:unlabeled_end - len(unlabeled_indices)]
                unlabeled_batch = part1 + part2
            
            batch_indices = labeled_batch + unlabeled_batch
            
            for idx in batch_indices:
                yield idx
                
    def __len__(self):
        return self.num_batches * self.batch_size

# 5 fold cross validation
for i in range(5):
    fold = i + 1
    
    experiment.reset_model()

    train = ar_train.loc[ar['Fold']!=f'Train_{fold}'].copy().reset_index()
    val   = ar_train.loc[ar['Fold']==f'Train_{fold}'].copy().reset_index()

    train_dataset = ProteinDatasetESMEffect(train, config['feature_columns'])
    val_dataset   = ProteinDatasetESMEffect(val, config['feature_columns'])
    
    train_sampler = SemiSupervisedSampler(
        dataset=train_dataset,
        labeled_indices=[i for i, sample in enumerate(train_dataset) if sample[-2] != 0],
        unlabeled_indices=[i for i, sample in enumerate(train_dataset) if sample[-2] == 0],
        labeled_batch_size=4,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_sampler = SemiSupervisedSampler(
        dataset=val_dataset,
        labeled_indices=[i for i, sample in enumerate(val_dataset) if sample[-2] != 0],
        unlabeled_indices=[i for i, sample in enumerate(val_dataset) if sample[-2] == 0],
        labeled_batch_size=4,
        batch_size=config['batch_size'],
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sampler=train_sampler,  # 커스텀 샘플러 사용
        collate_fn=collate_fn_esmeffect
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sampler=val_sampler,
        collate_fn=collate_fn_esmeffect
    )

    # Train model
    experiment.train(train_loader, val_loader, fold)

# Internal test for 5 models
ar_test_internal_avg = ar_test_internal.copy()
ar_test_internal_avg['score_prediction'] = 0

for i in range(5):
    fold = i + 1
    
    # Load the best model
    experiment.load_checkpoint(f"best_model_fold{fold}.pt")

    predicted_data = experiment.predict(ar_test_internal, config['feature_columns'])

    fig = plot_correlation(
        predicted_data,
        'score',
        'score_prediction',
        f"Optimized ESM-Effect on Validation set\n(non-overlapping positions)\n{experiment.experiment_name}",
        f'{experiment.exp_dir}/test_internal_fold{fold}.jpg'
    )
    
    ar_test_internal_avg['score_prediction'] += predicted_data['score_prediction'] / 5
    
fig = plot_correlation(
    ar_test_internal_avg,
    'score',
    'score_prediction',
    f"Optimized ESM-Effect on Validation set\n(non-overlapping positions)\n{experiment.experiment_name}",
    f'{experiment.exp_dir}/test_internal_ensemble.jpg'
)

# ClinVar test for 5 models
ar_test_clinvar_avg = ar_test_clinvar.copy()
ar_test_clinvar_avg['score_prediction'] = 0.0

for i in range(5):
    fold = i + 1
    
    # Load the best model
    experiment.load_checkpoint(f"best_model_fold{fold}.pt")

    predicted_data = experiment.predict(ar_test_clinvar, config['feature_columns'])
    
    isna_idx = ar_test_clinvar['score'].isna()

    fig = plot_correlation(
        predicted_data.loc[~isna_idx],
        'score',
        'score_prediction',
        f"Optimized ESM-Effect on Validation set\n(non-overlapping positions)\n{experiment.experiment_name}",
        f'{experiment.exp_dir}/test_clinvar_fold{fold}.jpg'
    )
    
    ar_test_clinvar_avg.loc[~isna_idx, 'score_prediction'] += predicted_data.loc[~isna_idx, 'score_prediction'] / 5
    
fig = plot_correlation(
    ar_test_clinvar_avg.loc[~isna_idx],
    'score',
    'score_prediction',
    f"Optimized ESM-Effect on Validation set\n(non-overlapping positions)\n{experiment.experiment_name}",
    f'{experiment.exp_dir}/test_clinvar_ensemble.jpg'
)

# ClinVar ROC analysis
clinvar_pred = ar_test_clinvar_avg['score_prediction']
clinvar_gt = ar_test_clinvar_avg['clinvar classification']

fpr, tpr, _ = roc_curve(clinvar_gt, clinvar_pred)
auroc = roc_auc_score(clinvar_gt, clinvar_pred)

# ROC Curve 플롯
plt.figure(figsize=(6, 6), dpi=200)
plt.plot(fpr, tpr, label=f'AUC = {auroc:.3f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # 대각선 기준선
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# 이미지 저장
plt.savefig(f'{experiment.exp_dir}/clinvar_roc.jpg', dpi=200, bbox_inches='tight')
plt.close()

print("ROC curve saved as 'clinvar_roc.jpg'")

# Last edit: No feat, true balanced position based split, lr changed, internal/clinvar test with ensemble
# Todo: multitask learning for alphamissense scores