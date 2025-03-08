# Original work Copyright (c) 2025 Moritz Glaser
# Modified work Copyright (c) 2025 Yumin Cheong

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from utils import ProteinDatasetESMEffect, ExperimentManager
from utils import collate_fn_esmeffect, plot_correlation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

random_seed = 216
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# Configuration

mode = 'AR' # or 'ENZ', 'PROTAC'

if mode == 'AR':
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'epochs': 10,
        'criterion': 'MSELoss',
        'lr_esm': 2e-5,
        'lr_head': 1e-3,
        'num_workers': 4,
        'feature_columns': ['classification']
    }
elif mode == 'ENZ':
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'epochs': 10,
        'criterion': 'MSELoss',
        'lr_esm': 2e-5,
        'lr_head': 1e-3,
        'num_workers': 4,
        'feature_columns': ['classification']
    }
else: # PROTAC
    config = {
        'device': 'cuda:2' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'epochs': 10,
        'criterion': 'MSELoss',
        'lr_esm': 2e-5,
        'lr_head': 1e-3,
        'num_workers': 4,
        'feature_columns': ['classification']
    }

# Load the DataFrame
if mode == 'AR':
    ar = pd.read_csv("data/filtered_dependency.csv")
    experiment = ExperimentManager(config, "ESM_ar_dependency_5cv")
elif mode == 'ENZ':
    ar = pd.read_csv("data/filtered_enz.csv")
    experiment = ExperimentManager(config, "ESM_ar_enz_5cv")
else:
    ar = pd.read_csv("data/filtered_arv.csv")
    experiment = ExperimentManager(config, "ESM_ar_arv_5cv")

ar_test_internal = ar.loc[ar['Fold']=='Test_internal']
ar_test_external = ar.loc[ar['Fold']=='Test_ClinVar'] if mode == 'AR' else ar.loc[ar['Fold']=='Test_external']
ar_train = ar.loc[~ar['Fold'].isin(['Test_ClinVar', 'Test_internal'])] if mode == 'AR' else ar.loc[~ar['Fold'].isin(['Test_external', 'Test_internal'])]

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# 5 fold cross validation
for i in range(5):
    fold = i + 1
    
    experiment.reset_model()

    # Split data
    train = ar_train.loc[ar['Fold']!=f'Train_{fold}'].copy().reset_index()
    val   = ar_train.loc[ar['Fold']==f'Train_{fold}'].copy().reset_index()

    train_dataset = ProteinDatasetESMEffect(train, config['feature_columns'])
    val_dataset   = ProteinDatasetESMEffect(val, config['feature_columns'])


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
    experiment.train(train_loader, val_loader, fold)

# Internal test for 5 models
ar_test_internal_avg = ar_test_internal.copy()
ar_test_internal_avg['prediction'] = 0

for i in range(5):
    fold = i + 1
    
    # Load the best model
    experiment.load_checkpoint(f"best_model_fold{fold}.pt")

    predicted_data = experiment.predict(ar_test_internal, config['feature_columns'])

    fig = plot_correlation(
        predicted_data,
        'label',
        'prediction',
        f"ESM-Effect on Validation set\n(non-overlapping positions)\n{experiment.experiment_name}",
        f'{experiment.exp_dir}/test_internal_fold{fold}.jpg'
    )
    
    ar_test_internal_avg['prediction'] += predicted_data['prediction'] / 5
    
fig = plot_correlation(
    ar_test_internal_avg,
    'label',
    'prediction',
    f"ESM-Effect on Validation set\n(non-overlapping positions)\n{experiment.experiment_name}",
    f'{experiment.exp_dir}/test_internal_ensemble.jpg'
)

ar_test_external_avg = ar_test_external.copy()
ar_test_external_avg['prediction'] = 0

for i in range(5):
    fold = i + 1
    
    # Load the best model
    experiment.load_checkpoint(f"best_model_fold{fold}.pt")

    predicted_data = experiment.predict(ar_test_external, config['feature_columns'])

    if mode == 'AR':
        fig = plot_correlation(
            predicted_data,
            'label',
            'prediction',
            f"ESM-Effect on ClinVar-listed variants\nSingle model (Fold {fold})\n{experiment.experiment_name}",
            f'{experiment.exp_dir}/test_clinvar_fold{fold}.jpg'
        )
    
    ar_test_external_avg['prediction'] += predicted_data['prediction'] / 5

if mode == 'AR':
    fig = plot_correlation(
        ar_test_external_avg,
        'label',
        'prediction',
        f"Ensembled ESM-Effect performance on ClinVar-listed variants\n{experiment.experiment_name}",
        f'{experiment.exp_dir}/test_clinvar_ensemble.jpg'
    )

    # ClinVar ROC analysis

    clinvar_pred = ar_test_external_avg['prediction']
    clinvar_gt = ar_test_external_avg['clinvar classification']

    fpr, tpr, _ = roc_curve(clinvar_gt, clinvar_pred)
    auroc = roc_auc_score(clinvar_gt, clinvar_pred)

    plt.figure(figsize=(6, 6), dpi=200)
    plt.plot(fpr, tpr, label=f'AUC = {auroc:.3f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # 대각선 기준선
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ClinVar ROC Curve')
    plt.legend(loc='lower right')

    plt.savefig(f'{experiment.exp_dir}/clinvar_roc.jpg', dpi=200, bbox_inches='tight')
    plt.close()

    print("ROC curve saved as 'clinvar_roc.jpg'")

missing = pd.read_csv('data/missing.csv')
missing_prediction = experiment.predict(missing, config['feature_columns'])

prediction = pd.concat([ar_test_internal_avg, ar_test_external_avg, missing_prediction], axis=0, ignore_index=True)
prediction.to_csv(f'{experiment.exp_dir}/test_prediction.csv')
