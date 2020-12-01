# Trust me, it looks way better in Jupyter :)

import os
import argparse
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from datetime import datetime
kernel_start = datetime.now()

from model import MoANeuralNetwork

cfg = argparse.Namespace()
cfg.n_seeds = 5
cfg.n_splits = 5
cfg.n_units = 512
cfg.lr_sched_factor = 0.4
cfg.batch_size_exp = 7


# Read data
input_dir = '../input/lish-moa'
train_features = pd.read_csv(os.path.join(input_dir, 'train_features.csv'))
train_targets_scored = pd.read_csv(os.path.join(input_dir, 'train_targets_scored.csv'))
test_features = pd.read_csv(os.path.join(input_dir, 'test_features.csv'))


# Feature cols (genes and cells)
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# Quantile transformation
for col in (GENES + CELLS):
    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
    train_len = len(train_features[col].values)
    test_len  = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(train_len, 1)
    transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, train_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(test_len, 1)).reshape(1, test_len)[0]
print(f'Train {train_features.shape} test {test_features.shape}')


# PCA on genes
n_comp = 500
comb_features = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
comb_features_pca = PCA(n_components=n_comp, random_state=42).fit_transform(comb_features[GENES])

train2 = comb_features_pca[:train_features.shape[0]]
test2  = comb_features_pca[-test_features.shape[0]:]
train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test2  = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)
test_features  = pd.concat((test_features, test2), axis=1)


# PCA on cells
n_comp = 50
comb_features = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
comb_features_pca = PCA(n_components=n_comp, random_state=42).fit_transform(comb_features[CELLS])

train2 = comb_features_pca[:train_features.shape[0]]
test2  = comb_features_pca[-test_features.shape[0]:]
train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test2  = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)
test_features  = pd.concat((test_features, test2), axis=1)

print(f'Train {train_features.shape} test {test_features.shape}')


# VarianceThreshold
var_thresh = VarianceThreshold(0.8)
data = train_features.append(test_features)
data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed  = data_transformed[-test_features.shape[0] : ]

train_features = pd.DataFrame(
    train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),
    columns=['sig_id','cp_type','cp_time','cp_dose'])
train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

test_features = pd.DataFrame(
    test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),
    columns=['sig_id','cp_type','cp_time','cp_dose'])
test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

print(f'Train {train_features.shape} test {test_features.shape}')


train_data = train_features.merge(train_targets_scored, on='sig_id')
train_data = train_data[train_data['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test_data  = test_features
print(f'Train {train_data.shape} test {test_data.shape}')


target_cols = train_data[train_targets_scored.columns]
target_cols = target_cols.drop('sig_id', axis=1).columns.values.tolist()


def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data

train_data = process_data(train_data)
test_data  = process_data(test_data)
feature_cols = [c for c in train_data.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id','cp_type']]


# Training & cross-validation
train_losses = []
valid_losses = []
saved_model_paths = []

for seed in range(1, cfg.n_seeds+1):

    pl.seed_everything(seed)
    folds = train_data.copy()
    k_fold = MultilabelStratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed * 111)
    for n, (train_index, valid_index) in enumerate(k_fold.split(folds, folds[target_cols])):

        df_train = folds.iloc[train_index]
        df_valid = folds.iloc[valid_index]

        train_x = df_train[feature_cols].values.astype(np.float32)
        train_y = df_train[target_cols].values.astype(np.float32)
        train_tensor = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y)) 

        valid_x = df_valid[feature_cols].values.astype(np.float32)
        valid_y = df_valid[target_cols].values.astype(np.float32)
        valid_tensor = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y)) 

        train_dl = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=2 ** cfg.batch_size_exp)  
        valid_dl = torch.utils.data.DataLoader(valid_tensor, shuffle=False, batch_size=2 ** cfg.batch_size_exp)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', dirpath='./', filename=f's{seed}f{n}')
        callbacks = [
            checkpoint_callback,
            pl.callbacks.EarlyStopping(monitor=f'val/loss', patience=20)]

        trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=callbacks)

        lit_module = MoANeuralNetwork(train_x.shape[1], train_y.shape[1], cfg)
        trainer.fit(lit_module, train_dl, valid_dl)

        fold_train_loss = trainer.logged_metrics['train/loss']
        fold_valid_loss = trainer.logged_metrics['val/loss']
        train_losses.append(fold_train_loss)
        valid_losses.append(fold_valid_loss)
        print(f'Train: {fold_train_loss}, valid: {fold_valid_loss} for seed {seed}/{cfg.n_seeds}, fold {n+1}/{cfg.n_splits}')    

        saved_model_paths.append(checkpoint_callback.best_model_path)


print(f'Train loss: {np.mean(np.array(train_losses)):.5f} +- {np.std(np.array(train_losses)):.5f}')
print(f'Valid loss: {np.mean(np.array(valid_losses)):.5f} +- {np.std(np.array(valid_losses)):.5f}')

print(saved_model_paths)


# Test set
test_tensor = torch.utils.data.TensorDataset(torch.tensor(test_data[feature_cols].values.astype(np.float32)))
test_dl     = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=1)
final_predictions = np.zeros((test_data.shape[0], len(target_cols)))

for model_path in saved_model_paths:
    saved_model = MoANeuralNetwork.load_from_checkpoint(
        model_path, input_size=len(feature_cols), output_size=len(target_cols), cfg=cfg)
    
    saved_model.eval()

    predictions = []
    for batch in test_dl:
        y_hat = saved_model(batch[0]).sigmoid().detach().squeeze().numpy()
        y_hat = np.clip(y_hat, 0.001, 0.999)
        predictions.append(y_hat)

    final_predictions += np.array(predictions)
    
final_predictions /= len(saved_model_paths)
print(final_predictions)


# Submission
df_submission = pd.read_csv(os.path.join(input_dir, 'sample_submission.csv'))
df_submission[target_cols] = final_predictions
df_submission.loc[test_data['cp_type']=='ctl_vehicle', target_cols] = 0.0
df_submission.to_csv('submission.csv', index=False)

# The end
kernel_stop = datetime.now()
print(kernel_stop - kernel_start)
