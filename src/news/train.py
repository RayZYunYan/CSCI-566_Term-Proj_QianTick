"""
News Unimodal Baseline Training
--------------------------------
训练仅用新闻特征的 baseline 分类器，用于消融实验。

运行方式：
    python -m src.news.train --config configs/news_config.yaml
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.news.dataset import NewsFeatureDataset
from src.news.model import NewsUnimodalBaseline

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float()

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    return {
        'loss': total_loss / max(1, len(loader)),
        'accuracy': accuracy_score(all_labels, preds),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'f1': f1_score(all_labels, preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan'),
        'probs': all_probs,
        'preds': preds,
        'labels': all_labels,
    }

def train(config: dict):
    data_cfg     = config['data']
    model_cfg    = config['model']
    train_cfg    = config['training']
    out_cfg      = config['output']

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_cfg['baseline_dir'], exist_ok=True)

    train_ds = NewsFeatureDataset(
        features_path=data_cfg['news_features'],
        labels_path=data_cfg['labels'],
        split='train',
    )
    val_ds = NewsFeatureDataset(
        features_path=data_cfg['news_features'],
        labels_path=data_cfg['labels'],
        split='val',
    )
    test_ds = NewsFeatureDataset(
        features_path=data_cfg['news_features'],
        labels_path=data_cfg['labels'],
        split='test',
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = NewsUnimodalBaseline(
        input_dim=model_cfg['feature_dim'],
        hidden_dim=model_cfg['hidden_dim'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    logs = []

    for epoch in range(train_cfg['epochs']):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)

        logs.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / max(1, len(train_loader)),
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss={logs[-1]['train_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), out_cfg['checkpoint'])
        else:
            patience_counter += 1

        if patience_counter >= train_cfg['early_stopping_patience']:
            print('Early stopping triggered.')
            break

    model.load_state_dict(torch.load(out_cfg['checkpoint'], map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    with open(out_cfg['metrics'], 'w', encoding='utf-8') as f:
        json.dump(
            {k: v for k, v in test_metrics.items() if k not in ('probs', 'preds', 'labels')},
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(logs).to_csv(out_cfg['training_log'], index=False)

    pred_df = pd.DataFrame({
        'ticker': test_ds.df['ticker'].values,
        'date': test_ds.df['date'].astype(str).values,
        'prob': test_metrics['probs'],
        'pred_label': test_metrics['preds'],
        'true_label': test_metrics['labels'],
    })

    try:
        pred_df.to_parquet(out_cfg['predictions'], index=False)
    except Exception:
        pred_df.to_csv(out_cfg['predictions'].replace('.parquet', '.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/news_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
