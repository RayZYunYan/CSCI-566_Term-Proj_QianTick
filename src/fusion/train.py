"""
Fusion Model Training
----------------------
Cross-Attention 融合模型，dual-task 输出（分类 + 回归）。

运行方式：
    python -m src.fusion.train --config configs/fusion_config.yaml
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
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.fusion.dataset import FusionDataset
from src.fusion.model import QianTickModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, alpha):
    model.eval()
    cls_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.HuberLoss()

    all_cls_logits, all_reg_preds = [], []
    all_cls_labels, all_reg_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            news  = batch["news_feat"].to(device)
            price = batch["price_feat"].to(device)
            y_cls = batch["label_cls"].to(device)
            y_reg = batch["label_reg"].to(device)

            cls_logit, reg_pred = model(news, price)
            loss = alpha * cls_criterion(cls_logit, y_cls) + \
                   (1 - alpha) * reg_criterion(reg_pred, y_reg)
            total_loss += loss.item()

            all_cls_logits.append(cls_logit.cpu())
            all_reg_preds.append(reg_pred.cpu())
            all_cls_labels.append(y_cls.cpu())
            all_reg_labels.append(y_reg.cpu())

    cls_logits = torch.cat(all_cls_logits).numpy()
    cls_labels = torch.cat(all_cls_labels).numpy()
    reg_preds  = torch.cat(all_reg_preds).numpy()
    reg_labels = torch.cat(all_reg_labels).numpy()

    cls_probs = 1 / (1 + np.exp(-cls_logits))
    cls_preds = (cls_probs >= 0.5).astype(int)

    metrics = {
        "loss":      total_loss / max(1, len(loader)),
        "accuracy":  float(accuracy_score(cls_labels, cls_preds)),
        "roc_auc":   float(roc_auc_score(cls_labels, cls_probs)) if len(np.unique(cls_labels)) > 1 else float("nan"),
        "reg_mae":   float(np.mean(np.abs(reg_preds - reg_labels))),
        "reg_rmse":  float(np.sqrt(np.mean((reg_preds - reg_labels) ** 2))),
    }
    return metrics


def train(config: dict):
    data_cfg  = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    out_cfg   = config["output"]

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(out_cfg["checkpoint"]), exist_ok=True)

    train_ds = FusionDataset(split="train",
                             news_features_path=data_cfg["news_features"],
                             price_features_dir=data_cfg["price_features"],
                             price_csv_path=data_cfg["price_csv"])
    val_ds   = FusionDataset(split="val",
                             news_features_path=data_cfg["news_features"],
                             price_features_dir=data_cfg["price_features"],
                             price_csv_path=data_cfg["price_csv"])
    test_ds  = FusionDataset(split="test",
                             news_features_path=data_cfg["news_features"],
                             price_features_dir=data_cfg["price_features"],
                             price_csv_path=data_cfg["price_csv"])

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = QianTickModel(
        news_dim=model_cfg["news_dim"],
        price_dim=model_cfg["price_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        nhead=model_cfg["nhead"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    cls_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.HuberLoss()
    alpha = train_cfg["alpha"]

    best_val_loss = float("inf")
    patience_counter = 0
    logs = []

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            news  = batch["news_feat"].to(device)
            price = batch["price_feat"].to(device)
            y_cls = batch["label_cls"].to(device)
            y_reg = batch["label_reg"].to(device)

            optimizer.zero_grad(set_to_none=True)
            cls_logit, reg_pred = model(news, price)
            loss = alpha * cls_criterion(cls_logit, y_cls) + \
                   (1 - alpha) * reg_criterion(reg_pred, y_reg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, alpha)

        logs.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss / max(1, len(train_loader)),
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
            "val_auc":    val_metrics["roc_auc"],
        })

        print(f"Epoch {epoch+1:3d} | "
              f"train_loss={logs[-1]['train_loss']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_auc={val_metrics['roc_auc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), out_cfg["checkpoint"])
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    # Test evaluation
    model.load_state_dict(torch.load(out_cfg["checkpoint"], map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, alpha)
    print(f"\nTest | acc={test_metrics['accuracy']:.4f} | "
          f"auc={test_metrics['roc_auc']:.4f} | "
          f"mae={test_metrics['reg_mae']:.4f} | "
          f"rmse={test_metrics['reg_rmse']:.4f}")

    with open(out_cfg["metrics"], "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in test_metrics.items() if k != "loss"}, f, indent=2)

    pd.DataFrame(logs).to_csv(
        out_cfg["metrics"].replace("metrics.json", "training_log.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
