from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dataset_builder import load_dataset
from lstm_model import extract_lstm_features, predict_lstm
from gru_model import extract_gru_features, predict_gru
from tcn_model import extract_tcn_features, predict_tcn

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent.parent

def auto_device():
    if (torch.cuda.is_available()):
        return "cuda"
    elif (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"

def export_price_features(
    data_dir: str | Path | None = None, 
    model_dir: str | Path | None = None, 
    output_dir: str | Path | None = None, 
    split: str = "test", 
    batch_size: int = 512
):
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    if (model_dir is None):
        model_dir = REPO_ROOT / "outputs" / "price" / "models"
    if (output_dir is None):
        output_dir = REPO_ROOT / "export_to_fusion"
    data_dir, model_dir, output_dir = Path(data_dir), Path(model_dir), Path(output_dir)
    features_dir = output_dir / "features"
    predictions_dir = output_dir / "predictions"
    features_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device()

    # Load all splits from the dataset directory.
    splits, _ = load_dataset(data_dir)

    # Define which models to export.
    model_configs = {}
    lstm_path = model_dir / "lstm_best.pt"
    if (lstm_path.exists()):
        model_configs["lstm"] = {
            "model_path": lstm_path,
            "extract_fn": extract_lstm_features,
            "predict_fn": predict_lstm
        }
    gru_path = model_dir / "gru_best.pt"
    if (gru_path.exists()):
        model_configs["gru"] = {
            "model_path": gru_path,
            "extract_fn": extract_gru_features,
            "predict_fn": predict_gru
        }
    tcn_path = model_dir / "tcn_best.pt"
    if (tcn_path.exists()):
        model_configs["tcn"] = {
            "model_path": tcn_path,
            "extract_fn": extract_tcn_features,
            "predict_fn": predict_tcn
        }

    if (not model_configs):
        raise FileNotFoundError(f"No trained model checkpoints found in {model_dir}")

    # Extract features and predictions for all splits, then concatenate.
    all_features = {name: [] for name in model_configs}
    all_meta_pieces = []

    for s in ("train", "val", "test"):
        meta_s = splits[s]["meta"].copy()
        meta_s["true_label"] = splits[s]["y"]
        meta_s["split"] = s

        for name, cfg in model_configs.items():
            feats = cfg["extract_fn"](cfg["model_path"], data_dir, s, device, batch_size)
            all_features[name].append(feats)
            y_pred, _ = cfg["predict_fn"](cfg["model_path"], data_dir, s, device, batch_size)
            meta_s[f"pred_return_{name}"] = y_pred

        all_meta_pieces.append(meta_s)

    # Save concatenated feature tensors (train+val+test in order).
    for name in model_configs:
        feat_all = np.concatenate(all_features[name], axis=0)
        torch.save(torch.from_numpy(feat_all), features_dir / f"{name}_features.pt")

    # Save unified metadata CSV with split column.
    all_meta_df = pd.concat(all_meta_pieces, ignore_index=True)
    base_cols = ["ticker", "as_of_date", "target_date", "split", "true_label"]
    pred_cols = [c for c in all_meta_df.columns if c.startswith("pred_return_")]
    all_meta_df = all_meta_df[base_cols + pred_cols]
    all_meta_df["as_of_date"] = all_meta_df["as_of_date"].astype(str).str[:10]
    all_meta_df["target_date"] = all_meta_df["target_date"].astype(str).str[:10]
    csv_path = predictions_dir / "price_model_outputs.csv"
    all_meta_df.to_csv(csv_path, index=False)

    # Export labels.parquet covering ALL splits for the news pipeline.
    # Must include train/val/test so NewsFeatureDataset can filter each split correctly.
    all_pieces = []
    for s in ("train", "val", "test"):
        m = splits[s]["meta"][["ticker", "target_date"]].copy()
        m["label"] = splits[s]["y"]
        all_pieces.append(m)
    all_labels_df = pd.concat(all_pieces, ignore_index=True)
    all_labels_df = all_labels_df.rename(columns={"target_date": "date"})
    all_labels_df["date"] = all_labels_df["date"].astype(str).str[:10]
    labels_path = REPO_ROOT / "data" / "features" / "labels.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    all_labels_df.to_parquet(labels_path, index=False)
    print(f"Exported labels.parquet ({len(all_labels_df)} rows, all splits) to {labels_path}")