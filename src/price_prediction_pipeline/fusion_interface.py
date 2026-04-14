from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dataset_builder import load_dataset
from lstm_model import extract_lstm_features, predict_lstm
from gru_model import extract_gru_features, predict_gru
from tcn_model import extract_tcn_features, predict_tcn

PIPELINE_ROOT = Path(__file__).resolve().parent

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
        model_dir = PIPELINE_ROOT / "outputs" / "models"
    if (output_dir is None):
        output_dir = PIPELINE_ROOT / "export_to_fusion"
    data_dir, model_dir, output_dir = Path(data_dir), Path(model_dir), Path(output_dir)
    features_dir = output_dir / "features"
    predictions_dir = output_dir / "predictions"
    features_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device()

    # Load metadata for the target split (ticker, as_of_date, target_date).
    splits, _ = load_dataset(data_dir)
    meta_df = splits[split]["meta"].copy()
    y_true = splits[split]["y"]

    # Define which models to export. Each entry maps a model name to its checkpoint path and extraction functions.
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

    # Extract feature vectors and predicted returns from each available model.
    for name, cfg in model_configs.items():
        features = cfg["extract_fn"](cfg["model_path"], data_dir, split, device, batch_size)
        feat_path = features_dir / f"{name}_features.pt"
        torch.save(torch.from_numpy(features), feat_path)

        y_pred, _ = cfg["predict_fn"](cfg["model_path"], data_dir, split, device, batch_size)
        meta_df[f"pred_return_{name}"] = y_pred

    # Attach true return labels to the metadata DataFrame.
    meta_df["true_label"] = y_true

    # Reorder columns: metadata first, then true_label, then prediction columns.
    base_cols = ["ticker", "as_of_date", "target_date", "true_label"]
    pred_cols = [c for c in meta_df.columns if (c.startswith("pred_return_"))]
    meta_df = meta_df[base_cols + pred_cols]

    # Save the combined metadata and predictions CSV for the fusion layer.
    csv_path = predictions_dir / "price_model_outputs.csv"
    meta_df.to_csv(csv_path, index=False)

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
    labels_path = Path(output_dir).parent.parent.parent.parent / "data" / "features" / "labels.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    all_labels_df.to_parquet(labels_path, index=False)
    print(f"Exported labels.parquet ({len(all_labels_df)} rows, all splits) to {labels_path}")