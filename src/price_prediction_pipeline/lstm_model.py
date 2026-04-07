import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import optuna # For Bayesian hyperparameter optimization
from dataset_builder import load_dataset
from metrics import compute_classification_metrics, save_metrics

PIPELINE_ROOT = Path(__file__).resolve().parent
ACTIVATION_REGISTRY = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if (num_layers > 1) else 0.0
        )
        self.feature_dropout = nn.Dropout(dropout)
        act_cls = ACTIVATION_REGISTRY.get(activation, nn.GELU)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            act_cls(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim // 2, 1)
        )

    def _encode(self, x):
        _, (h_n, _) = self.lstm(x)
        features = h_n[-1]
        features = self.feature_dropout(features)
        return features

    def forward(self, x):
        logits = self.classifier(self._encode(x))
        return logits.squeeze(-1)

    def extract_features(self, x):
        return self._encode(x)

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module):
        if (self.best_score is None or score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            # Save the current model's parameters in their entirety, designating them as the best model state to date.
            # This will cover the previous best model's parameters, ensuring that we always have the most up-to-date best model state saved.
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if (self.counter >= self.patience):
                self.should_stop = True

    def restore_best(self, model: nn.Module):
        if (self.best_state is not None):
            model.load_state_dict(self.best_state)

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def auto_device():
    # Automatically select the best available device GPU (CUDA) > MPS (Apple Silicon) > CPU.
    if (torch.cuda.is_available()):
        return "cuda"
    # MPS (Metal Performance Shaders) is Apple's GPU framework for accelerating machine 
    # learning on macOS devices with Apple Silicon.
    elif (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        # X_batch is the data for the current batch, and y_batch is the corresponding labels.
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Add the loss for the current batch, multiplied by the number of samples in the batch, to the total loss.
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.numpy())
    # Return (y_prob, y_true) as numpy arrays.
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)

def val_roc_auc(model, loader, device):
    # ROC-AUC on a validation set used inside the training loop.
    y_prob, y_true = get_predictions(model, loader, device)
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.0

def train_full(model, train_loader, val_loader, lr, weight_decay, max_epochs, patience, device, pos_weight=None, trial=None, verbose=True):
    # Use class-weighted loss to prevent prediction bias toward the majority class.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ReduceLROnPlateau scheduler will reduce the learning rate by a factor of 0.5 if the validation 
    # ROC-AUC does not improve for a certain number of epochs (patience // 3).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 3), min_lr=1e-6
    )
    es = EarlyStopping(patience=patience)
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = val_roc_auc(model, val_loader, device)
        scheduler.step(val_auc)
        es.step(val_auc, model)
        if (verbose and (epoch + 1) % 10 == 0):
            print(f"epoch {epoch+1:3d}: train_loss={train_loss:.4f} and val_auc={val_auc:.4f}.")
        if (trial is not None):
            trial.report(val_auc, epoch)
            if (trial.should_prune()):
                raise optuna.TrialPruned()
        # If the early stopping condition is met, break out of the training loop to prevent overfitting and save computational resources.
        if (es.should_stop):
            break
    es.restore_best(model)
    return es.best_score

def load_lstm_model(model_path: str | Path, device: str = "cpu"):
    # Load saved best LSTM checkpoint and return the model in eval mode on the specified device.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    p = ckpt["params"]
    model = LSTMClassifier(ckpt["input_dim"], p["hidden_dim"], p["num_layers"], p["dropout"], p["activation"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model

@torch.no_grad()
def predict_lstm(model_path: str | Path, data_dir: str | Path | None = None, split: str = "test", device: str = "cpu", batch_size: int = 512):
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    model = load_lstm_model(model_path, device)
    splits, _ = load_dataset(data_dir)
    loader = make_loader(splits[split]["X"], splits[split]["y"], batch_size, shuffle=False)
    y_prob, y_true = get_predictions(model, loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    # Return predicted probabilities, predicted labels, and true labels as numpy arrays.
    return y_prob, y_pred, y_true

@torch.no_grad()
def extract_lstm_features(model_path: str | Path, data_dir: str | Path | None = None, split: str = "test", device: str = "cpu", batch_size: int = 512):
    # Load a saved LSTM and return feature vectors of shape (N, hidden_dim).
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    model = load_lstm_model(model_path, device)
    splits, _ = load_dataset(data_dir)
    X = splits[split]["X"]
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks = []
    for (X_batch,) in loader:
        feat = model.extract_features(X_batch.to(device))
        chunks.append(feat.cpu().numpy())
    # First, take the trained LSTM and have it process all the samples. However, do not require it to provide a final verdict—such 
    # as "up" or "down"—but instead extract only the "internal representation" formed within its "mind" after it has 
    # finished reading the sequence. Using for fusion layer.
    return np.concatenate(chunks, axis=0)

def train_lstm(
    data_dir: str | Path | None = None, 
    output_dir: str | Path | None = None, 
    n_trials: int = 50, 
    max_epochs_per_trial: int = 50, 
    max_epochs_final: int = 200, 
    patience: int = 15, 
    seed: int = 68
):
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    if (output_dir is None):
        output_dir = PIPELINE_ROOT / "outputs"
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    model_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    splits, _ = load_dataset(data_dir)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    input_dim = X_train.shape[2]

    # Compute class weight to prevent the model from biasing toward the majority class.
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    
    # Optuna search
    # Only show warnings and errors from Optuna, not info/debug logs, to reduce clutter during hyperparameter optimization.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        hp = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]), 
            "num_layers": trial.suggest_int("num_layers", 1, 3), 
            "dropout": trial.suggest_float("dropout", 0.1, 0.5), 
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True), 
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]), 
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), 
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        }
        model = LSTMClassifier(input_dim, hp["hidden_dim"], hp["num_layers"], hp["dropout"], hp["activation"]).to(device)
        train_loader = make_loader(X_train, y_train, hp["batch_size"], shuffle=True)
        val_loader = make_loader(X_val, y_val, hp["batch_size"], shuffle=False)
        return train_full(
            model, train_loader, val_loader, hp["lr"], hp["weight_decay"], max_epochs_per_trial, patience, device, pos_weight=pos_weight, trial=trial
        )

    # Create an Optuna study for hyperparameter optimization, specifying that we want to maximize the validation ROC-AUC.
    # Use the TPE sampler for efficient hyperparameter search, and the Median pruner to stop unpromising trials early 
    # based on their intermediate validation ROC-AUC scores. Compare the performance of the current trial at a specific 
    # epoch against the median performance of historical trials at the same epoch.
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=seed), 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f"\nBest val ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best, indent=4)}")

    # Save best hyperparams
    with open(model_dir / "lstm_best_params.json", "w") as f:
        json.dump({"best_val_roc_auc": study.best_value, **best}, f, indent=2)

    # Retrain final model with best params
    final_model = LSTMClassifier(input_dim, best["hidden_dim"], best["num_layers"], best["dropout"], best["activation"]).to(device)
    train_loader = make_loader(X_train, y_train, best["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, best["batch_size"], shuffle=False)
    train_full(final_model, train_loader, val_loader, best["lr"], best["weight_decay"], max_epochs_final, patience, device, pos_weight=pos_weight, verbose=True)

    # Save checkpoint
    model_path = model_dir / "lstm_best.pt"
    torch.save({"model_state_dict": final_model.state_dict(), "params": best, "input_dim": input_dim}, model_path)

    # Evaluate on test set.
    test_loader = make_loader(X_test, y_test, best["batch_size"], shuffle=False)
    y_prob, y_true = get_predictions(final_model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    test_metrics["best_params"] = best
    test_metrics["best_val_roc_auc"] = study.best_value

    metrics_path = metrics_dir / "lstm_metrics.json"
    save_metrics(test_metrics, metrics_path)
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")