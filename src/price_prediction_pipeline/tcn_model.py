from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from dataset_builder import load_dataset
from metrics import compute_classification_metrics, save_metrics

PIPELINE_ROOT = Path(__file__).resolve().parent
ACTIVATION_REGISTRY = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}

class CausalConv1d(nn.Module):
    """
    Left-padded causal Conv1d wrapper.
    Pads on the left with (kernel_size-1)*dilation before applying Conv1d so
    that the output at time t depends only on <= t inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        if self.left_pad > 0:
            # add left padding of zeros
            x = F.pad(x, (self.left_pad, 0))
        
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    single block of TCN with residual connection. Output has same shape as input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, activation="gelu"):
        super().__init__()
        act_cls = ACTIVATION_REGISTRY.get(activation, nn.GELU)
        # two causal conv layers per block
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = act_cls()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = act_cls()
        self.drop2 = nn.Dropout(dropout)

        # If in_channels != out_channels, we need a 1x1 conv to match dimensions for the residual connection.
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        # connect residual and check if downsample is needed
        res = x if self.downsample is None else self.downsample(x)

        return out + res


class TCNClassifier(nn.Module):
    """
    TCN-based classifier for binary classification. Outputs a single logit per input sequence.
    """
    def __init__(self, input_dim: int, num_channels: int, num_layers: int, kernel_size: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        # Build stack of TCN blocks. Each block doubles dilation (1,2,4,...)
        layers = []
        in_ch = input_dim

        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = num_channels

            # add TCN block based on dilation
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout, activation))
            in_ch = out_ch
        
        # combine all blocks into a single sequential module
        self.tcn = nn.Sequential(*layers)

        self.feature_dropout = nn.Dropout(dropout)
        act_cls = ACTIVATION_REGISTRY.get(activation, nn.GELU)
        
        # Classifier head mirrors LSTM/GRU implementations
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, max(1, num_channels // 2)),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(max(1, num_channels // 2), 1)
        )

    def _encode(self, x: torch.Tensor):
        """
        encodes input sequences into features using the TCN layers. Returns the features before the classifier head.
        """
        # x: (batch, seq_len, features) -> transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        
        # take last time-step
        features = out[:, :, -1]
        # apply feature dropout before classifier
        features = self.feature_dropout(features)

        return features

    def forward(self, x: torch.Tensor):
        logits = self.classifier(self._encode(x))
        return logits.squeeze(-1)

    def extract_features(self, x: torch.Tensor):
        return self._encode(x)


class EarlyStopping:
    """
    early stopping utility that tracks the best score and corresponding model state, and signals when to stop.
    """
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
            # Save a copy of the model state dict on CPU to avoid GPU memory issues.
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module):
        if (self.best_state is not None):
            model.load_state_dict(self.best_state)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Converts numpy arrays to a PyTorch DataLoader."""
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
    """
    Trains the model for one epoch and returns the average loss.
    """
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
        
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        # Clip gradients to prevent exploding gradients, which can be an issue with TCNs due to their deep structure.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Gets predicted probabilities and true labels from the model for the entire dataset in the loader.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device).float()
        probs = torch.sigmoid(model(X_batch)).detach().cpu().numpy()
        
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())
    
    if len(all_probs) == 0:
        return np.array([]), np.array([])
    
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def val_roc_auc(model, loader, device):
    """Computes ROC-AUC on the validation set."""
    y_prob, y_true = get_predictions(model, loader, device)
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.0


def train_full(model, train_loader, val_loader, lr, weight_decay, max_epochs, patience, device, pos_weight=None, trial=None, verbose=True):
    """
    Trains the model for the full number of epochs with early stopping and returns the best validation ROC-AUC.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
            print(f"Epoch {epoch+1}/{max_epochs} — train_loss={train_loss:.4f} val_auc={val_auc:.4f}")
        
        # Report intermediate results to Optuna for pruning
        if (trial is not None):
            trial.report(val_auc, epoch)

            # Check if the trial should be pruned based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if (es.should_stop):
            break
    
    es.restore_best(model)
    
    return es.best_score


def load_tcn_model(model_path: str | Path, device: str = "cpu"):
    """
    Loads a trained TCN model from the specified path and returns it in evaluation mode.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    p = ckpt["params"]
    model = TCNClassifier(ckpt["input_dim"], p["num_channels"], p["num_layers"], p["kernel_size"], p["dropout"], p.get("activation", "gelu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    
    return model


@torch.no_grad()
def predict_tcn(model_path: str | Path, data_dir: str | Path | None = None, split: str = "test", device: str = "cpu", batch_size: int = 512):
    """
    Loads a trained TCN model and returns predicted probabilities, binary predictions, and true labels for the specified dataset split.
    """
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    
    model = load_tcn_model(model_path, device)
    splits, _ = load_dataset(data_dir)
    loader = make_loader(splits[split]["X"], splits[split]["y"], batch_size, shuffle=False)
    y_prob, y_true = get_predictions(model, loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_prob, y_pred, y_true


@torch.no_grad()
def extract_tcn_features(model_path: str | Path, data_dir: str | Path | None = None, split: str = "test", device: str = "cpu", batch_size: int = 512):
    """
    Loads a trained TCN model and returns the extracted features from the TCN layers for the specified dataset split.
    """
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    
    model = load_tcn_model(model_path, device)
    splits, _ = load_dataset(data_dir)
    X = splits[split]["X"]
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks = []
    
    for (X_batch,) in loader:
        feats = model.extract_features(X_batch.to(device).float()).detach().cpu().numpy()
        chunks.append(feats)
    
    return np.concatenate(chunks, axis=0)


def train_tcn(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_trials: int = 50,
    max_epochs_per_trial: int = 50,
    max_epochs_final: int = 200,
    patience: int = 15,
    seed: int = 68
):
    """
    Trains a TCN model with Optuna hyperparameter tuning and saves the best model and metrics.
    """
    if (data_dir is None):
        raise ValueError("data_dir must be provided explicitly. Use main_pipeline.py to select the dataset from feature_preset and lookback.")
    
    if (output_dir is None):
        output_dir = PIPELINE_ROOT / "outputs"
    
    # Set up directories, device, and random seeds for reproducibility.
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    model_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset splits and prepare training/validation/test sets.
    splits, _ = load_dataset(data_dir)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    input_dim = X_train.shape[2]

    # Compute class imbalance ratio for weighted loss to help with convergence, especially if the dataset is imbalanced.
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    
    # Suppress Optuna logging except for warnings and errors to keep the output clean during hyperparameter tuning.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Objective function for Optuna hyperparameter optimization. Trains a TCN model with the given hyperparameters and returns the best validation ROC-AUC."""
        params = {
            "num_channels": trial.suggest_categorical("num_channels", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 5),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]) 
        }

        # Initialize a TCN model with the suggested hyperparameters, create DataLoaders for training and validation sets, and train the model while tracking the best validation ROC-AUC for this trial.
        model = TCNClassifier(input_dim, params["num_channels"], params["num_layers"], params["kernel_size"], params["dropout"], params["activation"]).to(device)
        train_loader = make_loader(X_train, y_train, params["batch_size"], shuffle=True)
        val_loader = make_loader(X_val, y_val, params["batch_size"], shuffle=False)
        best_val = train_full(model, train_loader, val_loader, params["lr"], params["weight_decay"], max_epochs_per_trial, patience, device, pos_weight=pos_weight, trial=trial, verbose=False)
        
        return best_val

    # Run Optuna hyperparameter optimization with the defined objective function, using TPE sampler and median pruner for efficient search.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f"\nBest val ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best, indent=4)}")

    # Save the best hyperparameters and validation score to a JSON file for record-keeping and reproducibility.
    with open(model_dir / "tcn_best_params.json", "w") as f:
        json.dump({"best_val_roc_auc": study.best_value, **best}, f, indent=4)

    # Train the final model with the best hyperparameters on the combined training and validation sets, using early stopping to prevent overfitting, and save the best model checkpoint.
    final_model = TCNClassifier(input_dim, best["num_channels"], best["num_layers"], best["kernel_size"], best["dropout"], best["activation"]).to(device)
    train_loader = make_loader(X_train, y_train, best["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, best["batch_size"], shuffle=False)
    train_full(final_model, train_loader, val_loader, best["lr"], best["weight_decay"], max_epochs_final, patience, device, pos_weight=pos_weight, verbose=True)

    # Save the final model checkpoint along with the best hyperparameters and input dimension for later loading and inference.
    model_path = model_dir / "tcn_best.pt"
    torch.save({"model_state_dict": final_model.state_dict(), "params": best, "input_dim": input_dim}, model_path)

    # Evaluate the final model on the test set, compute classification metrics, and save them to a JSON file for analysis.
    test_loader = make_loader(X_test, y_test, best["batch_size"], shuffle=False)
    y_prob, y_true = get_predictions(final_model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    test_metrics["best_params"] = best
    test_metrics["best_val_roc_auc"] = study.best_value

    # Save test metrics to a JSON file and print key metrics to the console for quick reference.
    metrics_path = metrics_dir / "tcn_metrics.json"
    save_metrics(test_metrics, metrics_path)
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")