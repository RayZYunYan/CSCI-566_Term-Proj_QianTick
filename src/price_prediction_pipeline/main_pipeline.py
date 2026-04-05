from fetch_data import fetch_all_stocks
from feature_engineering import engineer_features_all
from dataset_builder import build_dataset
from lstm_model import train_lstm
from gru_model import train_gru
from fusion_interface import export_price_features

def data_preprocessing(
    if_fetch_data: bool = False, 
    if_feature_engineering: bool = False, 
    if_build_dataset: bool = False, 
    feature_preset: str = "full", 
    lookback: int = 30
):
    if (if_fetch_data):
        print("[FETCH DATA]\n")
        fetch_all_stocks()

    if (if_feature_engineering):
        print("[FEATURE ENGINEERING]\n")
        engineer_features_all()

    if (if_build_dataset):
        print("[BUILD DATASET]\n")
        build_dataset(feature_preset=feature_preset, lookback=lookback)
    
    print("[DATA PREPROCESSING COMPLETE]\n")

def price_prediction_task(
    if_train_lstm: bool = False, 
    if_train_gru: bool = False, 
    if_train_tcn: bool = False, 
    data_dir: str | None = None, 
    output_dir: str | None = None, 
    n_trials: int = 50, 
    max_epochs_per_trial: int = 60, 
    max_epochs_final: int = 300, 
    patience: int = 20
):
    if (if_train_lstm):
        print("[TRAIN LSTM]\n")
        train_lstm(
            data_dir=data_dir, 
            output_dir=output_dir, 
            n_trials=n_trials,
            max_epochs_per_trial=max_epochs_per_trial,
            max_epochs_final=max_epochs_final,
            patience=patience
        )

    if (if_train_gru):
        print("[TRAIN GRU]\n")
        train_gru(
            data_dir=data_dir, 
            output_dir=output_dir, 
            n_trials=n_trials,
            max_epochs_per_trial=max_epochs_per_trial,
            max_epochs_final=max_epochs_final,
            patience=patience
        )

    if (if_train_tcn):
        print("[TRAIN TCN]\n")
        # train_tcn(
        #     n_trials=n_trials,
        #     max_epochs_per_trial=max_epochs_per_trial,
        #     max_epochs_final=max_epochs_final,
        #     patience=patience
        # )

    print("[PRICE PREDICTION TASK COMPLETE]\n")

def export_features_for_fusion(if_export: bool = False, split: str = "test"):
    if (if_export):
        print("[EXPORT FEATURES FOR FUSION]\n")
        export_price_features(split=split)
        print("[FUSION EXPORT COMPLETE]\n")

if __name__ == "__main__":
    # For feature_preset, options include: "full", "relative_only".
    data_preprocessing(
        if_fetch_data=True, 
        if_feature_engineering=True, 
        if_build_dataset=True, 
        feature_preset="relative_only", 
        lookback=30
    )
    price_prediction_task(
        if_train_lstm=True, 
        if_train_gru=True, 
        if_train_tcn=True, 
        n_trials=50, 
        max_epochs_per_trial=60, 
        max_epochs_final=300, 
        patience=20
    )
    export_features_for_fusion(if_export=True)