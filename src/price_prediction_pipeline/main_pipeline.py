from fetch_data import fetch_all_stocks
from feature_engineering import engineer_features_all
from dataset_builder import build_dataset
from lstm_model import train_lstm
from gru_model import train_gru

def data_preprocessing(
    if_fetch_data: bool = False, 
    if_feature_engineering: bool = False, 
    if_build_dataset: bool = False, 
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
        build_dataset(lookback=lookback)
    
    print("[DATA PREPROCESSING COMPLETE]\n")

def price_prediction_task(
    if_train_lstm: bool = False, 
    if_train_gru: bool = False, 
    if_train_tcn: bool = False, 
    n_trials: int = 35, 
    max_epochs_per_trial: int = 40, 
    max_epochs_final: int = 150, 
    patience: int = 12
):
    if (if_train_lstm):
        print("[TRAIN LSTM]\n")
        train_lstm(
            n_trials=n_trials, 
            max_epochs_per_trial=max_epochs_per_trial, 
            max_epochs_final=max_epochs_final, 
            patience=patience
        )

    if (if_train_gru):
        print("[TRAIN GRU]\n")
        train_gru(
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

if __name__ == "__main__":
    data_preprocessing(if_fetch_data=False, if_feature_engineering=False, if_build_dataset=False)
    price_prediction_task(
        if_train_lstm=True, 
        if_train_gru=True, 
        if_train_tcn=False, 
        n_trials=50, 
        max_epochs_per_trial=60, 
        max_epochs_final=300, 
        patience=20
    )