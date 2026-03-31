from fetch_data import fetch_all_stocks
from feature_engineering import engineer_features_all
from dataset_builder import build_dataset
from lstm_model import train_lstm

def price_prediction_task(
    if_fetch_data: bool = False, 
    if_feature_engineering: bool = False, 
    if_build_dataset: bool = False, 
    if_train_lstm: bool = False, 
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

    if (if_train_lstm):
        print("[TRAIN LSTM]\n")
        train_lstm()

if __name__ == "__main__":
    price_prediction_task(
        if_fetch_data=False, 
        if_feature_engineering=False, 
        if_build_dataset=False, 
        if_train_lstm=False
    )