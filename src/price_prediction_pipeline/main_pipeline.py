from fetch_data import fetch_all_stocks
from feature_engineering import engineer_features_all
from dataset_builder import build_dataset
from lstm_model import train_lstm
from gru_model import train_gru
from tech_indicator_mlp import train_tech_mlp

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
    if_train_tech_mlp: bool = False
):
    if (if_train_lstm):
        print("[TRAIN LSTM]\n")
        train_lstm()

    if (if_train_gru):
        print("[TRAIN GRU]\n")
        train_gru()

    if (if_train_tcn):
        print("[TRAIN TCN]\n")
        # train_tcn()

    if (if_train_tech_mlp):
        print("[TRAIN TECH INDICATOR MLP]\n")
        train_tech_mlp()

    print("[PRICE PREDICTION TASK COMPLETE]\n")

if __name__ == "__main__":
    data_preprocessing(if_fetch_data=False, if_feature_engineering=False, if_build_dataset=False)
    price_prediction_task(if_train_lstm=False, if_train_gru=False, if_train_tcn=False, if_train_tech_mlp=False)