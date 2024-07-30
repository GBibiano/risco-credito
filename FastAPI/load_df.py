import pandas as pd
import pickle

# carrega o dataframe tratado após a Análise Exploratória de Dados
def load_df() -> pd.DataFrame:
    dataset_original = "EDA_df.pkl"
    with open(dataset_original, "rb") as f:
        df_original = pickle.load(f)
    return df_original