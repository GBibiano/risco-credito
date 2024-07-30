import pandas as pd
import pickle

# carrega um dataframe
def load_df() -> pd.DataFrame:
    dataset_original = "EDA_df.pkl"
    with open(dataset_original, "rb") as f:
        df_original = pickle.load(f)
    return df_original