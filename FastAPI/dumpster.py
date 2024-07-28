import datetime
import pickle
import pandas as pd
from xgboost import XGBClassifier

# arquivos
from tuning import optuna_tuning


# carrega um dataframe
def load_df() -> pd.DataFrame:
    dataset_original = "EDA_df.pkl"
    with open(dataset_original, "rb") as f:
        df_original = pickle.load(f)
    return df_original

# carrega um classificador existente no diretório
def load_classifier(modelo) -> pickle.load:
    # classificador remodelado sendo substituido
    with open(modelo, "rb") as f:
        classifier = pickle.load(f)
        return classifier

# salva um modelo em arquivo pickle
def dump_model(xgb_optuna_fit):
    # modelo que está sendo salvo
    with open(f'xgboostclassifier.pkl', mode='wb') as f:
        pickle.dump(xgb_optuna_fit, f)
        return '[INFO] Modelo salvo com sucesso.'

# carrega o XGBoost padrão ou com a tunagem de hiperparâmetros
def classificador(optuna: bool=False):
    if optuna:
        hiperparametros = optuna_tuning()
        xgbclassifier_optuna = XGBClassifier(
            n_estimators=hiperparametros['n_estimators'],
            max_depth=hiperparametros['max_depth'],
            learning_rate=hiperparametros['learning_rate'],
            gamma=hiperparametros['gamma'],
            random_state=47,
            )
        return xgbclassifier_optuna
    else:
        xgbclassifier = XGBClassifier(
            n_estimators=227,
            max_depth=5,
            learning_rate=0.4112354775681158,
            gamma=0.09972654226276964,
            random_state=47
            )
        return xgbclassifier