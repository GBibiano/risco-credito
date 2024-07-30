from xgboost import XGBClassifier

# arquivos
from tuning import optuna_tuning

# carrega o XGBoost padrão ou com a tunagem de hiperparâmetros
def classificador(optuna: bool=False) -> XGBClassifier:
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