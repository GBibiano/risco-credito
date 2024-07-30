import os
import pickle
from xgboost import XGBClassifier

# salva um modelo em arquivo pickle
def dump_model(xgb_optuna_fit) -> XGBClassifier:
    # Define o caminho completo para o arquivo na pasta 'remodel'
    model_path = os.path.join('model_report', 'xgboostclassifier.pkl')
    
    # modelo que está sendo salvo
    with open(model_path, mode='wb') as f:
        pickle.dump(xgb_optuna_fit, f)
        return '[INFO] Modelo salvo com sucesso.'