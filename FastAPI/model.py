from sklearn.pipeline import Pipeline

# arquivos
from train_model import treinar_modelo
from classificador import classificador
from data_processor import new_data_processing_
from dump_model import dump_model

# Executa o algoritmo de ML nos dados, retorna o classificador
def model(new_data: dict, tuning: bool=False) -> Pipeline:
    # chama o processamento de dados e atribui novas instâncias à uma variável
    df = new_data_processing_(new_data)

    X = df.drop(columns = ["status_emprestimo"], axis = 1)
    y = df['status_emprestimo']

    if tuning:
        xgbclassifier = classificador(optuna=True)
    else:
        xgbclassifier = classificador()
    
    features_RFE_xgboost = ['idade_cliente', 'renda_cliente', 'posse_residencia_cliente', 'tempo_emprego_cliente', 'finalidade_emprestimo', 'nota_emprestimo', 'taxa_juros_emprestimo', 'percentual_renda_emprestimo', 'retorno_emprestimo', 'ratio_renda_emp', 'media_valemp_nota', 'media_valemp_finalidade', 'std_valemp_residencia', 'ratio_emprego_renda']
    xgb_treinado, _, _, _ = treinar_modelo(xgbclassifier, X[features_RFE_xgboost], y)

    # retorna o classificador em pickle; e
    # acurácia, classification_report, confusion_matrix em pickle
    dump_model(xgb_treinado)
    
    print('[INFO] model.py: ', X[features_RFE_xgboost].loc[:, :])
    
    return xgb_treinado