# Tunagem de Hiperparâmetros
import optuna
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc
import datetime

# arquivos
from train_model import treinar_modelo


# optuna - opcional:
def objective(trial):
    # Definindo o espaço de busca dos hiperparâmetros
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0)
    gamma = trial.suggest_float('gamma', 0.001, 1.0)

    # Definindo o modelo XGBoost com os hiperparâmetros da tentativa atual
    modelo = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        random_state=47
    )
    
    features_RFE_xgboost = ['idade_cliente', 'renda_cliente', 'posse_residencia_cliente', 'tempo_emprego_cliente', 'finalidade_emprestimo', 'nota_emprestimo', 'taxa_juros_emprestimo', 'percentual_renda_emprestimo', 'retorno_emprestimo', 'ratio_renda_emp', 'media_valemp_nota', 'media_valemp_finalidade', 'std_valemp_residencia', 'ratio_emprego_renda']
    
    # Treinando o modelo com as features do RFE e exibindo performance
    modelo_treinado, X_test, y_test, _ = treinar_modelo(modelo, X[features_RFE_xgboost], y)
    
    y_pred_proba = modelo_treinado.predict_proba(X_test)[:, 1]
    
    # Calculando a curva Precision-Recall e AUC
    precisao, revocacao, _ = precision_recall_curve(y_test, y_pred_proba)
    metrica_pr_auc = auc(revocacao, precisao)

    # Caso o score da tentativa seja muito abaixo do mais alto encontrado
    trial.report(metrica_pr_auc, step=0)
    
    # Passamos para a próxima iteração
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # Retornar o pr_auc como a métrica a ser otimizada pelo Optuna
    return metrica_pr_auc

# utiliza a função objective para tunagem de hiperparâmetros
def optuna_tuning():
    # Registrando o tempo de início do estudo
    start_time = datetime.datetime.now()

    # Executando o estudo de otimização no xgboost
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    best_model_xgboost = XGBClassifier(**study.best_params)

    # Registrando o tempo de término do estudo
    end_time = datetime.datetime.now()

    # Calculando a duração do estudo
    duration = end_time - start_time
    print("Tempo total de execução do estudo: ", duration)

    hiperparametros_json = {'Melhores hiperparâmetros': study.best_params}
    print(hiperparametros_json)
    
    return study.best_params