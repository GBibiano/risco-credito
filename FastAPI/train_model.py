from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Criar o diretório 'model_report' se não existir
os.makedirs('model_report', exist_ok=True)

# Treina o modelo, salva precisão/recall/f1-score, matriz de confusão,
# feature_importance e
# retorna o modelo treinado, X_test, y_test, y_pred
def treinar_modelo(modelo, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    # Criando o StratifiedKFold com 10 folds
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=47)

    # Definimos quais são as features categóricas
    cat_features = [
        "posse_residencia_cliente",
        "finalidade_emprestimo",
        "nota_emprestimo",
    ]

    # Definindo um pipeline com StandardScaler, TargetEncoder e o modelo
    pipeline = Pipeline(
        [
            ("encoder", TargetEncoder(cols=cat_features)),
            ("scaler", StandardScaler()),
            ("modelo", modelo),
        ]
    )

    # Definindo as métricas de avaliação do modelo
    scoring = ["precision_weighted", "recall_weighted", "f1_weighted"]

    # Calculando as métricas usando cross_validate
    scores = cross_validate(
        pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False
    )

    # Separando os folds
    train_index, test_index = next(skf.split(X, y))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    # Treinamos a pipeline
    pipeline.fit(X_train, y_train)

    # Obtemos os resultados da previsão para aplicar nas funções das métricas
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # pr_auc: Precision-Recall Area Under the Curve
    precisao, revocacao, limiares = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(revocacao, precisao)

    # Salva métricas e matriz de confusão em .json
    metrics_json = {
        "report_precisao": f"Precision Mean (Weighted): {scores['test_precision_weighted'].mean():.2%}",
        "report_recall": f"Recall Mean (Weighted): {scores['test_recall_weighted'].mean():.2%}",
        "report_f1_score": f"F1 Score Mean (Weighted): {scores['test_f1_weighted'].mean():.2%}",
        "report_pr_auc": f"Precision x Recall, Area Under the Curve: ({pr_auc:.2%}) -> {pr_auc:.16f}",
    }

    # Salvando report_json em um arquivo .json na pasta 'remodel'
    with open(os.path.join('model_report', 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=4)

    # Plotando a matriz de confusão da pipeline
    matriz_confusao = confusion_matrix(y_test, y_pred)
    sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Greens")

    # Configurações do gráfico
    plt.title("Matriz de confusão")
    plt.xlabel("Valor Predito")
    plt.ylabel("Valor Real")
    plt.savefig(os.path.join('model_report', 'confusion_matrix.png'))

    # Salva a feature importance em .json

    if hasattr(pipeline.named_steps["modelo"], "feature_importances_"):
        feature_importances = pipeline.named_steps["modelo"].feature_importances_
        importances_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importances}
        )
        importances_df = importances_df.sort_values(
            by="Importance", ascending=False
        ).reset_index(drop=True)

        # Salvando a feature_importance em um arquivo .json na pasta 'remodel'
        importances_json = importances_df.to_dict(orient='records')
        with open(os.path.join('model_report', 'feature_importance.json'), 'w') as f:
            json.dump(importances_json, f, indent=4)

    return pipeline, X_test, y_test, y_pred
