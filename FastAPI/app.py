# 1. Importando libs
# disponibiliza em API
import uvicorn
from fastapi import FastAPI

# importa o arquivo model e sua função principal
from predict import processed_predict
from load_classifier import load_classifier
from model import model

# importa  o arquivo com a função de validação de dados de pydantic BaseModel
from inputvalidation import Validate
import os

# modelo que está sendo carregado
modelo = "xgboostclassifier-2024-07-27.pkl"

# Variáveis globais que servem de parâmetro para remodelagem
limite_remodelagem = 5 # exemplo fictício
new_data_count = 0
new_data = dict() 


def remodel() -> None:
    # se atingir o limite de 5 inputs, remodela
    global limite_remodelagem, new_data_count, new_data
    if new_data_count >= limite_remodelagem:
        """
        Caso as novas entradas de dados do método .get somarem o total de 5 clientes:
        new_data sofrerá as alterações da função abaixo:
        pré-processa, mescla com dataset original,
        remodela, carrega o pickle do novo classificador criado,
        retorna métricas e feature importance em json e
        retorna confusion_matrix em formato de imagem.
        """
        classifier = model(new_data)  # optuna=False

        # reseta o dicionário e contagem de novos dados após conclusão da remodelagem.
        new_data = dict()
        new_data_count = 0


# Cria o objeto API
app = FastAPI()
# Arquivo de modelagem original
classifier = load_classifier(modelo)
print("[INFO] Classifier: ", classifier.named_steps)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def index():
    return {
        "Previsão de Risco de Empréstimo": "https://github.com/Menotso/risco-credito"
    }


# Expor a função de predição, fazer uma predição com base em input '.json'
# e retornar o respectivo resultado
@app.post("/predict/")
def predict_post(
    val_idade_cliente: int,
    val_renda_cliente: int,
    val_posse_residencia_cliente: str,
    val_tempo_emprego_cliente: int,
    val_finalidade_emprestimo: str,
    val_valor_emprestimo: int,
):
    temporary_new_data = {
        new_data_count: [
            val_idade_cliente,
            val_renda_cliente,
            val_posse_residencia_cliente,
            val_tempo_emprego_cliente,
            val_finalidade_emprestimo,
            None,
            val_valor_emprestimo,
            None,
            None,
            None,
            None,
            None,
        ]
    }

    predict_data = processed_predict(temporary_new_data, count=new_data_count)

    prediction = classifier.predict(predict_data)
    if prediction[0] == 0:
        prediction = 0  # pagará empréstimo
    else:
        prediction = 1  # NÃO pagará empréstimo
    return {"prediction": prediction}


# exemplo de input via URL:
# http://127.0.0.1:8000/predict?val_idade_cliente=34&val_renda_cliente=56000&val_posse_residencia_cliente=Alugada&val_tempo_emprego_cliente=20&val_finalidade_emprestimo=Pessoal&val_valor_emprestimo=28000
# Expor a função de predição via requisição GET
@app.get("/predict")
def predict_get(
    val_idade_cliente: int,
    val_renda_cliente: int,
    val_posse_residencia_cliente: str,
    val_tempo_emprego_cliente: int,
    val_finalidade_emprestimo: str,
    val_valor_emprestimo: int,
):
    global new_data_count, new_data
    new_data[new_data_count] = [
        val_idade_cliente,
        val_renda_cliente,
        val_posse_residencia_cliente,
        val_tempo_emprego_cliente,
        val_finalidade_emprestimo,
        None,
        val_valor_emprestimo,
        None,
        None,
        None,
        None,
        None,
    ]

    predict_data = processed_predict(
        new_data, count=new_data_count
    )

    prediction = classifier.predict(predict_data)

    if prediction[0] == 0:
        prediction = 0  # pagará empréstimo
    else:
        prediction = 1  # NÃO pagará empréstimo
    new_data_count += 1
    remodel()
    return {"prediction": prediction}


# 5. Rodando a API com uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# http://127.0.0.1:8000/docs
# uvicorn app:app --reload
