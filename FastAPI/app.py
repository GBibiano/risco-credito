# 1. Importando libs
# disponibiliza em API
import uvicorn
from fastapi import FastAPI

# importa o arquivo model e sua função principal
from data_processor import model, new_data_processing_
from dumpster import load_classifier

# importa  o arquivo com a função de validação de dados de pydantic BaseModel
from inputvalidation import Validate
from train_model import model

# modelo que está sendo carregado
modelo = "xgboostclassifier-2024-07-27.pkl"

# Variáveis globais que servem de parâmetro para remodelagem
limite_remodelagem = 5
new_data = dict()

# se as novas entradas de dados do método .get somarem o total de 5 clientes,
# retreina o modelo, cria o pickle e substitui-o em tempo de execução
def remodel():
    # se atingir o limite de 5 inputs, remodela
    global limite_remodelagem, new_data_count
    if new_data_count >= limite_remodelagem:
        # new_data sofrerá as alterações da função abaixo:
        # pré-processa, mescla com dataset original,
        # remodela, substitui o pickle atual, retorna performance em json e
        # retorna confusion_matrix em formato de imagem.
        model(new_data) # optuna=True
        # Aqui deve ser garantido que o novo arquivo seja utilizado como classificador
        classifier = load_classifier(modelo)

        # reseta o dicionário de novos dados após conclusão da remodelagem.
        new_data = dict()
        new_data_count = 0


# Cria o objeto API
app = FastAPI()
# Arquivo de modelagem original
classifier = load_classifier(modelo)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def index():
    return {
        "Previsão de Risco de Empréstimo": "https://github.com/Menotso/risco-credito"
    }


# Expor a função de predição, fazer uma predição com base em input '.json'
# e retornar o respectivo resultado
# É um teste da utilização do BaseModel da biblioteca pydantic
@app.post("/predict/")
def predict_post(
    val_idade_cliente: int,
    val_renda_cliente: int,
    val_posse_residencia_cliente: str,
    val_tempo_emprego_cliente: int,
    val_finalidade_emprestimo: str,
    val_valor_emprestimo: int,
    ):
    global new_data_count, new_data
    new_data = {f"cliente_{new_data_count}": [ 
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
        None
        ]}
    
    predict_info = new_data_processing_(new_data, predict=True)
    
    prediction = classifier.predict(predict_info)
    if prediction[0] == 0:
        prediction = 0  # pagará empréstimo
    else:
        prediction = 1  # NÃO pagará empréstimo
    return {"prediction": prediction}


# ex de input URL:
# http://127.0.0.1:8000/predict?val_idade_cliente=1&val_renda_cliente=2&val_posse_residencia_cliente=3&val_tempo_emprego_cliente=4&val_finalidade_emprestimo=5&val_valor_emprestimo=6
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
    new_data = {f"cliente_{new_data_count}": [ 
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
        None
        ]}
    
    new_data_count += 1
    
    predict_info = new_data_processing_(new_data, predict=True)
    
    prediction = classifier.predict(predict_info)
    if prediction[0] == 0:
        prediction = 0  # pagará empréstimo
    else:
        prediction = 1  # NÃO pagará empréstimo
    remodel()
    return {"prediction": prediction}
    
# 5. Rodar a API com uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# http://127.0.0.1:8000/docs
# uvicorn app:app --reload
