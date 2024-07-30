import pickle

# carrega um classificador existente no diretório
def load_classifier(modelo) -> pickle.load:
    # classificador remodelado sendo substituido
    with open(modelo, "rb") as f:
        classifier = pickle.load(f)
    return classifier