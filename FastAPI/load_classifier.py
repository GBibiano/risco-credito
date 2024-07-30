import pickle

# carrega o classificador ao inicializar o projeto
def load_classifier(modelo) -> pickle.load:

    with open(modelo, "rb") as f:
        classifier = pickle.load(f)
    return classifier