import pandas as pd
import random as rd
from category_encoders import TargetEncoder


def processed_predict(new_data: dict, count: int) -> list[list]:
    # Obter os valores associados à chave
    data_values = new_data[count]
    
    df = pd.DataFrame(
        [data_values],
        columns=[
            "idade_cliente",
            "renda_cliente",
            "posse_residencia_cliente",
            "tempo_emprego_cliente",
            "finalidade_emprestimo",
            "nota_emprestimo",
            "valor_emprestimo",
            "taxa_juros_emprestimo",
            "percentual_renda_emprestimo",
            "historico_inadimplencia_cliente",
            "tempo_credito_cliente",
            "status_emprestimo",
        ],
    )

    print("[INFO] predict.py: Dados a serem previstos: ", df.loc[0, :])

    # O objetivo de treinar em produção é de montar portifolio, logo, o correto seria acrescentar dados
    # de treino/teste caso eles já tivessem os rótulos definidos previamente pelo banco.
    # Para as informações iniciais existirem, que são aquelas fornecidas pelo banco, criaremos elas com a lib random
    # 1.
    df["taxa_juros_emprestimo"] = df["taxa_juros_emprestimo"].apply(
        lambda x: rd.choice([i / 10 for i in range(5, 225)]) if pd.isnull(x) else x
    )
    # 2.
    df["nota_emprestimo"] = df["nota_emprestimo"].apply(
        lambda x: rd.choice(["A", "B", "C", "D", "E"]) if pd.isnull(x) else x
    )
    # 3.
    df["percentual_renda_emprestimo"] = df.apply(
        lambda row: (
            (row["valor_emprestimo"] / row["renda_cliente"])
            if pd.isnull(row["percentual_renda_emprestimo"])
            else row["percentual_renda_emprestimo"]
        ),
        axis=1,
    )
    # 4.
    df["historico_inadimplencia_cliente"] = df["historico_inadimplencia_cliente"].apply(
        lambda x: 0 if pd.isnull(x) else x
    )
    # 5.
    df["tempo_credito_cliente"] = df["tempo_credito_cliente"].apply(
        lambda x: 0 if pd.isnull(x) else x
    )
    # 6. Para evitar dados nulos, acrescentarei novos clientes com um viés fictício e aleatório de status_emprestimo.
    df["status_emprestimo"] = df["status_emprestimo"].apply(
        lambda x: rd.choice([0, 1]) if pd.isnull(x) else x
    )

    # Feature Engineering
    # faremos o preenchimento fictício do resto das variáveis
    # 1.
    df["retorno_emprestimo"] = df["valor_emprestimo"] * (
        df["taxa_juros_emprestimo"] / 100 + 1
    )
    # 2.
    df["ratio_renda_emp"] = df["renda_cliente"] / df["valor_emprestimo"]
    # 3.
    df["ratio_emprego_credito"] = (
        df["tempo_emprego_cliente"] / df["tempo_credito_cliente"]
    )
    # 4.
    df["media_valemp_nota"] = df.groupby(["nota_emprestimo"])[
        "valor_emprestimo"
    ].transform("mean")
    # 5.
    df["media_valemp_finalidade"] = df.groupby(["finalidade_emprestimo"])[
        "valor_emprestimo"
    ].transform("mean")
    # 6.
    df["std_valemp_residencia"] = 0
    df[df["std_valemp_residencia"] == 0.0] = df[
    df["std_valemp_residencia"] == 0.0
    ].replace({0.0: 1})
    # 7.
    df["media_renda_nota"] = df.groupby(["nota_emprestimo"])["renda_cliente"].transform(
        "mean"
    )
    # 8.
    df["media_renda_finalidade"] = df.groupby(["finalidade_emprestimo"])[
        "renda_cliente"
    ].transform("mean")

    # 9.
    df["std_renda_residencia"] = df.groupby(["posse_residencia_cliente"])[
        "renda_cliente"
    ].transform("std")
    # 10.
    df["ratio_emprego_renda"] = df["tempo_emprego_cliente"] / df["renda_cliente"]
    # 11.
    df["ratio_credito_renda"] = df["tempo_credito_cliente"] / df["renda_cliente"]

    # encoding de variáveis categóricas
    cat_features = [
        "posse_residencia_cliente",
        "finalidade_emprestimo",
        "nota_emprestimo",
    ]

    # Aplica o TargetEncoder
    encoder = TargetEncoder(cols=cat_features)
    df_encoded = encoder.fit_transform(df[cat_features], df["status_emprestimo"])
    df.posse_residencia_cliente = df_encoded.posse_residencia_cliente
    df.finalidade_emprestimo = df_encoded.finalidade_emprestimo
    df.nota_emprestimo = df_encoded.nota_emprestimo

    # para o predict em produção
    predict_columns = [
        "idade_cliente",
        "renda_cliente",
        "posse_residencia_cliente",
        "tempo_emprego_cliente",
        "finalidade_emprestimo",
        "nota_emprestimo",
        "taxa_juros_emprestimo",
        "percentual_renda_emprestimo",
        "retorno_emprestimo",
        "ratio_renda_emp",
        "media_valemp_nota",
        "media_valemp_finalidade",
        "std_valemp_residencia",
        "ratio_emprego_renda",
    ]

    predict_info = df.loc[0, predict_columns]

    return [predict_info]
