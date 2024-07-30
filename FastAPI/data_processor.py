import random as rd
import pandas as pd
from category_encoders import TargetEncoder

# arquivos
from load_df import load_df


# Pré-processamento dos dados do input
def new_data_processing_(new_data: dict) -> pd.DataFrame:
    """
    Para o treinamento em produção os
    dados devem sofrer processamento de dados para serem acrescidos ao dataset
    """
    # convertendo o dicionário em DataFrame
    df_new_data = pd.DataFrame.from_dict(
        new_data,
        orient="index",
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

    # importa o DataFrame da EDA
    df_original = load_df()
    df = pd.concat([df_original, df_new_data], ignore_index=True)

    """
    O objetivo de treinar em produção é de montar portifolio, logo, o correto seria acrescentar dados
    de treino/teste caso eles já tivessem os rótulos definidos previamente pelo banco.
    Para as informações iniciais existirem, que são aquelas fornecidas pelo banco, criaremos elas com a lib random
    """
    # 1. taxa_juros_emprestimo
    df["taxa_juros_emprestimo"] = df["taxa_juros_emprestimo"].apply(
        lambda x: rd.choice([i / 10 for i in range(5, 225)]) if pd.isnull(x) else x
    )
    print("[SUCCESS] data_processor.py: taxa_juros_emprestimo")
    # 2. nota_emprestimo
    df["nota_emprestimo"] = df["nota_emprestimo"].apply(
        lambda x: rd.choice(["A", "B", "C", "D", "E"]) if pd.isnull(x) else x
    )
    print("[SUCCESS] data_processor.py: nota_emprestimo")
    # 3.
    df["percentual_renda_emprestimo"] = df.apply(
        lambda row: (
            (row["valor_emprestimo"] / row["renda_cliente"])
            if pd.isnull(row["percentual_renda_emprestimo"])
            else row["percentual_renda_emprestimo"]
        ),
        axis=1,
    )
    print("[SUCCESS] data_processor.py: percentual_renda_emprestimo")
    # 4.
    df["historico_inadimplencia_cliente"] = df["historico_inadimplencia_cliente"].apply(
        lambda x: 0 if pd.isnull(x) else x
    )
    print("[SUCCESS] data_processor.py: historico_inadimplencia_cliente")
    # 5.
    df["tempo_credito_cliente"] = df["tempo_credito_cliente"].apply(
        lambda x: 0 if pd.isnull(x) else x
    )
    print("[SUCCESS] data_processor.py: tempo_credito_cliente")
    
    # 6. Para evitar dados nulos, acrescentarei novos clientes com um viés fictício e aleatório de status_emprestimo.
    df["status_emprestimo"] = df["status_emprestimo"].apply(
        lambda x: rd.choice([0, 1]) if pd.isnull(x) else x
    )
    print("[SUCCESS] data_processor.py: status_emprestimo")

    # Feature Engineering
    # faremos o preenchimento do resto das variáveis após o merge com o dataset original
    # 1.
    df["retorno_emprestimo"] = df["valor_emprestimo"] * (
        df["taxa_juros_emprestimo"] / 100 + 1
    )
    print("[SUCCESS] data_processor.py: retorno_emprestimo")
    # 2.
    df["ratio_renda_emp"] = df["renda_cliente"] / df["valor_emprestimo"]
    print("[SUCCESS] data_processor.py: ratio_renda_emp")
    # 3.
    df["ratio_emprego_credito"] = (
        df["tempo_emprego_cliente"] / df["tempo_credito_cliente"]
    )
    print("[SUCCESS] data_processor.py: ratio_emprego_credito")
    # 4.
    df["media_valemp_nota"] = df.groupby(["nota_emprestimo"])[
        "valor_emprestimo"
    ].transform("mean")
    print("[SUCCESS] data_processor.py: media_valemp_nota")
    # 5.
    df["media_valemp_finalidade"] = df.groupby(["finalidade_emprestimo"])[
        "valor_emprestimo"
    ].transform("mean")
    print("[SUCCESS] data_processor.py: media_valemp_finalidade")
    # 6.
    df["std_valemp_residencia"] = df.groupby(["posse_residencia_cliente"])[
        "valor_emprestimo"
    ].transform("std")
    print("[SUCCESS] data_processor.py: std_valemp_residencia")
    # 7.
    df["media_renda_nota"] = df.groupby(["nota_emprestimo"])["renda_cliente"].transform(
        "mean"
    )
    print("[SUCCESS] data_processor.py: media_renda_nota")
    # 8.
    df["media_renda_finalidade"] = df.groupby(["finalidade_emprestimo"])[
        "renda_cliente"
    ].transform("mean")
    print("[SUCCESS] data_processor.py: media_renda_finalidade")
    # 9.
    df["std_renda_residencia"] = df.groupby(["posse_residencia_cliente"])[
        "renda_cliente"
    ].transform("std")
    print("[SUCCESS] data_processor.py: std_renda_residencia")
    # 10.
    df["ratio_emprego_renda"] = df["tempo_emprego_cliente"] / df["renda_cliente"]
    print("[SUCCESS] data_processor.py: ratio_emprego_renda")
    # 11.
    df["ratio_credito_renda"] = df["tempo_credito_cliente"] / df["renda_cliente"]
    print("[SUCCESS] data_processor.py: ratio_credito_renda")

    # para o verificação no console em produção
    rfe_columns = [
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

    print("[INFO] data_processor.py: ", df[rfe_columns].loc[:, :])
    print("[INFO] data_processor.py: ", df[rfe_columns].dtypes)

    return df
