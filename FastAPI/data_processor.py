import random as rd
import pandas as pd
from sklearn.pipeline import Pipeline

# arquivos
from dumpster import load_df

# Pré-processamento dos dados do input
def new_data_processing_(new_data: dict=None, predict: bool=False) -> pd.DataFrame:
    # para o treinamento em produção
    # Dados devem sofrer processamento de dados para serem acrescidos ao dataset
    # Convertendo o dicionário em DataFrame
    df = pd.DataFrame.from_dict(new_data, orient='index', columns=[
    'idade_cliente',
    'renda_cliente',
    'posse_residencia_cliente',
    'tempo_emprego_cliente',
    'finalidade_emprestimo',
    'nota_emprestimo',
    'valor_emprestimo',
    'taxa_juros_emprestimo',
    'percentual_renda_emprestimo',
    'historico_inadimplencia_cliente',
    'tempo_credito_cliente',
    'status_emprestimo'
])
    # O objetivo de treinar em produção é de montar portifolio, logo, o correto seria acrescentar dados 
    # de treino/teste caso eles já tivessem os rótulos definidos previamente pelo banco.
    # Para as informações iniciais existirem, que são aquelas fornecidas pelo banco, criaremos elas com a lib random
    # 1.
    df['taxa_juros_emprestimo'] = df['taxa_juros_emprestimo'].apply(lambda x: rd.randint(0.5, 22.5) if pd.isnull(x) else x)
    # 2.
    df['nota_emprestimo'] = df['nota_emprestimo'].apply(lambda x: rd.random_choice(['A', 'B', 'C', 'D', 'E']) if pd.isnull(x) else x)
    # 3.
    df['percentual_renda_emprestimo'] = df.apply(
    lambda row: (row['valor_emprestimo'] / row['renda_cliente'])
    if pd.isnull(row['percentual_renda_emprestimo']) 
    else row['percentual_renda_emprestimo'], 
    axis=1
)
    # 4.
    df['historico_inadimplencia_cliente'] = df['historico_inadimplencia_cliente'].apply(lambda x: 0 if pd.isnull(x) else x)
    # 5.
    df['tempo_credito_cliente'] = df['tempo_credito_cliente'].apply(lambda x: 0 if pd.isnull(x) else x)
    # 6. Para evitar dados nulos, acrescentarei novos clientes com um viés fictício e aleatório de status_emprestimo.
    df['status_emprestimo'] = df['status_emprestimo'].apply(lambda x: rd.random_choice(0, 1) if pd.isnull(x) else x)
    
    # Feature Engineering
    # faremos o preenchimento do resto das variáveis antes do merge com o dataset original
    # 1. 
    df['retorno_emprestimo'] = df['valor_emprestimo'] * (df['taxa_juros_emprestimo'] / 100 + 1)
    # 2.
    df['ratio_renda_emp'] = df['renda_cliente'] / df['valor_emprestimo']
    # 3. 
    df['ratio_emprego_credito'] = df['tempo_emprego_cliente'] / df['tempo_credito_cliente']
    # 4.
    df['media_valemp_nota'] = df.groupby(['nota_emprestimo'])['valor_emprestimo'].transform('mean')
    # 5.
    df['media_valemp_finalidade'] = df.groupby(['finalidade_emprestimo'])['valor_emprestimo'].transform('mean')
    # 6.
    df['std_valemp_residencia'] = df.groupby(['posse_residencia_cliente'])['valor_emprestimo'].transform('std')
    # 7.
    df['media_renda_nota'] = df.groupby(['nota_emprestimo'])['renda_cliente'].transform('mean')
    # 8.
    df['media_renda_finalidade'] = df.groupby(['finalidade_emprestimo'])['renda_cliente'].transform('mean')
    # 9.
    df['std_renda_residencia'] = df.groupby(['posse_residencia_cliente'])['renda_cliente'].transform('std')
    # 10.
    df['ratio_emprego_renda'] = df['tempo_emprego_cliente'] / df['renda_cliente']
    # 11.
    df['ratio_credito_renda'] = df['tempo_credito_cliente'] / df['renda_cliente']
    
    # para o predict em produção
    if predict:    
        predict_columns = ['idade_cliente',
                    'renda_cliente',
                    'posse_residencia_cliente',
                    'tempo_emprego_cliente',
                    'finalidade_emprestimo',
                    'nota_emprestimo',
                    'taxa_juros_emprestimo',
                    'percentual_renda_emprestimo',
                    'retorno_emprestimo',
                    'ratio_renda_emp',
                    'media_valemp_nota',
                    'media_valemp_finalidade',
                    'std_valemp_residencia',
                    'ratio_emprego_renda'
                    ]
        
        predict_info = df.loc[0, predict_columns].tolist()
        
        return [predict_info]
    
    df_original = load_df()
    df_merged = df_original.append(df) # alterar
    
    return df_merged
