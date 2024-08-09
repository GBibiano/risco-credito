# Previsão de Risco de Crédito com Deploy em FastAPI

![tema](https://github.com/Menotso/risco-credito/blob/main/imagens/Tema.jpeg)

_Imagem criada com [ideogram](https://ideogram.ai/)_

## Objetivos

Entregar um site do gênero bancário, fictício, que permite o acesso e cadastro de clientes, visualização do empréstimo que possuem direito e saldo. Adicionalmente, existirá perfis de acesso onde a gestão poderá visualizar diretamente no site o impacto no negócio da implementação de um modelo de machine learning que prevê o risco de crédito dos clientes.

## Entregáveis

- **Previsão do Risco de Crédito**
  - **Análise Exploratória de Dados**
  - **Modelagem**
    - Feature Engineering, Importance, Scaling & Selection
    - Validação Cruzada
    - Tunagem de Hiperparâmetros
    - Deploy (Predição e Remodelagem)
    - Impacto do modelo no contexto de negócio
   
## Apêndice

- Análise Exploratória de Dados
- Modelo de Machine Learning
- Tunagem de Hiperparâmetros
- Deploy (Predição e Remodelagem)
- Aplicação Prática de Negócio

## Análise Exploratória de Dados

#### Descrição do problema
  Quando se trata de trabalhar com dados em um banco é esperado que a concessão de empréstimos seja rigorosamente avaliada com base em argumentos. Um dos problemas dessa empresa em específico era de prever quando uma concessão de empréstimo à um cliente resulta em inadimplência. A inadimplência ocorre quando uma pessoa física ou jurídica deixa de cumprir uma obrigação financeira dentro do prazo estipulado. Com a intenção de sanar este problema, pedimos uma base de dados dos clientes para elaborar uma solução utilizando algoritmos de aprendizagem de máquina.

#### Arquivos da EDA

- **credit_risk_dataset.csv** [visualizar](https://github.com/Menotso/risco-credito/blob/main/EDA/credit_risk_dataset.csv)
- **EDA_Credit_Risk.ipynb** [visualizar](https://github.com/Menotso/risco-credito/blob/main/EDA/EDA_Credit_Risk.ipynb)

#### Informações do dataset

- **`idade_cliente`**: Idade do cliente
- **`renda_cliente`**: Renda anual do cliente
- **`posse_residencia_cliente`**: tipo de posse da residência (casa alugada, quitada ou em hipoteca)
- **`tempo_emprego_cliente`**: anos trabalhados dos clientes
- **`finalidade_emprestimo`**: finalidade do empréstimo
- **`nota_emprestimo`**: grau do empréstimo
- **`valor_emprestimo`**: Valor solicitado pelo cliente
- **`taxa_juros_emprestimo`**: taxa de juros do empréstimo
- **`status_emprestimo`**: variável target dos modelos. 0 = pagou empréstimo 1 = não pagou;
- **`percentual_renda_emprestimo`**: percentagem do empréstimo dividido pela renda anual do cliente
- **`historico_inadimplencia_cliente`**: se possui histórico de inadimplência
- **`tempo_credito_cliente`**: histórico de crédito em anos

Reduzimos o tamanho do dataset de 3.0 MB para **1.6 MB** alterando o tipo das features.

#### Tratamento de dados nulos, duplicados, outliers

O dataset possuía:

- **Dados nulos:**
  - 895 clientes sem informação de anos trabalhados (`tempo_emprego_cliente`), foram preenchidos os nulos de anos trabalhados com o valor mais frequente (moda)
  - 3.116 clientes sem informação de taxa de juros (`taxa_juros_emprestimo`), foram retiradas as 3.116 instâncias de taxa de juros para evitar incluir viés fictício nos dados que poderia gerar underfitting mais à frente
  - Haviam clientes com 0% em `percentual_renda_emprestimo` que foram retirados coincidentemente junto do tratamento de `taxa_juros_emprestimo`
- **Dados duplicados:**
  - Não haviam dados duplicados nesta base de dados.
- **Outliers:**
  - Um cliente com 144 anos e outro com 123 anos, foram tratados como 44 e 23 anos, respectivamente
  - 50 clientes com renda anual acima de 500 mil, foram retirados da base de dados diminuindo o desvio-padrão da renda em quase 20.000
  - Dois clientes com 123 anos trabalhados, foram tratados como 23 anos.

#### Verificando padrões nos dados

Executei a análise univariada e bivariada em:

  - **Variáveis Numéricas:**
    - KDE Plots
    - Gráficos de Contagem
    - Box Plots
    - Gráficos de Violino
    - Histogramas
  - **Variáveis Categóricas:**
    - Gráficos de Pontos
    - Gráficos de Barras
      
#### Resultados Principais da EDA

#### Variáveis Numéricas

  ![numericas_eda1](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_EDA/numericas_eda1.png)
  - Todas as features envolvidas possuem uma **assimetria positiva, uma cauda à direita**.
  - **`idade_cliente`**: A população do dataset é composta majoritariamente por jovens e adultos entre 21 e 35 anos.
      - O maior grupo de clientes possui 23 anos com 3889 instâncias.
  - **`renda_cliente`**: A quantidade de clientes que recebem MENOS que 135 mil por ano é o grupo predominante com 27.848 instâncias.
      - Enquanto os que recebem entre 135 a 500 mil são aproximadamente 1530 clientes.
  - **`tempo_emprego_cliente`**: A maior concentração de tempo empregatício que se encontra em `tempo_emprego_cliente` está entre 0 e 10 anos. O valor máximo está em 40 anos trabalhados.
  - **`valor_emprestimo`**: Os valores de empréstimos se concentram aproximadamente entre 2 mil e 15 mil. O maior valor de empréstimo fornecido é de 35.000 para poucos clientes e não há outliers.
  - **`taxa_juros_emprestimo`**: A concentração de dados está em 7,5% e entre 10% a 15% de juros anual.
      - A taxa de juros do empréstimo mostra que há pelo menos um cliente com 23% de juros ao ano, que é a maior taxa.
      - A escala relativa de tempo não é conhecida, mas presumimos que está em anos.
  - **`percentual_renda_emprestimo`**: A concentração dos dados se encontra entre 1% e 30%; e
      - É possível visualizar que não há mais uma quantia considerável de clientes dentro da faixa de 0%. 
  - **`tempo_credito_cliente`**: Metade do dataset está entre 0 e 5 anos e 1/3 do dataset está entre 5 e 10 anos de tempo de crédito do cliente.
      - O valor máximo está em aproximadamente 30 anos. Assumimos que não há outliers neste contexto.
      - Não sabemos a escala relativa de tempo, mas supomos que está em anos.

#### Variáveis Categóricas

Para efeitos de EDA, considerei como variáveis categóricas todas as features que possuem 7 valores únicos ou menos.

  ![categoricas_eda1](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_EDA/categoricas_eda1.png)
  - **`posse_residencia_cliente`**: A maioria dos clientes alugam moradias e possuem hipoteca.
  - **`finalidade_emprestimo`**:  finalidade dos empréstimos possui uma leve assimetria positiva, sendo os três maiores motivos de solicitar empréstimo: Educação, Médico e Empreendimento.
  - **`nota_emprestimo`**: As notas de empréstimos mais presentes no dataframe são do tipo "A", "B" e "C".
  - **`historico_inadimplencia_cliente`**: Existe um desbalanceamento nestes dados exibindo **5.199** clientes (21.47%) com histórico de crédito ruim sendo minoria com uma proporção de **a cada 5 clientes um possui histórico de inadimplência.**

#### Variável Target

  ![target_eda](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_EDA/target_eda.png)
  - **`status_emprestimo`**: A variável alvo confirmou o desbalanceamento dos dados exibindo **6.459** clientes (21.96%) que estão classificados como inadimplentes com uma proporção de **a cada 4 clientes um é inadimplente.**

#### Comparando as variáveis numéricas com o Target

Após comparar todas as variáveis com o target e verificar a proporção de inadimplentes em cada uma, pudemos visualizar que:

  ![numericas_eda2](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_EDA/numericas_eda2.png)
  - **`idade_cliente`**: idade do cliente.
      - Há uma participação relativamente constante dos inadimplentes em todos os intervalos.
      - A maior concentração de inadimplentes está no intervalo entre 20 e 40 anos de idade com 6.128 instâncias.
      - O intervalo com a maior representação de inadimplentes está entre 60 e 70 anos de idade com 36% do total de clientes deste intervalo.
      - Há um cliente com 94 anos de idade classificado como **adimplente**.
  - **`renda_cliente`**: renda anual do cliente.
      - A maior concentração de inadimplentes está entre 20 e 80 mil de renda com 5.043 instâncias.
      - Os inadimplentes possuem uma representação majoritária acima de 80% para os intervalos até 20 mil de renda anual.
      - Há um cliente que recebe acima de 499.800 de renda classificado como **inadimplente**.
  - **`tempo_emprego_cliente`**: registro em anos de serviço do cliente.
      - A maior concentração de inadimplentes está evidenciado até 15 anos de serviço com 6.327 instâncias.
      - Os intervalos de **0 a 5**, **20 a 25** e **30 a 35 anos**, os inadimplentes representam acima de 25% do total de clientes destes intervalos.
      - Há um cliente com mais de 35 anos de serviço classificado como **adimplente**.
  - **`valor_emprestimo`**: valor recebido pelo interessado.
      - A maior concentração de inadimplentes se encontra entre mil e 20 mil do valor do empréstimo com 5.496 instâncias.
      - Os inadimplentes possuem uma representação de aproximadamente 34% nos intervalos acima de 20 mil do valor do empréstimo.
  - **`taxa_juros_emprestimo`**: juros anual do empréstimo.
      - A maior concentração de inadimplentes ocorre até 20% de juros com 6.396 instâncias.
      - Em 15% de juros e acima a representatividade de inadimplentes é superior à 50% em cada intervalo.
      - Há um cliente classificado como **inadimplente** com aproximadamente 23.22% de juros anual.
  - **`percentual_renda_emprestimo`**: Porcentagem -> razão do empréstimo sobre a renda.
      - A maior concentração de inadimplentes está nos primeiros intervalos entre 10% e 40% do empréstimo sobre a renda com 5.558 instâncias.
      - Há uma representatividade majoritária de inadimplentes dentro dos intervalos acima de 30%, somam aproximadamente 66% do total de clientes destes intervalos.
      - Há um cliente classificado como **adimplente** que possui aproximadamente 83% da razão do empréstimo sobre a renda.
  - **`tempo_credito_cliente`:** história de crédito do cliente em anos.
      - A maior concentração de inadimplentes aparece nos intervalos entre 2 e 15 anos de tempo de crédito com 6.147 instâncias.
      - Há uma participação relativamente constante dos inadimplentes em todos os intervalos.
      - Os inadimplentes possuem uma representação de 27% nos intervalos com tempo de crédito registrado acima de 20 anos.

#### Comparando as variáveis categóricas com o Target

  ![categoricas_eda2](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_EDA/categoricas_eda2.png)
  - **`posse_residencia_cliente`**:
      - Os inadimplentes na maioria das vezes possuem **hipoteca ou moradia alugada**, representando 6.263 instâncias.
      - Inadimplentes representam 29% ou mais quando possuem **moradia alugada ou outra forma de posse**.
  - **`finalidade_emprestimo`**:
      - As finalidades 'Consolidação de Dívidas', 'Médico', 'Pessoal' e 'Educação' possuem a maior concentração de inadimplentes com 4.833 instâncias.
      - Os inadimplentes possuem uma representação acima de 25% nas finalidades 'Melhoria da Casa', 'Consolidação de Dívidas', 'Médico' com 3.678 instâncias.
  - **`nota_emprestimo`**:
      - A maior concentração de inadimplentes se encontra nas notas 'A', 'B', 'C' e 'D' com 5.681 instâncias.
      - Inadimplentes possuem uma representatividade majoritária nas notas 'D', 'E', 'F' e 'G', com porcentagens acima de 50%, estes clientes representam 2.747 instâncias.
      - Especial atenção na nota de empréstimo 'G' que possui 58 clientes inadimplentes representando aproximadamente 99% da subcategoria.
  - **`historico_inadimplencia_cliente`**:
      - A maior concentração de inadimplentes não possui histórico de inadimplência com 4.480 instâncias (18.5% do total de clientes **sem** histórico de inadimplência).
      - Os inadimplentes representam 38.1% do total de clientes **com** histórico de inadimplência com 1.979 instâncias.

Ao final exportei o arquivo pickle da base de dados para utilizar na modelagem.

## Modelo de Machine Learning

#### Arquivo da modelagem

- **modelagem.ipynb** [visualizar](https://github.com/Menotso/risco-credito/blob/main/Modelagem_ML/modelagem.ipynb)

### Modelagem

Optei por utilizar três modelos que lidam relativamente bem com dados desbalanceados:

- **Random Forest Classifier**;
- **Gradient Boosting Classifier**; e
- **XGBoost Classifier**.

Utilizei principalmente três funções:
- **`feature_engineering`:**
  - Executa o feature engineering no DataFrame fornecido e retorna após as alterações.
- **`treinar_modelo()`:**
  - Treina o modelo, exibe precisão/recall/f1-score, matriz de confusão e feature_importance;
  - Utilizando:
    - Split em treino/teste: StratifiedKFold
    - Feature Encoding: Target Encoder
    - Feature Scaling: Standard Scaler
    - Cross Validate: (Weighted) -> Média da Precisão, Média da Revocação Média do F1 Score
      - Métrica utilizada: Precisão x Revocação, Área abaixo da Curva
  - Retorna o modelo treinado, X_test, y_test e y_pred
- **`rfe_report()`:**
  - Exibe um relatório até destacar o Recursive Feature Elimination que obteve a melhor métrica.
  - Retorna as features da melhor métrica
  
#### Correlação de Pearson

Há uma correlação positiva moderada entre as variáveis:

- **`idade_cliente`** e **`tempo_credito_cliente`**; e
- **`taxa_juros_emprestimo`** e **`historico_inadimplencia_cliente`**

#### Feature Engineering

Criei as features em cada fold para evitar data leakage com:
- Médias e desvios-padrão de features; e
- Ratios: divisão de uma feature por outra.

Criando 11 features adicionais e totalizando 23, sendo elas:

1. **`retorno_emprestimo`** = multiplicando (`taxa_juros_emprestimo` por `valor_emprestimo`) + `valor_empréstimo`;
2. **`ratio_renda_emp`** = dividindo `renda_cliente` por `valor_emprestimo`;
3. **`ratio_emprego_credito`** = `tempo_emprego_cliente` dividido por `tempo_credito_cliente`, caso dividir por zero, resulta em zero;
4. **`media_valemp_nota`** = média de `valor_emprestimo` para cada `nota_emprestimo`;
5. **`media_valoremp_finalidade`** = média de `valor_emprestimo` para cada `finalidade_emprestimo`;
6. **`std_valemp_residencia`** = desvio-padrão do `valor_emprestimo` por `posse_residencia_cliente`;
7. **`media_renda_nota`** = média de `renda_cliente` para cada `nota_emprestimo`;
8. **`media_renda_finalidade`** = média de `renda_cliente` para cada `finalidade_emprestimo`;
9. **`std_renda_residencia`** = desvio-padrão do `renda_cliente` por `posse_residencia_cliente`;
10. **`ratio_emprego_renda`** = `tempo_emprego_cliente` dividido por `renda_cliente`; e
11. **`ratio_credito_renda`** = `tempo_credito_cliente` dividido por `renda_cliente`.

#### Avaliando a performance de modelos e respectivas features importances

Para evitar overfitting e capturar ruídos dos dados, preferi verificar a importância das features em cada modelo e tratar as piores posteriormente com RFE. Os modelos foram treinados com 10 folds no StratifiedKFold (SKF) em razão do desbalanceamento de classes, situação em que o SKF lida bem na distribuição de folds em treino e teste estratificando-os.

- **Random Forest Classifier:**
    - Preferi utilizar a proporção inversa das classes como pesos ao invés de utilizar balanceamento de classes do gênero undersampling ou oversampling (RUS, ADASYN, SMOTE)
    - Média da Precisão (Weighted): 93.77%
    - Média da Revocação (Weighted): 93.44%
    - Média do F1 Score (Weighted): 93.04%
    - Precisão x Revocação, Área abaixo da Curva: 89.40%
      
    ![rf_padrao](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/rf_padrao.png)
  
    - Top 5 Feature Importance:
      - Inserir tabela com features com maior importância do modelo de ML
        - ratio_renda_emp;
        - percentual_renda_emprestimo;
        - renda_cliente;
        - taxa_juros_emprestimo;
        - media_valemp_nota;
    - Nenhuma feature sem importância para o modelo, sendo a menor **`historico_inadimplencia_cliente`** com 0.006285.
- **Gradient Boosting Classifier:**
    - Média da Precisão (Weighted): 92.46%
    - Média da Revocação (Weighted): 92.21%
    - Média do F1 Score (Weighted): 91.71%
    - Precisão x Revocação, Área abaixo da Curva: 86.61%
    - Matriz de Confusão:
  
    ![gbc_padrao](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/gbc_padrao.png)
  
    - Top 5 Feature Importance:
        - Inserir tabela com features com maior importância do modelo de ML
        - ratio_renda_emp
        - media_valemp_nota
        - nota_emprestimo
        - posse_residencia_cliente
        - renda_cliente
    - 8 features sem importância para o modelo, sendo 4 delas do DataFrame original.
- **XGBoost Classifier:** 
    - Média da Precisão (Weighted): 93.80%
    - Média da Revocação (Weighted): 93.53%
    - Média do F1 Score (Weighted): 93.16%
    - Precisão x Revocação, Área abaixo da Curva: 90.81%
    - Matriz de confusão:
      
    ![xgb_padrao](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/xgb_padrao.png)
  
    - Top 5 Feature Importance:
        - Inserir tabela com features com maior importância do modelo de ML
        - ratio_renda_emp
        - nota_emprestimo
        - posse_residencia_cliente
        - tempo_emprego_cliente
        - finalidade_emprestimo
    - 1 feature sem importância para o modelo, sendo ela **`std_renda_residencia`**.

O modelo que se saiu melhor em termos de minimizar os falsos positivos foi o Random Forest Classifier, contudo, desejamos diminuir o número de clientes inadimplentes que o modelo erra (falsos negativos). Logo o **XGBoost Classifier** obteve resultados melhores com um custo baixo de falsos positivos.

O modelo com a pior performance foi o Gradient Boosting Classifier.

#### Feature Selection

Utilizei o RFE (Recursive Feature Elimination). Testei a quantidade de features cuja métrica utilizada foi a área abaixo da curva entre precisão e revocação (pr_auc).

- **Random Forest Classifier:**
  - Melhor "X_train" com 13 features; e
  - Precision-Recall AUC:  0.8644429061026928
- **Gradient Boosting Classifier:**
  - Melhor "X_train" com 16 features; e
  - Precision-Recall AUC:  0.8376407331700388
- **XGBoost Classifier:**
  - Melhor "X_train" com 15 features; e
  - Precision-Recall AUC:  0.8956662881834732

O modelo com a maior métrica de Precision-Recall AUC é o **XGBoost Classifier**.
 
Após o RFE, 7 das 14 features que foram selecionadas para o **XGBoost Classifier** são do DataFrame original.

**Importância das Features (maior para menor)**
| feature   | Descrição      |
| :---------- | :--------- |
| `idade_cliente`      | Idade do Cliente |
| `renda_cliente`      | Renda do Cliente |
| `posse_residencia_cliente`      | Tipo de posse de residência |
| `tempo_emprego_cliente`      | Anos trabalhados |
| `finalidade_emprestimo`      | Finalidade do empréstimo |
| `nota_emprestimo`      | Grau atribuido ao empréstimo |
| `taxa_juros_emprestimo`      | Taxa de Juros do empréstimo |
| `percentual_renda_emprestimo`      | Razão do empréstimo sobre a renda |
| `retorno_emprestimo`      | Valor retornado ao banco ao quitar o empréstimo |
| `ratio_renda_emp`      | Razão da renda sobre o empréstimo |
| `media_valemp_nota`      | Média do valor do empréstimo por cada grau |
| `media_valemp_finalidade`      | Média do valor do empréstimo por finalidade |
| `std_valemp_residencia`      | Desvio-Padrão do valor do empréstimo por posse de residência |
| `ratio_emprego_renda`      | Razão de anos trabalhados pela renda do cliente |

### Tunagem de Hiperparâmetros

Optei por utilizar a pesquisa Bayesiana da biblioteca **Optuna**. Utilizei as features selecionadas pelo Recursive Feature Selection para evitar overfitting.

Performance dos modelos após a tunagem de hiperparâmetros:

- **Random Forest Classifier**:
  - Média da Precisão (Weighted): 93.02%
  - Média da Revocação (Weighted): 92.98%
  - Média do F1 Score (Weighted): 92.66%
  - Precisão x Revocação, Área abaixo da Curva: 89.49%
  - Matriz de confusão:

  ![rf_hiperparametros](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/rf_hiperparametros.png)

- **Gradient Boosting Classifier**:
  - Média da Precisão (Weighted): 93.81%
  - Média da Revocação (Weighted): 93.69%
  - Média do F1 Score (Weighted): 93.39%
  - Precisão x Revocação, Área abaixo da Curva: 90.41%
  - Matriz de confusão:
  
  ![gbc_hiperparametros](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/gbc_hiperparametros.png)

- **XGBoost Classifier**:
  - Média da Precisão (Weighted): 93.40%
  - Média da Revocação (Weighted): 93.38%
  - Média do F1 Score (Weighted): 93.11%
  - Precisão x Revocação, Área abaixo da Curva: 89.79%
  - Matriz de confusão:
  
  ![xgb_hiperparametros](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/xgb_hiperparametros.png)

O modelo **XGBoost Classifier** possui a segunda maior revocação, os melhores hiperparâmetros foram:

| Parâmetro   | Valor      |
| :---------- | :--------- |
| `n_estimators`      | 227 |
| `max_depth`      | 12 |
| `learning_rate`      | 0.4112354775681158 |
| `gamma`      | 0.09972654226276964 |

O respectivo espaço de busca do XGBoost que durou menos de 5 minutos (30 trials):

```bash
n_estimators = trial.suggest_int('n_estimators', 50, 300)
max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0)
gamma = trial.suggest_float('gamma', 0.001, 1.0)
```

### Modelo escolhido

Em um cenário real, utilizaríamos o Gradient Boosting Classifier pelos resultados serem melhores, mas para fins didáticos e de praticidade do deploy, utilizarei o XGBoost Classifier.

Escolhi o **XGBoost Classifier** pelas razões abaixo:
- Possui a revocação um pouco abaixo do Gradient Boosting e identificou relativamente bem os inadimplentes. 
- Tempo de treinamento do modelo e respectiva tunagem de hiperparâmetros são rápidos, resultando em pouco tempo de atraso na resposta em deploy;
- Em produção não utiliza tanto processamento e memória; e
- O arquivo pickle do XGBoost Classifier é o mais leve.

## Deploy

Resumidamente, um usuário pode fazer uma requisição GET via URL e a resposta retornada é a previsão se o cliente irá pagar ou não o empréstimo.

Utilizamos a biblioteca padrão `Random` para facilitar o processo:
  - De predição em deploy fazendo a inserção de dados fictícios para preencher informações que o banco deveria fornecer, como por exemplo, taxa de juros do empréstimo.
  - Da remodelagem em deploy utilizando dados fictícios na inserção da variável target, tarefa associada ao treino e teste.

A remodelagem em deploy ocorre sempre que a previsão de 5 novos clientes é feita e substitui o classificador que está sendo utilizado. 5 clientes foi apenas um número fictício para poder testar a funcionalidade de remodelagem, na prática este número aumentaria.

- Podemos inicializar o arquivo com o comando `uvicorn app:app --reload` no terminal.
- No destino web `http://127.0.0.1:8000/docs` podemos fazer uma requisição POST para testar a função predict; ou
- Executar uma requisição GET através de URL, um exemplo de um cliente com 34 anos, 56.000 de renda anual, com casa alugada, com 20 anos trabalhados e que solicitou o empréstimo de 28.000 para uso pessoal:
  - `http://127.0.0.1:8000/predict?val_idade_cliente=34&val_renda_cliente=56000&val_posse_residencia_cliente=Alugada&val_tempo_emprego_cliente=20&val_finalidade_emprestimo=Pessoal&val_valor_emprestimo=28000`
  - O restante das features exigidas pelo Recursive Feature Elimination na modelagem feita em [modelagem.ipynb](https://github.com/Menotso/risco-credito/blob/main/Modelagem_ML/modelagem.ipynb) é processado pelos módulos na pasta da FastAPI com dados fictícios

#### Arquivos da Remodelagem e Predição em Deploy

- Diretório da FastAPI [visualizar diretório](https://github.com/Menotso/risco-credito/tree/main/FastAPI)
- Diretório do Resultado da Remodelagem em Deploy [visualizar diretório](https://github.com/Menotso/risco-credito/tree/main/FastAPI/model_report)

#### Arquitetura das dependências de cada módulo

- arquivo principal -> **app.py**

![dependencias](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_FASTAPI/dependencias.png)

_Criado com a biblioteca `pydeps`._

Reutilizei as funções existentes no arquivo da modelagem com pequenas alterações e fiz os seguintes módulos:

- **app.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/app.py)
  - Arquivo com a FastAPI
  - Disponibiliza o modelo em uma API
  - Chama a função predict
  - Chama a função remodelagem caso tenha previsto 5 novos clientes
- **load_classifier.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/load_classifier.py)
  - Carrega um classificador ao iniciar o **app.py**
- **predict.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/predict.py)
  - Faz encoding de features categóricas com `TargetEncoder`
  - Efetua o Feature Engineering
  - Retorna os dados formatados para o classificador executar a previsão
- **tuning.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/tuning.py)
  - Executa a tunagem de hiperparâmetros do modelo XGBoostClassifier com Optuna, retorna melhores hiperparâmetros.
- **classificador.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/classificador.py)
  - Opcionalmente, faz a tunagem de hiperparâmetros ou utiliza os melhores hiperparâmetros encontrados pela primeira modelagem.
  - Serve para instanciar um classificador no momento de executar a remodelagem em deploy
- **data_processor.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/data_processor.py)
  - Cria um DataFrame com as informações do cliente
  - Utiliza **load_df.py** ([visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/load_df.py)) para carregar o dataset original e mescla os dois
  - Efetua o Feature Engineering
  - Retorna o DataFrame processado
  - Faz o processamento de dados enviados pela requisição GET do FastAPI antes de remodelar 
- **model.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/model.py)
  - Chama o dataframe retornado por **data_processor.py** para separar a variável alvo das independentes
  - Verifica se há necessidade de tunagem de hiperparâmetros, caso não, chama os melhores da primeira modelagem
  - Utiliza as features resultantes do Recursive Feature Elimination da primeira modelagem
  - Chama **train_model.py** para treinar o modelo
  - Utiliza o **dump_model.py** ([visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/dump_model.py)) para serializar o arquivo da remodelagem em formato `pickle` [visualizar diretório](https://github.com/Menotso/risco-credito/tree/main/FastAPI/model_report)
  - Retorna o modelo resultante da remodelagem
- **feature_engineering.py**: [visualizar]()
  - Recebe e retorna um DataFrame com o feature_engineering executado
  - Recebe também uma lista como parâmetro além do DataFrame, filtrando as colunas que serão retornadas pela função
- **train_model.py**: [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/train_model.py)
  - Treina o modelo
  - Faz encoding de features categóricas com `Target Encoder`
  - Normaliza com `StandardScaler`
  - Faz o split em folds de treino/teste com `StratifiedKFold`
  - Executa validação cruzada com `cross_validate`
  - Salva as métricas em formato `.json` [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/model_report/metrics.json)
  - Salva a feature importance e métricas em formato `.json` [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/model_report/feature_importance.json)
  - Salva a matriz de confusão em imagem `.png` [visualizar](https://github.com/Menotso/risco-credito/blob/main/FastAPI/model_report/confusion_matrix.png)

## Aplicação Prática de Negócio

Se encontra no final do arquivo **modelagem.ipynb** [visualizar código](https://github.com/Menotso/risco-credito/blob/main/Modelagem_ML/modelagem.ipynb)

Após o ajuste do threshold para `0.35`, podemos visualizar a performance do modelo escolhido:

- **Média da Precisão (Weighted):** 92.70%
- **Média da Revocação (Weighted):** 92.85%
- **Média do F1 Score (Weighted):** 92.70%
- **Precisão x Revocação, Área abaixo da Curva:** 90.68%

![xgb_hiperparametros_threshold](https://github.com/Menotso/risco-credito/blob/main/imagens/imagens_MODEL/xgb_hiperparametros_threshold.png)

O Recall nos diz que de todos os clientes que de fato não pagaram, o modelo conseguiu acertar 92.85% dos casos. Contudo, o modelo acertou apenas 92.70% dos casos em que ele previu que os clientes não pagariam.

**Inserindo em um contexto para visualizar a aplicabilidade deste modelo no negócio**

- Como não sabemos em quanto tempo um cliente pagará o empréstimo ou quanto está pagando por período, temos a proposta abaixo:

--------------------------------------------------------------------------------
**Não conceder o valor de empréstimo solicitado pelos clientes inadimplentes.**

--------------------------------------------------------------------------------

## Resultado

- Evitaríamos perder: R$ 5.022.375,00 (5 milhões).
- **Lucro Bruto** de: R$ 2.545.164,06 (2.5 milhões)
- Perderíamos: R$ 1.412.400,00 (1.4 milhão)
- Gerando um **Total Líquido** de: R$ 1.046.735,93 (1 milhão)
- No início concedemos: R$ 21.284.800.00 (21.3 milhões) e terminamos com o saldo final de **R$ 22.331.535.93** (22.3 milhões)
Isso tudo com um baixo custo computacional e também com praticidade na manutenção do modelo para posterior melhorias.

O cenário ideal seria reduzir os falsos negativos à zero. Assim o número de falsos positivos aumentarão substancialmente ao diminuir o threshold, gerando perdas significativas de dinheiro, entretanto, vimos anteriormente que o prejuízo que a classificação errada de um cliente que de fato será inadimplente é quatro vezes maior que de um cliente que pagaria sendo classificado como inadimplente. Com o nosso ajuste do threshold, o modelo agora abrange um pouco mais este prejuízo dos Falsos Negativos.
