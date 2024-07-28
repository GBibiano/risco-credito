from pydantic import BaseModel


# 2. Definir a validação de dados
class Validate(BaseModel):
    val_idade_cliente: int
    val_renda_cliente: int
    val_posse_residencia_cliente: str
    val_tempo_emprego_cliente: int
    val_finalidade_emprestimo: str
    #val_nota_emprestimo: str #
    val_valor_emprestimo: int
    #val_taxa_juros_emprestimo: float #
    #val_percentual_renda_emprestimo: float #
    #val_historico_inadimplencia_cliente: int #
    #val_tempo_credito_cliente: int #
    #val_status_emprestimo: int #
