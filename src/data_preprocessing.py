import os
import pandas as pd
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

def main():
    # Exemplo: ajuste o caminho do seu dataset
    df = pd.read_csv('data/colaboradores.csv')
    
    # Filtrar apenas colaboradores que pediram desligamento voluntário
    desligados = df[df['status_code'] == 4].copy()
    desligados = desligados[['colaborador_id', 'dt_yearmonth']].drop_duplicates()
    desligados = desligados.rename(columns={'dt_yearmonth': 'dt_desligamento'})
    
    # Juntar a data de desligamento voluntário ao histórico de cada colaborador
    df = df.merge(desligados, on='colaborador_id', how='left')
    
    # Calcular meses até o desligamento voluntário
    df['dt_yearmonth'] = pd.to_datetime(df['dt_yearmonth'])
    df['dt_desligamento'] = pd.to_datetime(df['dt_desligamento'])
    df['meses_ate_desligamento'] = ((df['dt_desligamento'] - df['dt_yearmonth'])/pd.offsets.MonthBegin(1)).astype('float')
    
    # Remover registros após o desligamento voluntário
    df = df[df['meses_ate_desligamento'] >= 0]
    
    # Remover colaboradores que nunca pediram desligamento voluntário
    df = df[~df['meses_ate_desligamento'].isna()]
    
    # Exemplo: salvar features para treino
    features = [col for col in df.columns if col not in ['colaborador_id', 'dt_yearmonth', 'dt_desligamento', 'status_code', 'meses_ate_desligamento']]
    df[features + ['meses_ate_desligamento']].to_csv('data/processed.csv', index=False)

if __name__ == '__main__':
    main()
