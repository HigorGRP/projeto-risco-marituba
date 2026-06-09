import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Função mantida para organização e clareza
def classificar_risco(frequencia, limiar):
    if frequencia > limiar:
        return "Risco Alto"
    else:
        return "Risco Normal"

# 1. Carregamento e Limpeza
df = pd.read_excel("07_04_26.xlsx")

df = df.rename(columns={
    'BAIRRO(S)': 'bairro',
    'MÊS DO FATO': 'mes',
    'DIA DA SEMANA': 'dia',
    'HORA DO FATO': 'hora_col'
})

df['bairro'] = df['bairro'].astype(str).str.lower().str.strip()
df['mes'] = df['mes'].astype(str).str.lower().str.strip()
df['dia'] = df['dia'].astype(str).str.lower().str.strip()
df['hora'] = pd.to_datetime(df['hora_col'].astype(str)).dt.hour

# 2. Definição do Risco
contagem = df.groupby(['bairro', 'mes', 'dia', 'hora']).size().reset_index(name='frequencia')
limiar = contagem['frequencia'].quantile(0.75)

contagem['status'] = contagem['frequencia'].apply(lambda x: classificar_risco(x, limiar))

# 3. Preparação para o Treino
df = df.merge(contagem, on=['bairro', 'mes', 'dia', 'hora'], how='left')
le_risco = LabelEncoder()
df['status_num'] = le_risco.fit_transform(df['status'])

encoders = {}
cols_treino = ['bairro', 'mes', 'dia', 'hora']

for col in cols_treino:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 4. Treino e Exportação
X = df[cols_treino]
y = df['status_num']
modelo = RandomForestClassifier().fit(X, y)

joblib.dump({'modelo': modelo, 'encoders': encoders, 'tradutor': le_risco}, 'modelo_crime.pkl')
print("Modelo treinado com sucesso!")