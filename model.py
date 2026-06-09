import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def classificar_risco(frequencia, limiar):
    """
    Define a categoria de risco baseada na frequência de ocorrências
    em comparação com o limiar estatístico (quartil 0.75).
    """
    return "Risco Alto" if frequencia > limiar else "Risco Normal"

# 1. CARREGAMENTO E LIMPEZA DOS DADOS
# Leitura da base de dados bruta
df = pd.read_excel("07_04_26.xlsx")

# Renomeação para facilitar a manipulação
df = df.rename(columns={
    'BAIRRO(S)': 'bairro',
    'MÊS DO FATO': 'mes',
    'DIA DA SEMANA': 'dia',
    'HORA DO FATO': 'hora_col'
})

# Sanitização: limpeza de espaços em branco e padronização para minúsculas
# Isso evita que 'Centro ' e 'centro' sejam vistos como diferentes
df['bairro'] = df['bairro'].astype(str).str.lower().str.strip()
df['mes'] = df['mes'].astype(str).str.lower().str.strip()
df['dia'] = df['dia'].astype(str).str.lower().str.strip()

# Extração da hora do fato para análise numérica
df['hora'] = pd.to_datetime(df['hora_col'].astype(str)).dt.hour

# 2. DEFINIÇÃO ESTATÍSTICA DO RISCO
# Agrupamento para calcular a frequência de crimes por combinação de contexto
contagem = df.groupby(['bairro', 'mes', 'dia', 'hora']).size().reset_index(name='frequencia')

# Cálculo do limiar: o que está acima do 3º quartil é considerado 'Risco Alto'
limiar = contagem['frequencia'].quantile(0.75)
contagem['status'] = contagem['frequencia'].apply(lambda x: classificar_risco(x, limiar))

# 3. PREPARAÇÃO PARA O TREINAMENTO (ENCODING)
# Merge dos dados para associar o status de risco a cada entrada original
df = df.merge(contagem, on=['bairro', 'mes', 'dia', 'hora'], how='left')

# Transformação do alvo (status) em valores numéricos para o modelo
le_risco = LabelEncoder()
df['status_num'] = le_risco.fit_transform(df['status'])

# Transformação das colunas de texto em números (Categorical Encoding)
# Os encoders são salvos no dicionário para usar posteriomente  na API
encoders = {}
cols_treino = ['bairro', 'mes', 'dia', 'hora']

for col in cols_treino:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 4. TREINAMENTO DO MODELO E SERIALIZAÇÃO
# Definição das variáveis preditoras (X) e alvo (y)
X = df[cols_treino]
y = df['status_num']

# Treino do classificador Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Salvamento do modelo e dos encoders via joblib
# O 'tradutor' (le_risco) permite transformar o número predito de volta para 'Risco Alto/Normal'
joblib.dump({
    'modelo': modelo, 
    'encoders': encoders, 
    'tradutor': le_risco
}, 'modelo_crime.pkl')

print("Modelo treinado com sucesso e salvo como 'modelo_crime.pkl'!")
