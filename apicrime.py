from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Subindo a API de Previsão de Segurança - Marituba
app = FastAPI(title="API de Previsão de Segurança - Marituba")

# Carregando o modelo e os encoders que foi preparado durante o treino
data = joblib.load('modelo_crime.pkl')
modelo = data['modelo']
encoders = data['encoders']
tradutor = data['tradutor']

class Consulta(BaseModel):
    """Estrutura dos dados que vai ser recebido nas consultas."""
    bairro: str
    mes: str
    dia: str
    hora: int

@app.post("/prever_seguranca")
def prever(dados: Consulta):
    """
    Endpoint que processa a consulta, aplica a lógica que foi criado
    e retorna o nível de risco previsto pelo modelo.
    """
    try:
        # 1. Preparando o input: limpo e padronizo os dados para bater com o que o modelo aprendeu
        entrada = pd.DataFrame([{
            'bairro': dados.bairro.lower().strip(),
            'mes': dados.mes.lower().strip(),
            'dia': dados.dia.lower().strip(),
            'hora': str(dados.hora) 
        }])
        
        # 2. Transformando: uso os encoders que foi salvo para traduzir o texto em números
        # O modelo não entende strings, então essa etapa é fundamental
        for col in ['bairro', 'mes', 'dia', 'hora']:
            entrada[col] = encoders[col].transform(entrada[col])
            
        # 3. Predição: o momento em que o modelo faz o cálculo de risco
        pred = modelo.predict(entrada)
        
        # 4. Traduzindo o resultado: volto do formato numérico para o rótulo ("Risco Alto/Normal")
        resultado = tradutor.inverse_transform(pred)[0]
        
        return {
            "status": "sucesso",
            "bairro_consultado": dados.bairro,
            "nivel_de_risco": resultado
        }

    except ValueError as ve:
        # Tratando casos onde o usuário manda um dado (como um bairro) que não existe na base
        return {"erro": "Valor inválido: dado não presente no meu histórico de treino", "detalhe": str(ve)}
    except Exception as e:
        # Lidando com qualquer erro inesperado no processamento
        return {"erro": "Ops, algo deu errado ao processar a predição", "detalhe": str(e)}
