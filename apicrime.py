from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Carregar modelo
data = joblib.load('modelo_crime.pkl')
modelo, encoders, tradutor = data['modelo'], data['encoders'], data['tradutor']

class Consulta(BaseModel):
    bairro: str
    mes: str
    dia: str
    hora: int

@app.post("/prever_seguranca")
def prever(dados: Consulta):
    try:
        entrada = pd.DataFrame([{
            'bairro': dados.bairro.lower().strip(),
            'mes': dados.mes.lower().strip(),
            'dia': dados.dia.lower().strip(),
            'hora': str(dados.hora)
        }])
        
        # Transformar dados
        for col in ['bairro', 'mes', 'dia', 'hora']:
            entrada[col] = encoders[col].transform(entrada[col])
            
        # Predição
        pred = modelo.predict(entrada)
        resultado = tradutor.inverse_transform(pred)[0]
        
        return {"local": dados.bairro, "nivel": resultado}
    except Exception as e:
        return {"erro": "Dados desconhecidos", "detalhe": str(e)}