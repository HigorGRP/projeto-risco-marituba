🛡️ Sistema de Classificação de Risco Criminal - Marituba/PA
Este projeto utiliza Inteligência Artificial para classificar o nível de risco de ocorrências em diferentes locais, com base em padrões históricos de frequência. A solução analisa dados reais para identificar se uma combinação de local e horário apresenta um Risco Alto ou Risco Normal, auxiliando na visualização de padrões de segurança urbana.

📍 Motivação e Contexto Local
Moro em Marituba/PA e convivo diariamente com os desafios da segurança pública na região. Este projeto nasceu da vontade de aplicar Ciência de Dados para transformar números brutos em informações úteis. Em vez de apenas prever "tipos de crime", o sistema foca em classificar a probabilidade de risco, ajudando a entender onde e quando a incidência de ocorrências se torna estatisticamente mais relevante.


📋 Sobre o Projeto
O objetivo é transformar dados históricos em uma ferramenta de suporte à decisão. O projeto demonstra o ciclo completo de um cientista de dados:

Limpeza e Tratamento: Padronização de bairros e horas para garantir a integridade dos dados.

Feature Engineering: Cálculo de limiares estatísticos (quantile 0.75) para definir o que é "Risco Alto".

Modelagem: Uso do algoritmo Random Forest Classifier para aprender a classificar o nível de periculosidade.

Deploy: Disponibilização de uma API REST para consultas de risco em tempo real.

🏗️ Arquitetura do Sistema
Abaixo está o fluxo lógico de como os dados são processados desde a planilha bruta até a resposta da API:
graph LR
    A[Dados Excel] --> B(Limpeza/Pandas)
    B --> C{Classificação de Risco}
    C -->|Acima do Limiar| D[Risco Alto]
    C -->|Abaixo do Limiar| E[Risco Normal]
    D & E --> F[Treino Random Forest]
    F --> G[Arquivo .pkl]
    H[Entrada Usuário] --> I[API FastAPI]
    I --> J(Normalização/Encoder)
    J --> K[Predição do Modelo]
    K --> L[Retorno JSON]


🛠️ Tecnologias Utilizadas
Python 3.10+

Pandas: Manipulação e análise de dados.

Scikit-Learn: Treinamento do modelo de classificação.

FastAPI: Criação da API de consulta.

Joblib: Serialização do modelo e dos encoders.

🚀 Como Executar
Instale as dependências:
pip install pandas scikit-learn fastapi uvicorn joblib openpyxl

Treine o modelo:
python model.py

Inicie a API:
uvicorn apicrime:app --reload

Consulte a API:
Acesse a documentação em http://127.0.0.1:8000/docs e faça um POST no endpoint /prever_seguranca com o JSON:
JSON
{
  "bairro": "centro",
  "mes": "janeiro",
  "dia": "segunda-feira",
  "hora": 14
}

🛠️ Diferencial Técnico: A Lógica de Classificação
Para garantir a confiabilidade, implementamos uma lógica de decisão robusta:

Classificação Estatística: O sistema utiliza o quartil superior (0.75) da frequência de crimes como o "limiar de perigo".

Sanitização: Implementação de .lower() e .strip() em todas as entradas, garantindo que variações de digitação não impactem o resultado.

Consistência: O uso de Encoders serializados garante que a tradução dos dados na API seja idêntica à do treinamento, evitando erros de predição.
