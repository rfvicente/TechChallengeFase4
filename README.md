# 🏥 Tech Challenge Fase 4: Sistema de Prevenção e Diagnóstico de Obesidade

Bem-vindo ao repositório do **Sistema de Prevenção e Diagnóstico de Obesidade**. 
Este projeto foi desenvolvido como parte da avaliação do programa de Pós-Graduação em Data Analytics. 
O objetivo principal é prever o risco de diferentes níveis de obesidade com base no estilo de vida, dieta e histórico clínico dos pacientes.

🔗 **https://tech-challenge-fase4-rm367826.streamlit.app/**

## 🎯 O Problema de Negócio e a Abordagem Analítica

Modelos preditivos tradicionais de obesidade muitas vezes sofrem de *Data Leakage* (vazamento de dados) ao utilizar o Peso e a Altura como *features* principais, o que transforma o modelo em uma mera calculadora de IMC hipertrofiada. 

Para contornar isso e focar na **prevenção**, a abordagem analítica deste projeto tomou a decisão consciente de remover variáveis de medidas físicas diretas durante o treinamento. A Inteligência Artificial aprendeu a diagnosticar o risco utilizando exclusivamente **fatores comportamentais, genéticos e alimentares** (como consumo de vegetais, tempo de tela, atividade física e histórico familiar). 

Isso entrega um valor real de negócio e saúde: permitir que médicos e nutricionistas atuem preventivamente com base nos hábitos do paciente antes que o ganho de peso se consolide.

## 🛠️ Arquitetura e Tecnologias Utilizadas

O pipeline de dados e o deploy foram construídos utilizando as seguintes ferramentas:

* **Linguagem:** Python 3.12+
* **Machine Learning:** Scikit-Learn, LightGBM / XGBoost, Joblib
* **Manipulação de Dados:** Pandas, NumPy
* **Front-end / Web App:** Streamlit
* **Visualização de Dados:** Plotly Express, Matplotlib, Seaborn
* **Deploy:** Streamlit Community Cloud

## 📁 Estrutura do Repositório

O repositório está organizado de forma mais limpa em pastas específicas:

* `app.py`: Código-fonte principal do aplicativo web (Front-end e lógica de inferência).
* `Dados/Obesity.csv`: Base de dados original, utilizada para popular a aba de Dashboards Analíticos.
* `Modelo ML/modelo_v2_techchallenge.pkl`: Modelo de Machine Learning final, treinado e empacotado.
* `Notebooks Python/`: Pasta destinada a notebooks Python com código fonte.
* `requirements.txt`: Lista de dependências e bibliotecas para o deploy na nuvem.
