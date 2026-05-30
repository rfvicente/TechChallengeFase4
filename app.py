
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Modelo ML" / "modelo_v2_techchallenge.pkl"
DATA_PATH = BASE_DIR / "Dados" / "Obesity.csv"

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Diagnóstico de Obesidade", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# 2. CARREGAMENTO DE MODELOS E DADOS
@st.cache_resource
def carregar_modelos():
    return joblib.load(MODEL_PATH)

@st.cache_data
def carregar_dados():
    return pd.read_csv(DATA_PATH)

modelo = carregar_modelos()
df_raw = carregar_dados()

# Dicionários de tradução e ordenação
traducao_diagnostico = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Grau I',
    'Overweight_Level_II': 'Sobrepeso Grau II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}

ordem_categorias = [
    'Abaixo do Peso', 'Peso Normal', 'Sobrepeso Grau I',
    'Sobrepeso Grau II', 'Obesidade Tipo I', 'Obesidade Tipo II', 'Obesidade Tipo III'
]

# 3. NAVEGAÇÃO POR ABAS NO TOPO
tab1, tab2 = st.tabs(["🩺 Diagnóstico Inteligente", "📊 Dashboards Analíticos"])

# ==========================================
# ABA 1: DIAGNÓSTICO
# ==========================================
with tab1:
    st.title("🏥 Sistema de Prevenção e Diagnóstico de Obesidade")
    st.markdown("Preencha os hábitos do paciente para o modelo prever o risco comportamental.")

    with st.form("form_diagnostico"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Dados Biológicos")
            genero_pt = st.selectbox("Sexo Biológico (Apenas Informativo)", ["Feminino", "Masculino"])
            idade = st.number_input("Idade", min_value=10, max_value=100, value=25)

            st.markdown("---")
            st.markdown("**Dados Físicos (Informativo)**")
            altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.20, value=1.70, format="%.2f")
            peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, format="%.1f")
            imc = peso / (altura ** 2)
            st.info(f"📊 IMC Calculado: {imc:.1f}")

        with col2:
            st.header("Histórico e Dieta")
            hist_familiar = st.selectbox("Histórico Familiar de Sobrepeso?", ["Sim", "Não"])
            comida_calorica = st.selectbox("Consome alimentos calóricos?", ["Sim", "Não"])
            vegetais = st.slider("Frequência de vegetais (1-3)", 1, 3, 2)
            refeicoes = st.slider("Refeições principais por dia", 1, 4, 3)
            lanches = st.selectbox("Lanches entre refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])

        with col3:
            st.header("Estilo de Vida")
            agua = st.slider("Consumo de água (L/dia)", 1, 3, 2)
            exercicio = st.slider("Atividade física (dias/semana)", 0, 3, 1)
            telas = st.slider("Tempo em telas (0-2)", 0, 2, 1)
            alcool = st.selectbox("Consumo de álcool?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
            transporte = st.selectbox("Meio de transporte", ["Automóvel", "Moto", "Bicicleta", "Transporte Público", "A pé"])
            monitora_calorias = st.selectbox("Monitora calorias?", ["Sim", "Não"])
            fumante = st.selectbox("Fumante?", ["Sim", "Não"])

        st.markdown("<br>", unsafe_allow_html=True)
        btn_analisar = st.form_submit_button("🔮 Gerar Diagnóstico Inteligente", type="primary", use_container_width=True)

    if btn_analisar:
        mapa_sim_nao = {"Não": 0, "Sim": 1}
        mapa_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
        mapa_transporte = {"Automóvel": "Automobile", "Moto": "Motorbike", "Bicicleta": "Bike", "Transporte Público": "Public_Transportation", "A pé": "Walking"}

        dados_paciente = {col: 0 for col in modelo.feature_names_in_}

        if 'Age' in dados_paciente: dados_paciente['Age'] = idade
        if 'family_history' in dados_paciente: dados_paciente['family_history'] = mapa_sim_nao[hist_familiar]
        if 'FAVC' in dados_paciente: dados_paciente['FAVC'] = mapa_sim_nao[comida_calorica]
        if 'FCVC' in dados_paciente: dados_paciente['FCVC'] = vegetais
        if 'NCP' in dados_paciente: dados_paciente['NCP'] = refeicoes
        if 'CAEC' in dados_paciente: dados_paciente['CAEC'] = mapa_freq[lanches]
        if 'CH2O' in dados_paciente: dados_paciente['CH2O'] = agua
        if 'FAF' in dados_paciente: dados_paciente['FAF'] = exercicio
        if 'TUE' in dados_paciente: dados_paciente['TUE'] = telas
        if 'CALC' in dados_paciente: dados_paciente['CALC'] = mapa_freq[alcool]
        if 'SCC' in dados_paciente: dados_paciente['SCC'] = mapa_sim_nao[monitora_calorias]
        if 'SMOKE' in dados_paciente: dados_paciente['SMOKE'] = mapa_sim_nao[fumante]

        col_transp = f"Transp_{mapa_transporte[transporte]}"
        if col_transp in dados_paciente:
            dados_paciente[col_transp] = 1

        df_modelo = pd.DataFrame([dados_paciente])
        df_modelo = df_modelo[modelo.feature_names_in_]

        pred_raw = modelo.predict(df_modelo)[0]
        classes_raw = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

        try:
            indice = int(pred_raw)
            resultado_en = classes_raw[indice]
        except (ValueError, TypeError):
            resultado_en = str(pred_raw)

        resultado_pt = traducao_diagnostico.get(resultado_en, resultado_en)

        st.markdown("---")
        st.markdown("<div id='ancora-resultado'></div>", unsafe_allow_html=True)

        st.subheader("📋 Resultado da IA")
        if "Obesity" in resultado_en:
            st.error(f"### Risco Comportamental: **{resultado_pt}**")
            st.markdown("⚠️ **Alerta:** Os hábitos indicam forte tendência à obesidade. Recomenda-se acompanhamento nutricional.")
        elif "Overweight" in resultado_en:
            st.warning(f"### Risco Comportamental: **{resultado_pt}**")
            st.markdown("🧡 **Aviso:** Paciente com hábitos que levam ao sobrepeso. Sugere-se ajuste no estilo de vida.")
        else:
            st.success(f"### Risco Comportamental: **{resultado_pt}**")
            st.markdown("✅ **Status:** Os hábitos atuais do paciente são condizentes com a manutenção de um peso saudável.")

        st.markdown("---")
        st.markdown("**🔍 Nível de Confiança do Modelo (Probabilidades):**")

        probabilidades = modelo.predict_proba(df_modelo)[0]

        df_probs = pd.DataFrame({
            'Diagnóstico': [traducao_diagnostico.get(c, c) for c in classes_raw],
            'Probabilidade (%)': probabilidades * 100
        }).sort_values(by='Probabilidade (%)', ascending=True)

        fig = px.bar(
            df_probs,
            x='Probabilidade (%)',
            y='Diagnóstico',
            orientation='h',
            text_auto='.1f',
            color='Probabilidade (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_title="Probabilidade (%)",
            yaxis_title="",
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        timestamp_atual = time.time()
        components.html(
            f"""
            <script>
                // Execução {timestamp_atual}
                window.parent.document.getElementById('ancora-resultado').scrollIntoView({{behavior: 'smooth'}});
            </script>
            """,
            height=0
        )

# ==========================================
# ABA 2: DASHBOARDS
# ==========================================
with tab2:
    st.title("📊 Dashboards Analíticos")

    opcoes_graficos = [
        "Distribuição Geral",
        "Obesidade por Gênero",
        "Impacto do Histórico Familiar",
        "Matriz de Correlação",
        "Principais Fatores (ML)"
    ]
    opcao = st.radio("Escolha a visão:", opcoes_graficos, horizontal=True)

    df_plot = df_raw.copy()

    if opcao == "Distribuição Geral":
        df_plot['Obesity'] = df_plot['Obesity'].map(traducao_diagnostico)
        df_pie = df_plot['Obesity'].value_counts().reset_index()
        df_pie.columns = ['Obesity', 'count']
        fig_dash = px.pie(
            df_pie,
            names='Obesity',
            values='count',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Distribuição das Categorias de Peso na Base Original"
        )
        st.plotly_chart(fig_dash, use_container_width=True)

    elif opcao == "Obesidade por Gênero":
        df_plot['Obesity'] = df_plot['Obesity'].map(traducao_diagnostico)
        df_plot['Gender'] = df_plot['Gender'].map({'Female': 'Feminino', 'Male': 'Masculino'})

        fig_dash = px.histogram(
            df_plot,
            x='Obesity',
            color='Gender',
            barmode='stack',
            color_discrete_map={'Feminino': '#9BB7D4', 'Masculino': '#2C4F73'},
            title="Distribuição de Categorias de Peso por Sexo Biológico"
        )
        fig_dash.update_layout(
            xaxis_title="Categoria de Peso",
            yaxis_title="Quantidade de Pacientes",
            legend_title="Gênero",
            xaxis={'categoryorder':'array', 'categoryarray': ordem_categorias}
        )
        st.plotly_chart(fig_dash, use_container_width=True)

    elif opcao == "Impacto do Histórico Familiar":
        df_plot['Obesity'] = df_plot['Obesity'].map(traducao_diagnostico)
        df_plot['family_history'] = df_plot['family_history'].map({'yes': 'Sim', 'no': 'Não'})

        fig_dash = px.histogram(
            df_plot,
            x='Obesity',
            color='family_history',
            barmode='stack',
            color_discrete_map={'Sim': '#2C4F73', 'Não': '#9BB7D4'},
            title="A Influência do Histórico Familiar na Obesidade"
        )
        fig_dash.update_layout(
            xaxis_title="Categoria de Peso",
            yaxis_title="Quantidade de Pacientes",
            legend_title="Histórico Familiar?",
            xaxis={'categoryorder':'array', 'categoryarray': ordem_categorias}
        )
        st.plotly_chart(fig_dash, use_container_width=True)

    elif opcao == "Matriz de Correlação":
        st.markdown("### Matriz de Correlação de Pearson")
        st.markdown("Mostra a correlação matemática direta entre todas as variáveis do conjunto de dados.")

        df_corr = df_plot.copy()

        # Dicionário para traduzir as colunas do dataset para Português
        mapa_colunas_pt = {
            'Gender': 'Gênero',
            'Age': 'Idade',
            'Height': 'Altura',
            'Weight': 'Peso',
            'family_history': 'Hist. Familiar',
            'FAVC': 'Comida Calórica',
            'FCVC': 'Consumo Vegetais',
            'NCP': 'Refeições/Dia',
            'CAEC': 'Lanches',
            'SMOKE': 'Fumante',
            'CH2O': 'Água/Dia',
            'SCC': 'Monitora Calorias',
            'FAF': 'Ativ. Física',
            'TUE': 'Tempo Tela',
            'CALC': 'Álcool',
            'MTRANS': 'Transporte',
            'Obesity': 'Obesidade'
        }

        # Renomeia as colunas antes de calcular a correlação
        df_corr = df_corr.rename(columns=mapa_colunas_pt)

        le = LabelEncoder()
        for col in df_corr.columns:
            if df_corr[col].dtype == 'object':
                df_corr[col] = le.fit_transform(df_corr[col])

        corr = df_corr.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmax=0.6,
            vmin=-0.6,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            ax=ax_corr
        )
        ax_corr.set_title("Matriz de Correlação (Variáveis Tratadas)", fontsize=16)
        st.pyplot(fig_corr)

    elif opcao == "Principais Fatores (ML)":
        st.markdown("### Principais Fatores Comportamentais que Levam à Obesidade")
        st.markdown("Ranking de importância das variáveis baseada nas árvores de decisão do modelo em produção.")

        importancias = modelo.feature_importances_
        features = modelo.feature_names_in_

        mapa_features_pt = {
            'Age': 'Idade',
            'family_history': 'Hist. Familiar',
            'FAVC': 'Comida Calórica',
            'FCVC': 'Consumo Vegetais',
            'NCP': 'Refeições/Dia',
            'CAEC': 'Lanches',
            'SMOKE': 'Fumante',
            'CH2O': 'Água/Dia',
            'SCC': 'Monitora Calorias',
            'CALC': 'Álcool',
            'FAF': 'Ativ. Física',
            'TUE': 'Tempo Tela',
            'Transp_Automobile': 'Transp. Carro',
            'Transp_Bike': 'Transp. Bicicleta',
            'Transp_Motorbike': 'Transp. Moto',
            'Transp_Public_Transportation': 'Transp. Público',
            'Transp_Walking': 'A Pé'
        }

        df_importancia = pd.DataFrame({
            'Feature Original': features,
            'Importância': importancias
        })

        df_importancia['Fator'] = df_importancia['Feature Original'].map(mapa_features_pt).fillna(df_importancia['Feature Original'])
        df_importancia = df_importancia.sort_values(by='Importância', ascending=True)

        fig_imp = px.bar(
            df_importancia,
            x='Importância',
            y='Fator',
            orientation='h',
            color='Importância',
            color_continuous_scale='inferno'
        )
        fig_imp.update_layout(
            xaxis_title="Nível de Importância",
            yaxis_title="",
            showlegend=False,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
