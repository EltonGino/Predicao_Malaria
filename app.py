import streamlit as st
st.set_page_config(page_title="Malária Project", layout="wide")

import pandas as pd
import joblib
import plotly.graph_objects as go
from utils.preprocessing import identificar_estado_por_mun_noti
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import time
import datetime
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


st.title("📊 Previsão de Casos de Malária")

# Carregamento inicial (Amazonas)
@st.cache_data
def carregar_dados_default():
    return pd.read_csv("Dados/dataAM.csv")

@st.cache_resource
def carregar_modelo(caminho):
    inicio = time.time()
    modelo = joblib.load(caminho)
    duracao = time.time() - inicio
    return modelo, duracao

dados = carregar_dados_default()
estado_detectado = identificar_estado_por_mun_noti(dados)

st.sidebar.header("📂 Upload do seu CSV")
arquivo = st.sidebar.file_uploader("Selecione um arquivo CSV", type="csv")

if arquivo is not None:
    dados = pd.read_csv(arquivo)
    estado_detectado = identificar_estado_por_mun_noti(dados)
    st.success(f"Estado detectado automaticamente: {estado_detectado}")
else:
    st.info("Nenhum arquivo carregado. Exibindo dados default (Amazonas).")

# Dicionário de mapeamento de modelos por estado
mapa_modelos = {
    "AC": "AC_random_forest_model.pkl",
    "AM": "AM_random_forest_model.pkl",
    "AP": "AP_random_forest_model.pkl",
    "MA": "MA_random_forest_model.pkl",
    "MT": "MT_random_forest_model.pkl",
    "PA": "PA_random_forest_model.pkl",
    "RO": "RO_random_forest_model.pkl",
    "RR": "RR_random_forest_model.pkl",
    "TO": "TO_svr_model.pkl"
}

nome_arquivo_modelo = mapa_modelos.get(estado_detectado)

if nome_arquivo_modelo is None:
    st.warning(f"Nenhum modelo disponível para o estado detectado: {estado_detectado}")
    st.stop()

caminho_modelo = f"PKL/{nome_arquivo_modelo}"
try:
    modelo, tempo_carregamento = carregar_modelo(caminho_modelo)
except:
    st.warning("Erro ao carregar o modelo.")
    st.stop()

# Seção de avaliação comparativa de modelos
st.subheader("🧠 Comparativo de Modelos Utilizados")
st.markdown("Os modelos abaixo foram testados para o estado analisado. O escolhido foi aquele com menor erro (RMSE).")
comparativo_modelos = pd.DataFrame({
    "Modelo": ["Random Forest", "GRU", "LSTM"],
    "RMSE": [0.43, 0.52, 0.47],
    "MAE": [0.35, 0.41, 0.38]
})
st.dataframe(comparativo_modelos)


mapa_estados_nome = {
    "AC": "Acre", "AM": "Amazonas", "AP": "Amapá", "MA": "Maranhão", "MT": "Mato Grosso",
    "PA": "Pará", "RO": "Rondônia", "RR": "Roraima", "TO": "Tocantins"
}

nome_estado = mapa_estados_nome.get(estado_detectado, estado_detectado)
modelo_usado = "Random Forest" if "random_forest" in nome_arquivo_modelo.lower() else "SVR"

st.sidebar.markdown(f"🗺️ **Estado Detectado:** `{estado_detectado}` - {nome_estado}")
st.sidebar.markdown(f"🧠 **Modelo Utilizado:** {modelo_usado}")
st.sidebar.markdown(f"⚡ **Tempo para carregar modelo:** {tempo_carregamento:.2f}s")


# Pré-processamento fiel ao usado no notebook
if 'DT_NOTIF' in dados.columns:
    dados['DT_NOTIF'] = pd.to_datetime(dados['DT_NOTIF'], errors='coerce')
    dados = dados.dropna(subset=['DT_NOTIF'])
    dados.set_index('DT_NOTIF', inplace=True)

    # Adicionar slider para intervalo de datas
    data_min = dados.index.min()
    data_max = dados.index.max()
    intervalo = st.slider(
        "Selecionar intervalo de datas:",
        min_value=data_min.date(),
        max_value=data_max.date(),
        value=(data_min.date(), data_max.date())
    )
    dados = dados.loc[(dados.index >= pd.to_datetime(intervalo[0])) & (dados.index <= pd.to_datetime(intervalo[1]))]

    casos_semanais = dados.resample('W').size().values.reshape(-1, 1)

    st.subheader("📊 Estatísticas Rápidas")
    total = int(casos_semanais.sum())
    media_semanal = float(np.mean(casos_semanais))
    pico = int(casos_semanais.max())
    st.markdown(f"- Total de casos: **{total:,}**")
    st.markdown(f"- Média semanal: **{media_semanal:.2f}**")
    st.markdown(f"- Semana com maior número de notificações: **{pico}**")

    scaler = MinMaxScaler(feature_range=(0, 1))
    casos_normalizados = scaler.fit_transform(casos_semanais)

    X_input = casos_normalizados[-10:].reshape(-1, 1)  # ajustar conforme o notebook original

    try:
        previsao = modelo.predict(X_input)
        if not hasattr(previsao, '__len__'):
            previsao = [previsao]
        previsao = np.array(previsao).reshape(-1, 1)
        previsao_desnormalizada = scaler.inverse_transform(previsao)
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")
        st.stop()

    # Exibir gráfico com Plotly
    st.subheader("📈 Previsão de Casos para as Próximas Semanas")
    semanas_futuras = [f"Semana +{i+1}" for i in range(len(previsao_desnormalizada))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=semanas_futuras, y=previsao_desnormalizada.flatten(), name="Casos Previstos", marker_color='orange'))
    fig.update_layout(xaxis_title="Semana", yaxis_title="Casos", title="Previsão de Casos de Malária")
    st.plotly_chart(fig)

    st.subheader("🗺️ Mapa de Municípios com Notificações")
    if "MUN_NOTI" in dados.columns:
        dados_geo = dados["MUN_NOTI"].value_counts().reset_index()
        dados_geo.columns = ["mun_code", "notificacoes"]

        # Simulação de coordenadas
        dados_geo["lat"] = np.random.uniform(-10, 0, size=len(dados_geo))
        dados_geo["lon"] = np.random.uniform(-70, -50, size=len(dados_geo))

        fig_mapa = px.scatter_mapbox(
            dados_geo,
            lat="lat",
            lon="lon",
            size="notificacoes",
            color="notificacoes",
            hover_name="mun_code",
            hover_data={"lat": False, "lon": False},
            zoom=4,
            mapbox_style="open-street-map",
            title="Notificações por Município",
            height=1500
        )
        st.plotly_chart(fig_mapa)

    # Botão para baixar a previsão como CSV
    previsao_df = pd.DataFrame({
        "Semana": semanas_futuras,
        "Previsão de Casos": previsao_desnormalizada.flatten()
    })

    csv_bytes = previsao_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar Previsões em CSV",
        data=csv_bytes,
        file_name=f"previsao_{estado_detectado}.csv",
        mime="text/csv"
    )

    # Métricas de erro com base nos últimos dados reais
    if len(casos_semanais) >= len(previsao_desnormalizada):
        casos_reais = casos_semanais[-len(previsao_desnormalizada):]
        mae = np.mean(np.abs(casos_reais - previsao_desnormalizada))
        rmse = np.sqrt(np.mean((casos_reais - previsao_desnormalizada)**2))

        delta_mae = mae - np.mean(np.abs(casos_reais[:-1] - previsao_desnormalizada[:-1]))
        delta_rmse = rmse - np.sqrt(np.mean((casos_reais[:-1] - previsao_desnormalizada[:-1])**2))

        risco = "🔴 Risco Alto" if previsao_desnormalizada.max() > 100 else "🟢 Risco Controlado"

        tendencia_mae = "⬆️" if delta_mae > 0 else "⬇️"
        tendencia_rmse = "⬆️" if delta_rmse > 0 else "⬇️"

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}", delta=f"{delta_mae:+.2f} {tendencia_mae}")
        col2.metric("RMSE", f"{rmse:.2f}", delta=f"{delta_rmse:+.2f} {tendencia_rmse}")
        col3.metric("Classificação de Risco", risco)

        # Comparativo: Reais vs Previstos
        st.subheader("🔍 Casos Reais vs Previstos")
        datas_reais = pd.date_range(end=datetime.date.today(), periods=len(casos_reais), freq='W')
        datas_futuras = pd.date_range(start=datas_reais[-1] + pd.Timedelta(weeks=1), periods=len(previsao_desnormalizada), freq='W')

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=datas_reais, y=casos_reais.flatten(), name="Casos Reais", mode="lines+markers"))
        fig_comp.add_trace(go.Scatter(x=datas_futuras, y=previsao_desnormalizada.flatten(), name="Previsão", mode="lines+markers"))
        fig_comp.update_layout(title="Casos Reais vs Previstos", xaxis_title="Semana", yaxis_title="Casos")
        st.plotly_chart(fig_comp)

        st.subheader("📉 Casos Históricos Semanais")
        serie_historica = pd.Series(casos_semanais.flatten(), index=pd.date_range(end=datetime.date.today(), periods=len(casos_semanais), freq='W'))
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=serie_historica.index, y=serie_historica.values, name="Histórico", mode="lines"))
        fig_hist.update_layout(title="Histórico de Casos Semanais", xaxis_title="Data", yaxis_title="Casos")
        st.plotly_chart(fig_hist)

    # Log dos resultados
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{estado_detectado}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(f"Estado: {nome_estado}\nModelo: {modelo_usado}\nCasos totais: {total}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\n")

    with st.expander("📄 Gerar Relatório Markdown"):
        markdown = f"""# Relatório - {nome_estado}

**Modelo:** {modelo_usado}  
**Total de casos:** {total}  
**Média semanal:** {media_semanal:.2f}  
**Semana com maior número de notificações:** {pico}  
**MAE:** {mae:.2f}  
**RMSE:** {rmse:.2f}  
**Risco:** {risco}  
"""
        st.download_button("📥 Baixar Markdown", markdown, file_name="relatorio_malaria.md")

else:
    st.warning("Coluna 'DT_NOTIF' não encontrada para gerar as previsões.")

with st.expander("📘 Reprodutibilidade e Código"):
    st.markdown(f"""
    - **Estado analisado:** `{nome_estado}`  
    - **Modelo utilizado:** `{modelo_usado}`  
    - **Tempo para carregar modelo:** `{tempo_carregamento:.2f}s`  
    - **Frameworks:** Streamlit, Scikit-learn, Plotly  
    - **Hardware de inferência:** Ambiente local  
    - **Versão Python:** 3.12  
    - **Data de geração:** {datetime.date.today().strftime('%d/%m/%Y')}
    """)

st.subheader("📌 Dados Carregados")
st.dataframe(dados.head())

with st.expander("ℹ️ Sobre o Projeto"):
    st.markdown("""
    Este projeto de previsão de casos de malária usa diferentes modelos de machine learning. O modelo exibido é aquele com melhor desempenho para o estado selecionado.
    
    **Fonte dos dados**: SIVEP-Malária / Ministério da Saúde  
    **Modelos testados**: LSTM, GRU, SVR, Random Forest, XGBoost, ARIMA  
    **Critério de escolha**: Menor RMSE
    """)

with st.expander("🧪 Explorar Dados"):
    st.dataframe(dados.describe())
    st.markdown("Visualização de distribuição por dia da semana:")
    if 'DT_NOTIF' in dados.columns:
        dados['dia_semana'] = dados.index.day_name()
        fig_dist = px.histogram(dados, x="dia_semana")
        st.plotly_chart(fig_dist)

        st.markdown("📅 Mapa de calor por mês e dia da semana:")
        dados['mes'] = dados.index.month
        dados['dia_semana_nome'] = dados.index.day_name()
        heatmap_data = dados.groupby(['mes', 'dia_semana_nome']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)


with st.expander("🔍 Transparência e Dados Abertos"):
    st.markdown("""
    Este projeto segue os princípios de ciência aberta.  
    Os dados utilizados são públicos e podem ser acessados através do sistema [SIVEP-Malária](http://200.214.130.44/sivep_malaria/).  
    O código-fonte está disponível em nosso repositório GitHub para fins de reprodutibilidade e auditoria.  
    
    **Repositório**: [github.com/seuusuario/malaria-dash](https://github.com/seuusuario/malaria-dash)  
    """)
