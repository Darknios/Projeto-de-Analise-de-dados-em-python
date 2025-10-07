import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import joblib
import numpy as np
import datetime
from xgboost import XGBRegressor
import sklearn


# ============================
# CONFIGURAÇÕES DO APP
# ============================
st.set_page_config(page_title="Dashboard de Crimes", layout="wide")
st.title("Dashboard de Crimes em Pernambuco")

# ============================
# CARREGAR DATASET
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/shpctac1001c/Downloads/Projeto PI 3.0/Projeto PI/dataset_ocorrencias_delegacia_5.csv")

    df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"], errors="coerce")
    df["mes"] = df["data_ocorrencia"].dt.month_name()
    df["ano"] = df["data_ocorrencia"].dt.year
    return df

df = load_data()

# ============================
# FILTROS
# ============================
col1, col2 = st.sidebar.columns(2)
anos = st.sidebar.multiselect("Selecione o ano", options=df["ano"].unique(), default=df["ano"].unique())
bairros = st.sidebar.multiselect("Selecione os bairros", options=df["bairro"].unique(), default=df["bairro"].unique())
estados = st.sidebar.multiselect("Selecione os estados", options=["Pernambuco"], default=["Pernambuco"])

# aplicar filtros
df_filtered = df[(df["ano"].isin(anos)) & (df["bairro"].isin(bairros))]

# ============================
# GRÁFICO 1: STACKED BAR HORIZONTAL (ORDENADO)
# ============================
st.subheader("Tipos de Crimes nos Meses com Maior Incidência (Horizontal)")

# Dicionário para converter número do mês para nome em português
meses_pt = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# Cria uma coluna numérica do mês para ordenar
df_filtered["mes_num"] = df_filtered["data_ocorrencia"].dt.month

# Agrupa os dados
crimes_mes = df_filtered.groupby(["mes_num", "tipo_crime"]).size().unstack(fill_value=0)

# Renomeia os índices para os nomes em português
crimes_mes.index = crimes_mes.index.map(meses_pt)

# Ordena corretamente de Janeiro a Dezembro
ordem_meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
               "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
crimes_mes = crimes_mes.reindex(ordem_meses)

# Plot horizontal
fig1, ax1 = plt.subplots(figsize=(12,8))
crimes_mes.plot(kind="barh", stacked=True, ax=ax1, colormap="tab10")

ax1.set_xlabel("Número de Crimes")
ax1.set_ylabel("Mês")
ax1.set_title("Distribuição dos Crimes por Tipo e Mês (Horizontal)")

# Inverter o eixo Y para começar em Janeiro no topo
ax1.invert_yaxis()

# Adicionando rótulos dentro das barras
for p in ax1.patches:
    width = p.get_width()
    if width > 0:
        ax1.annotate(f'{int(width)}',
                     (p.get_x() + width/2, p.get_y() + p.get_height()/2),
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')

st.pyplot(fig1)



# ============================
# GRÁFICO 2: TOP 10 BAIRROS
# ============================
st.subheader("Top 10 Bairros com Maior Número de Crimes")

top_bairros = df_filtered["bairro"].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(12,6))
bars = ax2.bar(top_bairros.index, top_bairros.values, color=plt.cm.viridis(range(10)))
ax2.set_title("Top 10 Bairros com Maior Número de Crimes")
ax2.set_xlabel("Bairro")
ax2.set_ylabel("Número de Crimes")

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.xticks(rotation=45)
st.pyplot(fig2)

# ============================
# GRÁFICO 3: DISTRIBUIÇÃO TIPOS DE CRIME
# ============================
st.subheader("Distribuição dos Tipos de Crimes durante o Horário de Pico")

top_crimes = df_filtered["tipo_crime"].value_counts().head(10)

fig3, ax3 = plt.subplots(figsize=(12,6))
bars2 = ax3.bar(top_crimes.index, top_crimes.values, color=plt.cm.magma(range(10)))
ax3.set_title("Distribuição dos Tipos de Crimes durante o Horário de Pico")
ax3.set_xlabel("Tipo de Crime")
ax3.set_ylabel("Número de Ocorrências")

for bar in bars2:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.xticks(rotation=45)
st.pyplot(fig3)

# ============================
# GRÁFICO 4: MAPA DE CRIMES
# ============================
st.subheader("Onde os crimes se concentram?")

bairro_escolhido = st.selectbox("Selecione um bairro para visualizar no mapa", top_bairros.index)

df_mapa = df_filtered[df_filtered["bairro"] == bairro_escolhido]

if "latitude" in df_mapa.columns and "longitude" in df_mapa.columns:
    mapa = folium.Map(location=[df_mapa["latitude"].mean(), df_mapa["longitude"].mean()], zoom_start=13)

    for _, row in df_mapa.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=f"{row['tipo_crime']} - {row['data_ocorrencia']}"
        ).add_to(mapa)

    st_folium(mapa, width=1000, height=500)
else:
    st.warning("⚠️ O dataset não contém colunas de latitude/longitude para plotar no mapa.")


# ============================
# NOVOS GRÁFICOS (SEM ALTERAR OS ANTERIORES)
# ============================

# 1) Crimes por tipo
st.subheader("Distribuição de Crimes por Tipo")
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.countplot(data=df, x="tipo_crime", order=df["tipo_crime"].value_counts().index, ax=ax1)
plt.xticks(rotation=45)
# Adicionando rótulos
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='bottom')
st.pyplot(fig1)

# 2) Crimes por bairro
st.subheader("Top 10 Bairros com Mais Ocorrências")
bairros_top = df["bairro"].value_counts().head(10)
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(x=bairros_top.values, y=bairros_top.index, ax=ax2)
# Adicionando rótulos
for i, v in enumerate(bairros_top.values):
    ax2.text(v + 0.5, i, str(v), va='center')
st.pyplot(fig2)

# 3) Evolução temporal
st.subheader("Ocorrências ao Longo do Tempo")
ocorrencias_por_mes = df.groupby(df["data_ocorrencia"].dt.to_period("M")).size()
fig3, ax3 = plt.subplots(figsize=(10,5))
ocorrencias_por_mes.plot(kind="line", ax=ax3, marker='o')
# Adicionando rótulos em cada ponto
for x, y in zip(ocorrencias_por_mes.index.astype(str), ocorrencias_por_mes.values):
    ax3.text(x, y, str(y), ha='center', va='bottom')
st.pyplot(fig3)

# 4) Armas utilizadas
if "arma_utilizada" in df.columns:
    st.subheader("Uso de Armas nas Ocorrências")
    fig4, ax4 = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x="arma_utilizada", order=df["arma_utilizada"].value_counts().index, ax=ax4)
    plt.xticks(rotation=45)
    # Adicionando rótulos
    for p in ax4.patches:
        ax4.annotate(f'{p.get_height()}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom')
    st.pyplot(fig4)

# ============================
# RECOMENDAÇÃO DE RONDAS PARA A PATRULHA (K-MEANS com 2 CLUSTERS)
# ============================
st.subheader("Recomendações de Rondas Policiais")

if "latitude" in df_filtered.columns and "longitude" in df_filtered.columns:
    
    # Filtro simples: escolha de bairro
    bairro_escolhido_cluster = st.selectbox(
        "Selecione o bairro para receber recomendações de rondas:",
        options=sorted(df_filtered["bairro"].dropna().unique())
    )

    # Filtra apenas o bairro escolhido
    df_cluster = df_filtered[df_filtered["bairro"] == bairro_escolhido_cluster]

    # Seleciona coordenadas
    coords = df_cluster[["latitude", "longitude"]].dropna()

    if not coords.empty and len(coords) >= 2:
        # Define 2 clusters fixos (vermelho e azul)
        n_clusters = min(2, len(coords))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        coords["cluster"] = kmeans.fit_predict(coords)

        # Cria mapa centralizado
        mapa_cluster = folium.Map(
            location=[coords["latitude"].mean(), coords["longitude"].mean()],
            zoom_start=14
        )

        # Paleta com apenas duas cores
        colors = ["red", "blue"]

        # Adiciona ocorrências no mapa
        for _, row in coords.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color=colors[int(row["cluster"])],
                fill=True,
                fill_opacity=0.5
            ).add_to(mapa_cluster)

        # Adiciona centroides como pontos de ronda
        recomendacoes = []
        for i, center in enumerate(kmeans.cluster_centers_):
            folium.Marker(
                location=[center[0], center[1]],
                popup=f"Ponto recomendado para rondas (Cluster {i+1})",
                icon=folium.Icon(color="blue" if i == 1 else "red", icon="shield")
            ).add_to(mapa_cluster)

            recomendacoes.append(
                f"**Ponto {i+1}:** Realizar rondas nas proximidades de "
                f"**latitude {round(center[0], 4)} / longitude {round(center[1], 4)}**."
            )

        # Exibe mapa no painel
        st_folium(mapa_cluster, width=1000, height=500)

        # Exibe roteiro de rondas
        st.markdown("### Roteiro de Rondas Recomendado")
        for r in recomendacoes:
            st.write(r)

        st.success("Essas recomendações foram geradas com base nas áreas de maior concentração de ocorrências (2 regiões principais).")
    else:
        st.warning("Não há coordenadas suficientes neste bairro para gerar recomendações.")
else:
    st.warning("O dataset não contém latitude/longitude suficientes para aplicar o modelo de clusters.")


    
# ============================
# PREVISÃO DE QUANTIDADE DE CRIMES
# ============================
st.markdown("---")
st.header("Previsão de Quantidade de Crimes (Próximos Dias)")

# Carregar modelo salvo
model2 = joblib.load("modelo2_previsao_crimes.pkl")

# Selecionar número de dias à frente
dias_prev = st.slider("Quantos dias à frente deseja prever?", 1, 7, 3)

# Calcular data base
data_hoje = datetime.date.today()
datas_futuras = [data_hoje + datetime.timedelta(days=i) for i in range(1, dias_prev + 1)]

# Gerar previsões simples baseadas nas features de tempo
preds = []
for d in datas_futuras:
    features = pd.DataFrame([{
        "ano": d.year,
        "mes": d.month,
        "dia_semana": d.weekday(),
        "is_fim_semana": 1 if d.weekday() >= 5 else 0,
        "lag_1": 0,  # valores neutros porque estamos prevendo independente de histórico
        "lag_7": 0,
        "roll_mean_7": 0,
        "roll_std_7": 0,
    }])
    pred = model2.predict(features)[0]
    preds.append({"Data": d, "Crimes Previstos": round(pred)})

df_prev = pd.DataFrame(preds)

# Mostrar tabela e gráfico
st.dataframe(df_prev)
st.line_chart(df_prev.set_index("Data"))
st.info("Estimativa baseada em padrões históricos e sazonais.")
