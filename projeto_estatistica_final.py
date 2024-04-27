# %%
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# %%
# Carregando base de dados
dados = pd.read_csv('./Base/Grupo 1 - Imobiliario - Preço de Casas.csv', encoding='ISO-8859-1', sep=';')

dados.head()
# %%
# Verificando o tipo dos dados
dados.info()

# %%
# Gráfico histograma da variável Preco
fig = px.histogram(dados, x='Preco', title='Distribuição dos Preços dos Imóveis', labels={'Preco':'Preços'})
fig.show()
# %%
# Médidas de posição da variável Preco
print('Média:', round(dados['Preco'].mean()))
print('Mediana:', round(dados['Preco'].median()))
print('Desvio Padrão:', round(dados['Preco'].std()))

# %%
# Correlação das variáveis numéricas
df = dados[['Preco', 'Area', 'Quartos', 'Banheiros', 'NumAndares', 'VagasGaragem']]
correlacao = df.corr()
correlacao

# %%
# Gráfico de correlação
fig = px.imshow(correlacao, text_auto=True, color_continuous_scale='RdBu_r', title='Mapa de Correlação das Vareáveis Numéricas', width=550, height=550)
fig.show()

# %%
# Relação entre Preço e Area
area_preco = dados.groupby('Area')['Preco'].mean().reset_index()

fig = px.scatter(area_preco, x='Area', y='Preco', title='Relação entre Area do Imóvel e seu Preço', color='Area')
fig.show()

# %%
# Relação entre o preço e a localização na estrada principal
df = dados.groupby('EstradaPrincipal')['Preco'].mean().reset_index()
df['EstradaPrincipal'] = df['EstradaPrincipal'].replace({'Nao':'Fora da Principal', 'Sim':'Na Principal'})

fig = px.bar(df, y='Preco', x='EstradaPrincipal', title='Relação entre o Preço e a Localização na Estrada Principal', labels={'EstradaPrincipal':'Localização','Preco':'Preços'}, color='EstradaPrincipal')
fig.show()

# %%
# Relação do preço com quantidade de banheiros
df = dados.groupby('Banheiros')['Preco'].mean().reset_index()

fig = px.bar(df, x='Banheiros', y='Preco' , title='Influência da Quantidade de Banheiros no Preço', color='Banheiros')
fig.show()

# %%
# Relação do preço com quantidade de quartos
df = dados.groupby('Quartos')['Preco'].mean().reset_index()

fig = px.bar(df, x='Quartos', y='Preco' , title='Influência da Quantidade de Quartos no Preço', color='Quartos')
fig.update_traces(textposition='inside')
fig.show()

# %%
df = dados.groupby('QuartoHospedes')['Preco'].mean().reset_index()

fig = px.bar(df, x='Preco', y='QuartoHospedes', title='Influência de Quarto de Hospedes no Preço', labels={'QuartoHospedes':'Quartos de Hospedes', 'Preco':'Preço'}, color='QuartoHospedes')
fig.show()

# %%
# Influencia do andar no preço
df = dados.groupby('NumAndares')['Preco'].mean().reset_index()

fig = px.bar(df, x='NumAndares', y='Preco' , title='Influência do Andar do Imóvel no Preço', labels={'NumAndares':'Andar'}, color='NumAndares')
fig.show()

# %%
fig = px.box(dados, x='Mobilia', y='Preco', title='Preço do Imovel Influênciado Pela Mobilia', color='Mobilia')
fig.show()

# %%
df = dados.groupby('VagasGaragem')['Preco'].mean().reset_index()

fig = px.bar(df, x='VagasGaragem', y='Preco' , title='Infliuência do Número de Vagas de Garagem no Preço', labels={'VagasGaragem':'Qtd Vagas'}, color='VagasGaragem')
fig.show()

# %%
df = dados.groupby('AquecimentoAgua')['Preco'].mean().reset_index()

df['AquecimentoAgua'] = df['AquecimentoAgua'].replace({'Nao':'Sem Aquecimento', 'Sim':'Com Aquecimento'})

fig = px.bar(df, y='AquecimentoAgua', x='Preco', title='Influência do Aquecimento do Imóvel no Preço', color='AquecimentoAgua', labels={'AquecimentoAgua':'Aquecimento de Água', 'Preco':'Preço'})
fig.show()

# %%
# Influência do quarto de hóspedes no preço
df = dados.copy()

df['QuartoHospedes'] = df['QuartoHospedes'].map({'Sim':1, 'Nao':0})

quarto_hospede = sm.add_constant(df['QuartoHospedes'])
modelo = sm.OLS(df['Preco'], quarto_hospede)
resultado = modelo.fit()

# Teste de hipótese
p_valor = resultado.pvalues['QuartoHospedes']
ic = resultado.conf_int().loc['QuartoHospedes']

# Imprimindo os resultados
if p_valor <= 0.05:
    print(f'''
    Hipótese nula rejeitada, o valor-p {p_valor:.10f}
    é extremamente pequeno. O intervalo de confiança é
    aproximadamente {ic[0]:.2f} | {ic[1]:.2f} em 95% das vezes.
    ''')
else:
    print('A hipótese é nula, quartos de hospedes não influenciam no preço.')

# %%
# Influência da quantidade quarto no preço
df = dados.copy()

quarto_hospede = sm.add_constant(df['Quartos'])
modelo = sm.OLS(df['Preco'], quarto_hospede)
resultado = modelo.fit()

# Teste de hipótese
p_valor = resultado.pvalues['Quartos']
ic = resultado.conf_int().loc['Quartos']

# Imprimindo os resultados
if p_valor <= 0.05:
    print(f'''
    Hipótese nula rejeitada, o valor-p {p_valor:.10f}
    é extremamente pequeno. O intervalo de confiança é
    aproximadamente {ic[0]:.2f} | {ic[1]:.2f} em 95% das vezes.
    ''')
else:
    print('A hipótese é nula, quantidade de quartos não influenciam no preço.')