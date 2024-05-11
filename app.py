import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def main():
    st.title('Netflix DataFrame')
    dados = pd.read_csv('../cp/netflix_titles_atualizado.csv')
    linhas = 1
    st.table(dados.head(linhas))
    
    st.title('Distribuição da coluna "release_year" padronizada')
    criacaoVariavel = pd.read_csv('../cp/netflix_titles.csv')
    criacaoVariavel['date_added'] = criacaoVariavel['date_added'].str.strip()
    criacaoVariavel['date_added'] = pd.to_datetime(criacaoVariavel['date_added'], errors='coerce')
    scaler = StandardScaler()
    variavelPadronizada = criacaoVariavel.copy()
    variavelPadronizada['release_year_standardized'] = scaler.fit_transform(criacaoVariavel[['release_year']])
    media = variavelPadronizada['release_year_standardized'].mean()
    desvioPadrao = variavelPadronizada['release_year_standardized'].std()
    st.write("Média da coluna padronizada:", media)
    st.write("Desvio padrão da coluna padronizada:", desvioPadrao)
    plt.figure(figsize=(10, 6))
    plt.hist(variavelPadronizada['release_year_standardized'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribuição da coluna "release_year" padronizada')
    plt.xlabel('Valor Padronizado')
    plt.ylabel('Frequência')
    plt.grid(True)
    st.pyplot(plt)
    
    st.title('Número de Registros por Ano de Adição')
    criacaoVariavel = pd.read_csv('../cp/netflix_titles.csv')
    criacaoVariavel['date_added'] = criacaoVariavel['date_added'].str.strip()
    criacaoVariavel['date_added'] = pd.to_datetime(criacaoVariavel['date_added'], errors='coerce')
    data_net = criacaoVariavel['date_added'].dt.year.value_counts().sort_index()
    plt.figure(figsize=(15, 6))
    data_net.plot(kind='bar', color='skyblue')
    plt.title('Número de Registros por Ano de Adição')
    plt.xlabel('Ano de Adição')
    plt.ylabel('Número de Registros')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

    st.title('Análise do Modelo de Classificação')

    dataFrame = pd.read_csv('../cp/netflix_titles.csv')
    dataFrame['duration'] = dataFrame['duration'].str.extract('(\d+)').astype(float)

    X = dataFrame[['duration', 'release_year']] 
    y = dataFrame['type']  
    xTreino, xTeste, yTreino, yTeste = train_test_split(X, y, test_size=0.2, random_state=42)
    modeloTreino = DecisionTreeClassifier()
    modeloTreino.fit(xTreino, yTreino)
    prev = modeloTreino.predict(xTeste)
    calculo = confusion_matrix(yTeste, prev)
    st.subheader('Matriz de Confusão')
    fig, ax = plt.subplots()
    sns.heatmap(calculo, annot=True, cmap='Blues', fmt='g', xticklabels=modeloTreino.classes_, yticklabels=modeloTreino.classes_, ax=ax)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Existente')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)
    st.subheader('Relatório de Classificação')
    st.text(classification_report(yTeste, prev))

if __name__ == '__main__':
    main()
    