import streamlit as st
import pandas as pd

def main():
    st.title('Netflix DataFrame')
    dados = pd.read_csv('../cp/netflix_titles_atualizado.csv')
    linhas = 4
    st.table(dados.head(linhas))





if __name__ == '__main__':
    main()