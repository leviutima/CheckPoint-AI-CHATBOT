import streamlit as st
import pandas as pd

def main():
    st.title('Netflix DataFrame')
    dados = pd.read_csv('../cp/netflix_titles.csv')


    st.subheader('Visualização dos primeiros registros:')
    st.write(dados.head(5).transpose())
    

    if 's1' in dados['show_id'].values:
        dados = dados[dados['show_id'] != 's1']
        st.write('O ID "s1" foi excluído do conjunto de dados.')
    

    st.subheader('Nomes das colunas:')
    st.write(dados.columns)
    

    dados['nova_coluna'] = 'valor_constante'
    

    dados.to_csv('../cp/netflix_titles_atualizado.csv', index=False)
    st.write('Uma nova variável chamada "nova_coluna" foi criada com valores constantes.')
    
    st.subheader('Visualização dos primeiros registros do DataFrame atualizado:')
    st.write(dados.head(5).transpose())





if __name__ == '__main__':
    main()