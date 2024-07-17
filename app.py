import streamlit as st
from search import SearchEngine
from pymilvus import MilvusClient


milvus_client = MilvusClient(host='localhost',port='19530')
milvus_collection_name = 'nvidia_dataset_6'

searchEngine = SearchEngine(milvus_client, milvus_collection_name)

st.title('StepAI Code Submission')

question = st.text_input("Enter your question:")

if st.button('Search'):
    if question:
        answer = searchEngine.search(user_query=question)
        st.write("Answer:", answer,width=1000)
    else:
        st.write("Please enter a question.")