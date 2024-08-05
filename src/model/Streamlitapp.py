import streamlit as st
from pipline import load_embedding_function, initialize_vectorstore, initialize_retriever, query_bot ,extract_product_info
from langchain_groq import ChatGroq

# Charger la fonction d'embedding
embedding_function = load_embedding_function()

# Initialiser le modèle LLM
GROQ_TOKEN = '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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
url="https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key= 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'
collection_name= "icecat_collection"
# Initialiser le vectorstore et le retriever
vectorstore = initialize_vectorstore(embedding_function,url,api_key,collection_name)
retriever = initialize_retriever(llm, vectorstore)

# Interface Streamlit
st.set_page_config(
    page_title="EquotIA",
    page_icon="🧠",
)
st.title(" 🧠 Sales Smart Assistant DGF")
question = st.chat_input("ex : trouve les Ordinateurs intel core i5 de la marque Samsung")

if question:
    result = query_bot(retriever, embedding_function, question)
    
    st.markdown(result)  


    
