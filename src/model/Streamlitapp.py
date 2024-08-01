import streamlit as st
from pipline import load_embedding_function, initialize_vectorstore, initialize_retriever, query_bot ,extract_product_info
from langchain_groq import ChatGroq
import pandas as pd
# Charger la fonction d'embedding
embedding_function = load_embedding_function()

# Initialiser le modèle LLM
GROQ_TOKEN = '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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)

# Initialiser le vectorstore et le retriever
vectorstore = initialize_vectorstore(embedding_function)
retriever = initialize_retriever(llm, vectorstore)

# Interface Streamlit
st.title("Sales Smart Assistant DGF")
st.write("Posez une question sur les produits:")
question = st.text_input("Question", value="trouve les Ordinateurs intel core i5 de la marque Samsung")

if st.button("Rechercher"):
    result = query_bot(retriever, embedding_function, question)
    #df = pd.DataFrame(product_list)
    #st.table(df)
    product_list=extract_product_info(result)
    st.write(product_list)


    
