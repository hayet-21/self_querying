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
collection_name="lenovoHP_collection"
# Initialiser le vectorstore et le retriever
vectorstore = initialize_vectorstore(embedding_function,url,api_key,collection_name)
retriever = initialize_retriever(llm, vectorstore)

# Interface Streamlit
st.set_page_config(
    page_title="EquotIA",
    page_icon="🧠",
)
st.title("🧠 Sales Smart Assistant DGF")

# Initialiser la session_state pour stocker l'historique des messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Entrée utilisateur
question = st.chat_input("ex : trouve les Ordinateurs intel core i5 de la marque Samsung")

if question:
    # Append the user's input to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Get the bot's response
    result = query_bot(retriever, embedding_function, question)
    
    # Append the bot's response to the chat history
    st.session_state.messages.append({"role": "ai", "content": result})
    
    # Display the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])