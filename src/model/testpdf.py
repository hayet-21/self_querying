import streamlit as st
from pipline import load_embedding_function, initialize_vectorstore, initialize_retriever, query_bot ,parse_file
from langchain_groq import ChatGroq
import asyncio
import time ,os
from langchain_core.prompts import ChatPromptTemplate


# Charger la fonction d'embedding
embedding_function = load_embedding_function()

# Initialiser le modèle LLM
GROQ_TOKEN = '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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
url="https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
FILE_TYPES= ['.png', '.jpeg', '.jpg', '.pdf']
api_key= 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'
collection_name="lenovoHP_collection"

document_content_description = "Informations sur le produit, incluant la référence et la description."
metadata_field_info = [
        {
            'name': "Marque",
            'description': "La Marque du produit.",
            'type': "string",
        },
        {
            'name': "Categorie",
            'description': "La Categorie du produit.",
            'type': "string",
        }
    ]
# Initialiser le vectorstore et le retriever
vectorstore = initialize_vectorstore(embedding_function,url,api_key,collection_name)
retriever = initialize_retriever(llm, vectorstore,metadata_field_info,document_content_description)
# Construire le template de prompt
prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                Répond seulement si tu as la réponse. Affiche les produits un par un sous forme de tableau qui contient ces colonne Référence,Categorie, Marque, Description.
                Il faut savoir que laptop, ordinateur, ordinateurs portable , pc et poste de travail ont tous le même sens.
                Il faut savoir que téléphone portable et smartphone ont le même sens.
                Il faut savoir que tout autre caractéristique du produit tel que la RAM stockage font partie de la description du produit et il faut filtrer selon la marque et la catégorie seulement.
                Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.
                lorsque une question de similarite entre des produits est poser, il faut dabord commencer par les produit qui ont des processeur qui se ressemble le plus, puis la memoire ram , puis le stockage, puis les autres caracteristique
                la question peut contenir  plusieurs produits avec differentes descriptions, il faut chercher sur les differents produits demandé .
                si je te pose une question sur les question ou les reponses fournient precedement tu doit me repondre selon l'historique.
                tu ne doit pas oublier l'historique car parfois le user continue a te poser des question sur tes reponses que tas deja fourni aupatavant

                Contexte: {context}
                historique :{historique}
                Question: {question}

                Réponse :  
                """
            ),
        ]
    )

# Interface Streamlit
st.set_page_config(
    page_title="EquotIA",
    page_icon="🧠",
)
st.title("🧠 Sales Smart Assistant DGF")

uploaded_file = st.file_uploader("Upload a PDF file")
# Initialiser la session_state pour stocker l'historique des messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parse the file and use the result as the input question
    pdf_text = parse_file(temp_file_path)
    st.markdown(pdf_text)
    question =pdf_text # Use the output of parse_file as the input question
    # Proceed with the bot response using the extracted question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        start_time = time.time()

        # Get the bot's response
        result = asyncio.run(query_bot(retriever, embedding_function, question, prompt))

        exec_time = time.time() - start_time
        st.session_state.messages.append({"role": "ai", "content": f"{result}\n\n(Temps d'exécution: {exec_time:.2f} secondes)"})

        # Display the conversation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])