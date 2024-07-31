from dotenv import load_dotenv
import os
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API Hugging Face
HF_TOKEN = os.getenv('API_TOKEN')
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")
@st.cache_resource
def load_embedding_function():
    try:
        embedding_function = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="intfloat/multilingual-e5-large"
        )
        print("Embedding function loaded successfully.")
        return embedding_function
    except Exception as e:
        print("Error loading embedding function:", e)
        return None

embedding_function = load_embedding_function()



# Description du contenu du document
document_content_description = "Informations sur le produit, incluant la reference et la description. Do not use the word 'contains' or 'contain' as filters. "

# Métadonnées des attributs
metadata_field_info = [
    AttributeInfo(
        name="Marque",
        description="La Marque du produit.",
        type="string",
        
    ),
    AttributeInfo(
        name="Categorie",
        description="La Categorie du produit.",
        type="string",
    )
]

GROQ_TOKEN='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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key= GROQ_TOKEN,temperature=0)

# Initialiser Chroma pour la récupération
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')

# Charger la collection Chroma
vectorstore = Chroma(persist_directory=CHROMA_PATH, 
                     embedding_function=embedding_function, 
                     collection_name=COLLECTION_CSV)

# Initialiser le SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    search_kwargs={'k': 50}
)
# Interface Streamlit
st.title("Assistant Vendeur Intelligent")
st.write("Posez une question sur les produits:")
question = st.text_input("Question", value="trouve les Ordinateurs intel core i5 de la marque Samsung")
#question = "trouve les Ordinateurs intel core i5 de la marque Samsung"
if st.button("Rechercher"):
    # Récupérer le contexte associé à la question
    context = retriever.invoke(question)

    # Vérification si le contexte est vide
    if not context:
        print("Je n'ai pas trouvé de produits correspondants.")
    else:
        # Embed the question using the Hugging Face embedding function
        query_embedding = embedding_function.embed_query(question)

        # Embed the documents in the context using the Hugging Face embedding function
        doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in context]

        # Calculate the cosine similarity between the query embedding and each document embedding
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Get the indices of the top N most similar documents
        filtered_docs = [
            doc for doc, similarity in zip(context, similarities) if similarity >= 0.7
        ]

        

        # Pass the top N most similar documents to the template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """
                    Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                    Répond seulement si tu as la réponse. Affiche les produits un par un dans le format suivant:

                    Produit 1:
                    - Référence: 
                    - Categorie: 
                    - Marque: 
                    - Description: 
                    il faut savoir que laptop et pc et poste de travail ont le meme sens
                    Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.

                    Contexte:
                    {context}

                    Question: {question}

                    Réponse : je vous propose ...
                    """
                ),
            ]
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        result = document_chain.invoke(
            {
                "context": filtered_docs,
                "question": question, 
            }
        )

        st.write(result)


