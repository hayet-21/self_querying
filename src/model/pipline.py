from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Charger les variables d'environnement
load_dotenv()

# Récupérer les clés API et chemins nécessaires
HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'

def load_embedding_function():
    try:
        embedding_function = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="intfloat/multilingual-e5-large"
        )
        return embedding_function
    except Exception as e:
        print(f"Error loading embedding function: {e}")
        return None

def initialize_vectorstore(embedding_function):
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function, 
        collection_name=COLLECTION_CSV
    )
    return vectorstore

def initialize_retriever(llm, vectorstore):
    document_content_description = "Informations sur le produit, incluant la reference et la description."
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

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={'k': 50}
    )
    return retriever

def query_bot(retriever, embedding_function, question):
    context = retriever.invoke(question)
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."

    query_embedding = embedding_function.embed_query(question)
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in context]
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    filtered_docs = [
        doc for doc, similarity in zip(context, similarities) if similarity >= 0.7
    ]

    data = [
        {
            'Référence': doc.metadata.get('Ref_produit', 'N/A'),
            'Categorie': doc.metadata.get('Categorie', 'N/A'),
            'Marque': doc.metadata.get('Marque', 'N/A'),
            'Description': doc.page_content
        }
        for doc in filtered_docs
    ]
    
    
    return data
def extract_product_info(text):
    products = []
    lines = text.strip().split('\n\n')
    for line in lines:
        product = {}
        lines = line.split('\n')
        for l in lines:
            if l.startswith('- Référence: '):
                product['Référence'] = l.replace('- Référence: ', '').strip()
            elif l.startswith('- Categorie: '):
                product['Categorie'] = l.replace('- Categorie: ', '').strip()
            elif l.startswith('- Marque: '):
                product['Marque'] = l.replace('- Marque: ', '').strip()
            elif l.startswith('- Description: '):
                product['Description'] = l.replace('- Description: ', '').strip()
        products.append(product)
    return products