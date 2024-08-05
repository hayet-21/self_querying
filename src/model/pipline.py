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
from langchain_community.vectorstores import Qdrant
import qdrant_client

# Charger les variables d'environnement
load_dotenv()

# Récupérer les clés API et chemins nécessaires
HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
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

def initialize_vectorstore(embedding_function,QDRANT_URL,QDRANT_API_KEY,collection_name):
    qdrantClient = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)
    return Qdrant(qdrantClient, collection_name, embedding_function, vector_name='vector_params')


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

    # Construire le template de prompt
    prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """
                    Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                    Répond seulement si tu as la réponse. Affiche les produits un par un sous forme de tableau qui contient ces colonne Référence,Categorie, Marque, Description.
                    il faut savoir que laptop , ordinateur et pc et poste de travail ont tous le meme sens
                    il faut savoir que telephone portable et smartphone ont le meme sens
                    il faut savoir que tout autre caracterisque du produit tel que la ram stockage font partie de la description du produit et il faut filtrer selon la marque et le categorie seulement.
                
                    Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.
                    Contexte:
                    {context}

                    Question: {question}

                    Réponse :
                    """
                ),
            ]
        )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    result = document_chain.invoke(
            {
                "context": filtered_docs,
                "question": question # Use 'question' instead of 'messages'
            },
    )
    return result
def extract_product_info(text):
    if not text or text.strip() == "Je n'ai pas trouvé de produits correspondants.":
        return []
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
