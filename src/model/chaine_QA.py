from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API Hugging Face
HF_TOKEN = os.getenv('API_TOKEN')

# Description du contenu du document
document_content_description = "Information about the product, including part number, supplier, description, brand, quantity, and price. Do not use the word 'contains' or 'contain' as filters. "

# Métadonnées des attributs
metadata_field_info = [
    AttributeInfo(
        name="marque",
        description="The brand of the product.",
        type="string",
    ),
    AttributeInfo(
        name="categorie",
        description="The category of the product.",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="The source of the product.",
        type="string",
    ),
    AttributeInfo(
        name="row",
        description="The row of the product.",
        type="integer",
    ),
]

GROQ_TOKEN='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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key= GROQ_TOKEN,temperature=0)

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
)
# This example only specifies a relevant query
retrieved_docs = retriever.invoke("donne moi un imprimante marque hp ")
print(retrieved_docs)
print(len(retrieved_docs))

'''
# Construire le template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des infos si elles ne sont pas dans le contexte.
            Répond seulement si tu as la réponse. Affiche les produits trouvés dans un tableau avec comme colonnes: Référence, Marque, et Description.
            Ne formate pas le contexte.si le context est vide dis moi que tu nas pas trouve
            je veux que la reponse soit clair
            avec des saut de ligne.
            ne me donne pas des produit qui ne sont pas dans le context
            
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
        "context": docs,
        "messages": [
            HumanMessage(content=query)
        ],
    }
)

print(result)

'''