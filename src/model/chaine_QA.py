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
from langchain_community.document_loaders.csv_loader import CSVLoader

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API Hugging Face
HF_TOKEN = os.getenv('API_TOKEN')
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")
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
'''
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
loader = CSVLoader(file_path=DATA_PATH_CSV,
    metadata_columns=["Marque","Categorie"],   # Optional: Include 'Marque' as metadata
    csv_args={"delimiter": ","},  # Specify delimiter if needed
    encoding="utf-8"
)
documents = loader.load()
chroma_instance = Chroma(embedding_function= embedding_function, persist_directory=CHROMA_PATH, collection_name=COLLECTION_CSV)
# chroma_instance.delete(chroma_instance.get()['ids'])
chroma_instance.add_documents(documents)
chroma_instance.persist()
print("There are", chroma_instance._collection.count(), "documents in the collection")
'''
# Description du contenu du document
document_content_description = "Information about the product, including part number, supplier, description, brand, quantity, and price. Do not use the word 'contains' or 'contain' as filters. "

# Métadonnées des attributs
metadata_field_info = [
    AttributeInfo(
        name="Categorie",
        description="The brand of the product.",
        type="string",
    ),
    AttributeInfo(
        name="Categorie",
        description="The category of the product.",
        type="string",
    ),
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
    enable_limit=True
   
)
question="find 2 TV samsung "


context =retriever.get_relevant_documents(question)

# Construire le template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
            Répond seulement si tu as la réponse. Affiche les produits un par un dans le format suivant:

            - Produit 1:
            - Référence: 
            - Categorie: 
            - Marque: 
            - Description: 
          
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
        "context": context,
        "question": question,  # Use 'question' instead of 'messages'
    }
)

print(result)

