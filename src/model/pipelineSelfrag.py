from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationBufferMemory # Import de la mémoire
from langchain_community.vectorstores import Qdrant
import qdrant_client
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
import pymupdf4llm , pymupdf,pytesseract as ocr
from langchain_core.output_parsers import StrOutputParser # parser
from langchain_core.prompts import PromptTemplate


#trouve les PC HP intel core i7
# Charger les variables d'environnement
load_dotenv()

# Récupérer les clés API et chemins nécessaires
HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
FILE_TYPES= ['.png', '.jpeg', '.jpg', '.pdf']

#llama3-8b-8192
# Initialize memory and conversation chain globally
memory = ConversationBufferMemory()
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

def initialize_vectorstore(embedding_function, QDRANT_URL, QDRANT_API_KEY, collection_name):
    qdrantClient = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)
    return Qdrant(qdrantClient, collection_name, embedding_function) #, vector_name='vector_params'


def initialize_retriever(llm, vectorstore,metadata_field_info,document_content_description):
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={'k': 30}
    )
    return retriever

async def query_bot(retriever, question,prompt):
    context = retriever.invoke(question)
    document_chain = create_stuff_documents_chain(llm, prompt)
              
    # Charger l'historique des conversations
    conversation_history = memory.load_memory_variables({})
    result = document_chain.invoke(
            {
                "context": context,
                "historique":conversation_history['history'],
                "question": question  # Utiliser 'question' au lieu de 'messages'
            },
    )
    # Save context
    memory.save_context({"question": question}, {"response": result})
    

    return result

