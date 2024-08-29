from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationBufferMemory # Import de la mémoire
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Qdrant
import qdrant_client
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
import pymupdf4llm , pymupdf,pytesseract as ocr
from langchain_core.output_parsers import StrOutputParser # parser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
compressor_cohere =CohereRerank(model="rerank-multilingual-v3.0",cohere_api_key="3Kqve3WTMgv563shw0tfOOpLL2pUItly7DxBQQhv",top_n=8)
# Charger les variables d'environnement
load_dotenv()
# Récupérer les clés API et chemins nécessaires
COHERE_API_KEY=os.getenv('COHERE_API_KEY')
#print('COHERE_API_KEY =',COHERE_API_KEY)
HF_TOKEN = os.getenv('API_TOKEN')
openAi8key=os.getenv('openAi8key')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'
#llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
modelName = "gpt-4o-mini"
llm = ChatOpenAI(model_name=modelName, api_key=openAi8key, temperature=0.5)
FILE_TYPES= ['.png', '.jpeg', '.jpg', '.pdf']
modelName2='gemma2-9b-it'
#llama3-8b-8192
# Initialize memory and conversation chain globally
#memory = ConversationBufferMemory()
memory = ConversationBufferWindowMemory( k=0)
memory.clear()
print('memory before',memory.load_memory_variables)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openAi8key)
def initialize_vectorstore(embeddings, QDRANT_URL, QDRANT_API_KEY, collection_name):
    qdrantClient = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)
    return Qdrant(qdrantClient, collection_name, embeddings) #, vector_name='vector_params'


def initialize_retriever(llm, vectorstore,metadata_field_info,document_content_description):
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=0),
        search_kwargs={'k': 200, 'fetch_k': 10},
        search_type='mmr'
    )
    return retriever



def query_bot(retriever,question,prompt): 
    re_rank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
    compressed_docs = re_rank_retriever.invoke(question)
    print('compressed_docs =',compressed_docs)
    context =compressed_docs
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."
    print(context)
    print('length de context : ', sum(len(sublist) for sublist in context))
    flattened_docs = [item for sublist in context for item in sublist]
    print('la flattened_docs : ', flattened_docs)
    #print('la liste : ', liste)

    document_chain = create_stuff_documents_chain(llm, prompt)
              
    # Charger l'historique des conversations
    conversation_history = memory.load_memory_variables({})
    result = document_chain.invoke(
            {
                "context": flattened_docs,
                "historique":conversation_history['history'],
                "question": question  # Utiliser 'question' au lieu de 'messages'
            },
    )
    # Save context
    memory.save_context({"question": question}, {"response": result})
    

    return result
def docs_to_string(documents):
    all_results = []
    
    for doc in documents:
        # Récupérer toutes les métadonnées
        metadata = doc.metadata
        
        # Construire une chaîne avec toutes les métadonnées
        metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
        
        # Récupérer le contenu de la page
        page_content = doc.page_content
        
        # Construire le texte final pour chaque document
        result = f"document(metadata:{metadata_str} page_content : {page_content}),"
        
        # Ajouter le texte à la liste des résultats
        all_results.append(result)
    
    # Joindre tous les résultats avec deux sauts de ligne entre eux
    doc_txt = "\n\n".join(all_results)
    return doc_txt
#Unique union of retrieved docs
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]
class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content
async def batch_query_bot(retriever,questions: list[str] | str,prompt):
    encoding = tiktoken.get_encoding("cl100k_base")
    question_tokens = sum(len(tokenizer.encode(q)) for q in questions)
    print('question_tokens =',question_tokens)
    #prompt_tokens=sum(len(tokenizer.encode(prompt)))
    #print('prompt_tokens',prompt_tokens)
    #print(input = question_tokens+prompt_tokens)
    re_rank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_cohere, base_retriever=retriever
)
    context=[[]]
    flattened_docs=[]
    if not isinstance(questions, list):
            questions= [questions]
    context =re_rank_retriever.batch(questions)
    print(context)
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."
    #print(context)
    print('length de context : ', sum(len(sublist) for sublist in context))
    flattened_docs = [item for sublist in context for item in sublist]
    #print('la flattened_docs : ', flattened_docs)
    print('la liste : ', len(flattened_docs))


    document_chain = create_stuff_documents_chain(llm, prompt)
              
    # Charger l'historique des conversations
    conversation_history = memory.load_memory_variables({})
    result = document_chain.invoke(
            {
                "context": flattened_docs,
                "historique":conversation_history['history'],
                "question": questions  # Utiliser 'question' au lieu de 'messages'
            }
    )
    result_tokens = len(tokenizer.encode(result))
    print('result_tokens =',result_tokens)
    # Save context
    memory.save_context({"question": questions}, {"response": result})
    #print('conversation_history',memory.load_memory_variables({}))
    return result
