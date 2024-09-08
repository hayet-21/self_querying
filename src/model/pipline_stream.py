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

compressor_cohere =CohereRerank(model="rerank-multilingual-v3.0",cohere_api_key="weyxiNeVkznwJueXMTuj9Mu7JtwTDf5nMfgPA8XF",top_n=10)
# Charger les variables d'environnement
load_dotenv()
# Récupérer les clés API et chemins nécessaires
HF_TOKEN = os.getenv('API_TOKEN')
openAi8key=os.getenv('openAi8key')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'
#llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
modelName = "gpt-4o-mini"
llm = ChatOpenAI(model_name=modelName, api_key=openAi8key, temperature=0.3)
FILE_TYPES= ['.png', '.jpeg', '.jpg', '.pdf']
modelName2='gemma2-9b-it'
#llama3-8b-8192
# Initialize memory and conversation chain globally
#memory = ConversationBufferMemory()
memory = ConversationBufferWindowMemory( k=0)
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
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold':0.4, 'k':25}
    )
    return retriever
async def query_bot(retriever,question,prompt):
    context = retriever.invoke(question)
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."

    #query_embedding = embedding_function.embed_query(question)
    #doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in context]
    #similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

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

#Unique union of retrieved docs
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def batch_query_bot(retriever, questions: list[str] | str, prompt):
    re_rank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor_cohere, base_retriever=retriever
    )
    context = [[]]
    flattened_docs = []
    if not isinstance(questions, list):
        questions = [questions]
    print('questions =', questions)
    
    if not questions or all(q.strip() == '' for q in questions):
        yield "Please provide at least one valid question."
        return
    
    context = re_rank_retriever.batch(questions)
    
    if not context:
        yield "Je n'ai pas trouvé de produits correspondants."
        return
    
    liste = get_unique_union(context)
    print(f"Nombre total de liste : ", {len(liste)})

    document_chain = create_stuff_documents_chain(llm, prompt)
    conversation_history = memory.load_memory_variables({})
    
    # Simulate streaming by iterating over results
    for result in document_chain.stream({
        "context": liste,
        "historique": conversation_history['history'],
        "question": questions
    }):
        yield result

    memory.save_context({"question": questions},{"response":result})