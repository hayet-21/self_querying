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
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Charger les variables d'environnement
load_dotenv()
openAi8key = "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"
modelName = "gpt-4o-mini"
# Récupérer les clés API et chemins nécessaires
HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'
#llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
llm = ChatOpenAI(model_name=modelName, api_key=openAi8key, temperature=0)
FILE_TYPES= ['.png', '.jpeg', '.jpg', '.pdf']
modelName2='gemma2-9b-it'

#llama3-8b-8192
# Initialize memory and conversation chain globally
memory = ConversationBufferMemory()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key='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')
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
        search_kwargs={'k': 30}
    )
    return retriever

async def query_bot(retriever,question,prompt):
    context = retriever.invoke(question)
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."

    #query_embedding = embedding_function.embed_query(question)
    #doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in context]
    #similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    #filtered_docs = [
    #    doc for doc, similarity in zip(context, similarities) if similarity >= 0.7
    #]
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


# Define the prompt template and chain
pdf_prompt_instruct= """Tu es Un assistant AI super helpful. Étant donné un contexte qui est le texte brut issu d'une image ou d'un PDF, ton travail est simple :
1- Extraire toutes les descriptions des produits qui se trouvent à l'intérieur du contexte.
2- Reformuler, si besoin, les descriptions en étant le plus fidèle possible à la description originale.
3- NE JAMAIS GÉNÉRER de réponse de ta part si le contexte est vide ou s'il n'y a pas assez d'informations.
4- Pour chaque produit identifie et structure les informations suivantes si et seulement si elles existent :
        - **Part Number** : [la référence du produit]
        - **Marque** : [la marque du produit]
        - **Catégorie** : [la catégorie du produit]
        - **Description** : [la description du produit]
5- NE DONNE AUCUN COMMENTAIRE NI INTRODUCTION JUST DROP THE TEXT.

{contexte}
------------
Réponse: trouve les produits qui correspondent à ces descriptions :
"""

def llm_generation(modelName, GROQ_TOKEN):         
    llm = ChatGroq(model_name=modelName, api_key=GROQ_TOKEN, temperature=0)
    return llm
llm2= llm_generation(modelName2,GROQ_TOKEN)
pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)
pdf_chain=  pdf_prompt | llm2
def extract_text_from_img(img):
        text= ocr.image_to_string(img)
        return text

def extract_text_from_pdf(pdf):
        file= pymupdf.open(pdf)
        p= file[0].get_text()
        text= ''
        if bool(p.strip()):
                text= pymupdf4llm.to_markdown(pdf)
                file.close()
        else:
                for page in file:
                        tp= file.get_textpage_ocr()
                        text += f" {tp.extractTEXT()} \n\n" 
        return text
                        

def extract_text_from_file(filepath):
        _, ext= os.path.splitext(filepath)
        assert ext in FILE_TYPES, f'wrong file type. the file must be on of following {FILE_TYPES}'
        if ext == '.pdf':
                return extract_text_from_pdf(filepath)
        else:
                return extract_text_from_img(filepath)
        
def parse_file(filepath, parser= pdf_chain):
        text= extract_text_from_file(filepath)
        return parser.invoke(text)