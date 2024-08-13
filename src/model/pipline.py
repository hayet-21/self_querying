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
import pymupdf4llm

#trouve les PC HP intel core i7
# Charger les variables d'environnement
load_dotenv()

# Récupérer les clés API et chemins nécessaires
HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
GROQ_TOKEN = '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'
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
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
        search_kwargs={'k': 50}
    )
    return retriever

async def query_bot(retriever, embedding_function, question,prompt):
    context = retriever.invoke(question)
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."

    query_embedding = embedding_function.embed_query(question)
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in context]
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    filtered_docs = [
        doc for doc, similarity in zip(context, similarities) if similarity >= 0.7
    ]
    document_chain = create_stuff_documents_chain(llm, prompt)
              
    # Charger l'historique des conversations
    conversation_history = memory.load_memory_variables({})
    result = document_chain.invoke(
            {
                "context": filtered_docs,
                "historique":conversation_history['history'],
                "question": question  # Utiliser 'question' au lieu de 'messages'
            },
    )
    # Save context
    memory.save_context({"question": question}, {"response": result})
    

    return result

# Load the PDF
def load_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Load the PDF
def load_doc(pdf_path: str):
    loader = Docx2txtLoader(pdf_path)
    documents = loader.load()
    return documents

def pdf_reader(file_path,llm):
    # Load the document based on its extension
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        documents_text = pymupdf4llm.to_markdown(file_path)
    elif file_extension == '.docx' :
        doc=load_doc(file_path)
        documents_text= ' '.join([doc.page_content for doc in doc])
    else:
        return "Unsupported file format"

    # Create the prompt template
    prompt_template = PromptTemplate.from_template(
       " Tu es Un assistant AI super helpful. Etant donnee un contexte, ton travail est simple. il consiste a: \
        1- Extraire toutes les decriptions et les descriptions des produits qui se trouvent à l'interieur du contexte. \
        2- Reformuler, si besoin, les descriptions en etant le plus fidele possible à la description originale. \
        3- NE JAMAIS GENERER de reponse de ta part si le contexte est vide ou y a pas assez d'info. \
        4- Mettre chaque description sur une ligne. \
        5-repond avec la meme langue du text dans le context\
        6- Ne donne pas de commentaire de ta part.\
        context : {documents_text} \
        ------------\
        response : "
    )

    # Generate the response based on the products listed in the PDF
    query = "Quels sont les produits listés dans ce document ?"
    chain = prompt_template | llm
    response = chain.invoke({"documents_text": documents_text, "query": query})

    return response.content
'''
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
'''