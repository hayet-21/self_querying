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

import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

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
        result = f"{metadata_str}\n\nContenu de la page:\n{page_content}"
        
        # Ajouter le texte à la liste des résultats
        all_results.append(result)
    
    # Joindre tous les résultats avec deux sauts de ligne entre eux
    doc_txt = "\n\n".join(all_results)
    
    return doc_txt
async def query_bot(retriever,question,prompt):
    context= retriever.invoke(question)
    r=docs_to_string(context)
    print('nombre de token du context : ',num_tokens_from_string(r))
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

pdf_prompt= PromptTemplate.from_template(pdf_prompt_instruct)
pdf_chain = pdf_prompt | llm | StrOutputParser()
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