import streamlit as st
from langchain_groq import ChatGroq
import asyncio
import time
import os
import uuid
import pymupdf
import pytesseract
import pymupdf4llm
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Qdrant
import qdrant_client
from langchain_core.prompts import PromptTemplate

# Configuration de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Dell\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Variables et constantes
HF_TOKEN = os.getenv('API_TOKEN')
url = "https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key = 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'
collection_name = "lenovoHP_collection"

FILE_TYPES = ['png', 'jpeg', 'jpg', 'pdf', 'docx', 'xlsx', 'PNG']
modelName = "llama-3.1-70b-versatile"
modelName2 = 'llama3-70b-8192'
# Initialiser le modèle LLM
GROQ_TOKEN = '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'

@st.cache_resource
def llm_generation(modelName, GROQ_TOKEN):         
    llm = ChatGroq(model_name=modelName, api_key=GROQ_TOKEN, temperature=0)
    return llm

llm = llm_generation(modelName, GROQ_TOKEN)
llm2 = llm_generation(modelName2, GROQ_TOKEN)

# Prompt pour extraction de texte depuis PDF
pdf_prompt_instruct = """ Tu es Un assistant AI super helpful. Etant donnee un contexte, ton travail est simple. il consiste a: 
    1- Extraire touts les produit et leurs description des produits qui se trouvent à l'interieur du contexte. 
    2- Reformuler, si besoin, les descriptions en etant le plus fidele possible à la description originale. 
    3- NE JAMAIS GENERER de reponse de ta part si le contexte est vide ou y a pas assez d'info. 
    4- Surtout met chaque produit sur une ligne distincte des autre produits.
    5- repond avec la meme langue du text dans le context
    6- donne moi ta reponse directement sans introduction ou des phrase de ta part.

{contexte}
------------
Réponse :"""

pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)

# Initialize memory
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

def initialize_vectorstore(embedding_function, url, api_key, collection_name):
    qdrantClient = qdrant_client.QdrantClient(url=url, prefer_grpc=True, api_key=api_key)
    return Qdrant(qdrantClient, collection_name, embedding_function)

def initialize_retriever(llm, vectorstore, metadata_field_info, document_content_description):
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={'k': 50}
    )
    return retriever

def extract_text_from_img(img):
    return pytesseract.image_to_string(img)

def extract_text_from_pdf(pdf):
    file = pymupdf.open(pdf)
    p = file[0].get_text()
    if bool(p.strip()):
        text = pymupdf4llm.to_markdown(pdf)
    else:
        text = ''.join(page.get_textpage_ocr().extractTEXT() + "\n\n" for page in file)
    file.close()
    return text

def extract_text_from_docx_xlsx(docx):
    return pymupdf4llm.to_markdown(docx)

def extract_text_from_file(file):
    _, ext = os.path.splitext(file)
    ext = ext.strip('.')
    assert ext in FILE_TYPES, f'wrong file type. The file must be one of the following {FILE_TYPES}'
    if ext in ['docx', 'xlsx']:
        return extract_text_from_docx_xlsx(file)
    if ext == 'pdf':
        return extract_text_from_pdf(file)
    return extract_text_from_img(file)

pdf_chain = pdf_prompt | llm2

def parse_file(filepath, parser=pdf_chain):
    text = extract_text_from_file(filepath)
    result = parser.invoke(text)
    return result.content

document_content_description = "Informations sur le produit, incluant la référence et la description."
metadata_field_info = [
    {'name': "Marque", 'description': "La Marque du produit.", 'type': "string"},
    {'name': "Categorie", 'description': "La Categorie du produit.", 'type': "string"},
]

embedding_function = load_embedding_function()
vectorstore = initialize_vectorstore(embedding_function, url, api_key, collection_name)
retriever = initialize_retriever(llm, vectorstore, metadata_field_info, document_content_description)

prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                Répond seulement si tu as la réponse. affiche les produit un par un, pour chaque produit affiche son nom puis juste en dessous un tableau qui contient ces colonne Référence,Categorie, Marque, Description.        
                Il faut savoir que laptop, ordinateur, ordinateurs portable , pc et poste de travail ont tous le même sens.
                Il faut savoir que téléphone portable et smartphone ont le même sens.
                Il faut savoir que tout autre caractéristique du produit tel que la RAM stockage font partie de la description du produit et il faut filtrer selon la marque et la catégorie seulement.
                Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.
                lorsque une question de similarite entre des produits est poser, il faut dabord commencer par les produit qui ont des processeur qui se ressemble le plus, puis la memoire ram , puis le stockage, puis les autres caracteristique
                la question peut contenir  plusieurs produits avec differentes descriptions, il faut chercher sur les differents produits demandé .
                si je te pose une question sur les question ou les reponses fournient precedement tu doit me repondre selon l'historique.
                tu ne doit pas oublier l'historique car parfois le user continue a te poser des question sur tes reponses que tas deja fourni aupatavant

                Contexte: {context}
                historique :{historique}
                Question: {question}

                Réponse :
                """
            ),
        ]
    )
    
async def query_bot(retriever, embedding_function, question, prompt):
    context = retriever.invoke(question)
    if not context:
        return "Je n'ai pas trouvé de produits correspondants."

    query_embedding = embedding_function.embed_query(question)
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in context]
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    filtered_docs = [doc for doc, similarity in zip(context, similarities) if similarity >= 0.7]

    document_chain = create_stuff_documents_chain(llm, prompt)
    conversation_history = memory.load_memory_variables({})
    result = document_chain.invoke({
        "context": filtered_docs,
        "historique": conversation_history['history'],
        "question": question
    })
    memory.save_context({"question": question}, {"response": result})
    return result

st.title("🧠 Sales Smart Assistant DGF")
if 'messages' not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "file_up_key" not in st.session_state:
    st.session_state.file_up_key = uuid.uuid4().hex

query = st.chat_input('comment puis-je vous aidez?')

with st.sidebar:
    uploaded_file = st.file_uploader("Téléchargez un fichier contenant les produits que vous cherchez", type=FILE_TYPES, key=st.session_state.file_up_key)
    if uploaded_file:
        filepath = uploaded_file.name
        data = uploaded_file.read()
        uploaded_file.close()
        with open(filepath, 'wb') as up_file:
            up_file.write(data)
        st.session_state.extracted_text = parse_file(filepath)
        os.remove(filepath)
        st.session_state.file_up_key = uuid.uuid4().hex
        st.markdown(st.session_state.extracted_text)

extracted_text = st.session_state.extracted_text

if query:
    full_query = f"{query}\n{extracted_text}"
    st.session_state.messages.append({"role": "user", "content": full_query})

    start_time = time.time()
    result = asyncio.run(query_bot(retriever, embedding_function, full_query, prompt))
    exec_time = time.time() - start_time

    st.session_state.messages.append({"role": "ai", "content": f"{result}\n\n(Temps d'exécution: {exec_time:.2f} secondes)"})
    st.session_state.extracted_text = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
