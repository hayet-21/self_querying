import streamlit as st
from pipline import  initialize_vectorstore, initialize_retriever, query_bot, batch_query_bot
from langchain_groq import ChatGroq
import asyncio
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # parser
import pymupdf, pymupdf4llm
import pytesseract 
from langchain_core.prompts import PromptTemplate
import os
import uuid
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Dell\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

load_dotenv()
# Set your OpenAI API key
openAi8key=os.getenv('openAi8key')
# Charger la fonction d'embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openAi8key)
file_up_key= uuid.uuid4().hex
# Initialize the chat model
modelName = "gpt-4o-mini"
llm = ChatOpenAI(model_name=modelName, api_key= openAi8key, temperature=0.3)
# Initialiser le modèle LLM
GROQ_TOKEN = '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'

@st.cache_resource
def llm_generation(modelName, GROQ_TOKEN):         
    llm = ChatGroq(model_name=modelName, api_key=GROQ_TOKEN, temperature=0)
    return llm

url="https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io"
api_key= "lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA"
collection_name="OpenAI_SELFQ_collection"
FILE_TYPES= ['png', 'jpeg', 'jpg', 'pdf', 'docx', 'xlsx','PNG']
modelName2='gemma2-9b-it'
modelName = "gpt-4o-mini"

llm2= llm_generation(modelName2,GROQ_TOKEN)
pdf_prompt_instruct = """ Tu es Un assistant AI super helpful. Etant donnee un contexte, ton travail est simple. il consiste a: \
    1- Extraire touts les produit et leurs description des produits qui se trouvent à l'interieur du contexte. \
    2- Reformuler, si besoin, les descriptions en etant le plus fidele possible à la description originale. \
    3- NE JAMAIS GENERER de reponse de ta part si le contexte est vide ou y a pas assez d'info. \
    4-met chaque produit dans une ligne 
    5-repond avec la meme langue du text dans le context\
    6-NE DONNE AUCUN COMMENTAIRE NI INTRODUCTION, JUSTE LES INFORMATIONS SUIVANTES.\

{contexte}
------------
Réponse :"""

pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)
pdf_chain=  pdf_prompt | llm2

def extract_text_from_img(img):
    text = pytesseract.image_to_string(img)
    return text

def extract_text_from_pdf(pdf):
        file= pymupdf.open(pdf)
        p= file[0].get_text()
        text= ''
        if bool(p.strip()):
                text= pymupdf4llm.to_markdown(pdf)
        else:
                for page in file:
                        tp= page.get_textpage_ocr()
                        text += f" {tp.extractTEXT()} \n\n" 
        file.close()
        return text


def extract_text_from_docx_xlsx(docx):
        text= pymupdf4llm.to_markdown(docx)
        return text
    
def extract_text_from_file(file):
    _, ext = os.path.splitext(file)
    ext= ext.strip('.')
    assert ext in FILE_TYPES, f'wrong file type. The file must be one of the following {FILE_TYPES}'
        # print(ext)
    if ext in ['docx', 'xlsx']:
            return extract_text_from_docx_xlsx(file)
    if ext == 'pdf':
            return extract_text_from_pdf(file)
    return extract_text_from_img(file)
                        

def parse_file(filepath, parser=pdf_chain):
    text = extract_text_from_file(filepath)
    result=parser.invoke(text)
    return result.content

document_content_description = "Informations sur le produit, incluant la référence et la description."
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
# Initialiser le vectorstore et le retriever

vectorstore = initialize_vectorstore(embeddings,url,api_key,collection_name)
retriever = initialize_retriever(llm, vectorstore,metadata_field_info,document_content_description)
# Construire le template de prompt
prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                Tu es un assistant vendeur spécialisé dans la recherche de produits informatiques. Tu as accès au contexte seulement, et tu dois utiliser ces informations pour répondre aux questions de l'utilisateur. Ne génère aucune information si elle n'est pas présente dans le contexte. 

            - Réponds uniquement si tu as la réponse.
            - Affiche les produits un par un.
            - Pour chaque produit, affiche d'abord son nom, suivi d'un tableau contenant les colonnes suivantes : Référence, Catégorie, Marque, Description.
            - Considère que "laptop", "ordinateur", "ordinateur portable", "PC" et "poste de travail" ont tous le même sens.
            - Considère que "téléphone portable" et "smartphone" ont le même sens.
            - Toutes les caractéristiques du produit, comme la RAM ou le stockage, font partie de la **Description**. Tu dois filtrer les produits selon la marque et la catégorie uniquement.
            - Si le contexte est vide, indique clairement que tu n'as trouvé aucun produit correspondant.

            Lorsque l'utilisateur pose une question sur la similarité entre des produits, commence par rechercher les produits ayant des **processeurs similaires**, puis continue avec la **mémoire RAM**, ensuite le **stockage**, et enfin les autres caractéristiques. 

            - La question peut contenir plusieurs produits avec différentes descriptions ; tu dois rechercher chaque produit demandé.
            - Si l'utilisateur te pose une question à propos des réponses que tu as déjà fournies, référez-toi à l'historique des conversations pour donner une réponse cohérente.

            Contexte : {context}
            Historique : {historique}
            Question : {question}

            Réponse :

                """
            ),
        ]
    )
    

st.title("🧠 Sales Smart Assistant DGF")

if "query" not in st.session_state:
    st.session_state.query = " "

if 'messages' not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    print('im in ext text')
    st.session_state.extracted_text =" "
if "file_up_key" not in st.session_state:

    print('im in filu up key')
    st.session_state.file_up_key= uuid.uuid4().hex

# CSS pour ajuster la taille des boutons
st.markdown("""
<style>
.stButton button {
    height: 60px;
    width: 200px;
    font-size: 20px;
}
.send-button {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 100;
}
.disabled-button {
    background-color: #d3d3d3; /* Couleur de fond grise pour les boutons désactivés */
    color: #a9a9a9; /* Couleur du texte grise pour les boutons désactivés */
    cursor: not-allowed; /* Curseur pour indiquer que le bouton est désactivé */
}
.disabled-button:hover {
    background-color: #d3d3d3; /* Couleur de fond grise pour le survol des boutons désactivés */
}
.message {
    text-align: center;
    font-size: 15px; /* Même taille de police que pour la zone de texte */
    margin: 5px 0; /* Espacement autour du message */
}
           
</style>
""", unsafe_allow_html=True)

# Organisation des boutons en colonnes
col1, col2, col3 = st.columns(3)

with col1:
    # Bouton désactivé
    st.button("Disponibilité", disabled=True, help="Ce bouton est temporairement désactivé.", key="disabled_button_1")

with col2:
    # Bouton désactivé
    st.button("Prix", disabled=True, help="Ce bouton est temporairement désactivé.", key="disabled_button_2")
with col3:
    if st.button("Produits Similaires"):
        st.write("Recherchez des produits similaires.")
        # Ajoutez la logique pour rechercher des produits similaires ici



with st.sidebar:
    
    query = st.text_area('Collez votre texte ici :')
    uploaded_file =  st.file_uploader("Téléchargez un fichier contenant les produits que vous chercher", type=FILE_TYPES, key=st.session_state.file_up_key)
    if uploaded_file:
        filepath= uploaded_file.name
        data= uploaded_file.read()
        uploaded_file.close()
        uploaded_file= None
        with open(filepath, 'wb') as up_file:
            up_file.write(data)
        st.session_state.extracted_text = parse_file(filepath)
        os.remove(filepath)
        st.session_state.file_up_key= uuid.uuid4().hex
        st.markdown(st.session_state.extracted_text.strip('\n').split('\n'))
    
    # Ajouter un conteneur pour le bouton
    with st.container():
        if st.button('Rechercher '):
            full_query= f"{query}{st.session_state.extracted_text}"
            
            queries = full_query.strip('\n').split('\n')
            if queries and any(q.strip() for q in queries):  # Vérifie que queries n'est pas vide
                start_time = time.time()
                # Obtenir la réponse du bot
                st.session_state.messages.append({"role": "user", "content": full_query})
                result = asyncio.run(batch_query_bot(retriever, queries, prompt))
                exec_time = time.time() - start_time
                # Ajouter la réponse du bot à l'historique
                st.session_state.messages.append({"role": "ai", "content": f"{result}\n\n(Temps d'exécution: {exec_time:.2f} secondes)"})
            else:
                result = "Aucune requête valide trouvée."
                st.session_state.messages.append({"role": "ai", "content": result})
            
            # Réinitialiser la query et le texte extrait
            st.session_state.extracted_text = ""
            query = "" 

# Display the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

