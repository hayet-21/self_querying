import streamlit as st
from RerankingPipline import  initialize_vectorstore, initialize_retriever, query_bot, batch_query_bot
from langchain_groq import ChatGroq
import asyncio
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # parser
import pymupdf, pymupdf4llm
import pytesseract 
from langchain_core.prompts import PromptTemplate
import os
import uuid, base64
from PIL import Image
from io import BytesIO
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
FILE_TYPES= ['png', 'jpeg', 'jpg', 'pdf', 'docx', 'xlsx','msg']
modelName2='gemma2-9b-it'
modelName = "gpt-4o-mini"
#llm2= llm_generation(modelName2,GROQ_TOKEN)
llm2 = ChatOpenAI(model_name=modelName, api_key= openAi8key, temperature=0.3)

pdf_prompt_instruct = """ " Tu es Un assistant AI super helpful. Etant donnee un contexte, ton travail est simple. il consiste a: \
    1- Extraire touts les produit et leurs description des produits qui se trouvent à l'interieur du contexte. \
    2- Reformuler, si besoin, les descriptions en etant le plus fidele possible à la description originale. \
    3- NE JAMAIS GENERER de reponse de ta part si le contexte est vide ou y a pas assez d'info. \
    4-met chaque produit dans une ligne 
    5-repond avec la meme langue du text dans le context\
    6-NE DONNE AUCUN COMMENTAIRE NI INTRODUCTION, JUSTE LES INFORMATIONS SUIVANTES.\

{contexte}
------------
Réponse :"""


# image prompt
image_system_prompt= """ Etant donné l'image ci-dessous, TU fais :
1. **IDENTIFIES** toutes les descriptions completes des produits dans l'image.
2. **Ne GENERE aucune** réponse si l'image n'est pas claire. Retourne "image pas claire" dans ce cas.
3. **Format** : un produit décrit par ligne et une ligne suffit pour un produit, numérotée.
4. **CONTENU**: doit contenir que la description technique pas d'info supplementaire
4. **SANS COMMENTAIRES**
"""
image_user_prompt=  [
                        {
                            "type": "image_url",
                            "image_url":   {"url": "data:image/{img_format};base64,{image_data}",
                                            "detail": "low"}
                        }
                    ]

image_prompt= ChatPromptTemplate.from_messages(
    messages= [
        ('system', image_system_prompt),
        ('user', image_user_prompt)
    ]
)

pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)
pdf_chain=  pdf_prompt | llm2
image_chain= image_prompt | llm | StrOutputParser()

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
    return parse_image(file)

def encode_image(image_path: str, size: tuple[int, int]= (320, 320)):
    _, ext= os.path.splitext(image_path)
    ext= ext.strip('.').lower()
    
    with open(image_path, 'rb') as f:
        img_f= Image.open(BytesIO(f.read())).resize(size, Image.Resampling.BILINEAR)
    
    buffered= BytesIO()
    if ext in ['jpeg', 'jpg']:
        img_format= 'JPEG'
    else:
        img_format= 'PNG'
    img_f.save(buffered, format= img_format)
    return base64.b64encode(buffered.getvalue()).decode(), img_format

def parse_image(image_path: str, chain, size: tuple[320, 320]):
    _, ext= os.path.splitext(image_path)
    ext= ext.strip('.').lower()
    subtypes= ['png', 'jpg', 'jpeg', 'webp']
    assert ext in subtypes, f'wrong file type. the file must be on of following {subtypes}'
    b64_img, img_format= encode_image(image_path, size)
    
    return chain.invoke({'image_data': b64_img, 'img_format': img_format})
              

def parse_file(filepath, parser1=pdf_chain, parser2=image_chain, size=(320, 320)):
    _, ext= os.path.splitext(filepath)
    ext= ext.strip('.').lower()
    assert ext in FILE_TYPES, f'wrong file type. the file must be on of following {FILE_TYPES}'
    
    if ext in ['pdf', 'docx', 'xlsx','msg']:
        text = extract_text_from_file(filepath)
        return parser1.invoke(text)
    else:
        return parse_image(filepath, parser2, size)
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
                Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                Répond seulement si tu as la réponse. affiche les produit un par un, pour chaque produit affiche son nom puis juste en dessous tableau qui contient ces colonne Référence,Categorie, Marque, Description.        
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
with st.sidebar:
    
    st.session_state.query = st.text_area(label='recherche manuelle', placeholder='veuillez introduire votre requete ?', height=150)
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
            full_query= f"{st.session_state.query}{st.session_state.extracted_text}"
            if not full_query.strip() :
                st.warning("La requête est vide. Veuillez entrer une requête valide.")
            else :
                # Append the user's input to the chat history
                st.session_state.messages.append({"role": "user", "content": full_query})
                queries = full_query.strip('\n').split('\n')
                # Delete the temporary file
            
                start_time =time.time()
                # Get the bot's response
                result= asyncio.run(batch_query_bot(retriever, queries,prompt))
                #print(f"Résultat: {result}, Temps d'exécution: {exec_time} secondes")
                exec_time=time.time() - start_time
                # Append the bot's response to the chat history
                st.session_state.messages.append({"role": "ai", "content" :f"{result}\n\n(Temps d'exécution: {exec_time:.2f} secondes)"})
                st.session_state.extracted_text= ""
                st.session_state.query = "" 

# Display the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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
        with st.container():
            full_query= f"{st.session_state.query}{st.session_state.extracted_text}"
            if not full_query.strip() :
                st.warning("La requête est vide. Veuillez entrer une requête valide.")
            else :
                # Append the user's input to the chat history
                st.session_state.messages.append({"role": "user", "content": full_query})
                queries = full_query.strip('\n').split('\n')
                # Delete the temporary file
            
                start_time =time.time()
                # Get the bot's response
                result= asyncio.run(batch_query_bot(retriever, queries,prompt))
                #print(f"Résultat: {result}, Temps d'exécution: {exec_time} secondes")
                exec_time=time.time() - start_time
                # Append the bot's response to the chat history
                st.session_state.messages.append({"role": "ai", "content" :f"{result}\n\n(Temps d'exécution: {exec_time:.2f} secondes)"})
                st.session_state.extracted_text= ""
                st.session_state.query = "" 

        # Ajoutez la logique pour rechercher des produits similaires ici


