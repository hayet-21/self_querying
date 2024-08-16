import streamlit as st
from pipline import load_embedding_function, initialize_vectorstore, initialize_retriever, query_bot
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

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Charger la fonction d'embedding
embedding_function = load_embedding_function()
file_up_key= uuid.uuid4().hex

# Initialiser le modèle LLM
GROQ_TOKEN = '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'
@st.cache_resource
def llm_generation(modelName,apiKey):         
    llm = ChatGroq(model_name=modelName, api_key=apiKey, temperature=0)
    return llm 

url="https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key= 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'
collection_name="lenovoHP_collection"
FILE_TYPES= ['png', 'jpeg', 'jpg', 'pdf', 'docx', 'xlsx','PNG']
modelName2='gemma2-9b-it'
modelName="llama-3.1-70b-versatile"
llm = llm_generation(modelName,GROQ_TOKEN)
llm2= llm_generation(modelName2,GROQ_TOKEN)
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
vectorstore = initialize_vectorstore(embedding_function,url,api_key,collection_name)
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
if 'messages' not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    print('im in ext text')
    st.session_state.extracted_text =" "
if "file_up_key" not in st.session_state:
    print('im in filu up key')
    st.session_state.file_up_key= uuid.uuid4().hex
query= st.chat_input('comment puis-je vous aidez?')
with st.sidebar:
    uploaded_file =  st.file_uploader("Téléchargez un fichier contenant les produits que vous chercher", type=FILE_TYPES, key=st.session_state.file_up_key)
    if uploaded_file:
        filepath= uploaded_file.name
        data= uploaded_file.read()
        uploaded_file.close()
        uploaded_file= None
        with open(filepath, 'wb') as up_file:
            up_file.write(data)
        st.session_state.extracted_text = parse_file(filepath)
        #os.remove(filepath)
        st.session_state.file_up_key= uuid.uuid4().hex
        st.markdown(st.session_state.extracted_text)

extracted_text= st.session_state.extracted_text

if query:
    full_query= f"{query}\n{extracted_text}"
    # Append the user's input to the chat history
    st.session_state.messages.append({"role": "user", "content": full_query})

    # Delete the temporary file
   
    start_time =time.time()
    # Get the bot's response
    result= asyncio.run(query_bot(retriever, full_query,prompt))
    #print(f"Résultat: {result}, Temps d'exécution: {exec_time} secondes")
    exec_time=time.time() - start_time
    # Append the bot's response to the chat history
    st.session_state.messages.append({"role": "ai", "content" :f"{result}\n\n(Temps d'exécution: {exec_time:.2f} secondes)"})
    st.session_state.extracted_text= ""

    # Display the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

