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
import uuid, base64
from PIL import Image
from io import BytesIO
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain
import extract_msg
from email.parser import BytesParser
from email import policy
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

# Set your OpenAI API key
openAi8key=os.getenv('openAi8key')
# Charger la fonction d'embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openAi8key)
file_up_key= uuid.uuid4().hex
# Initialize the chat model
modelName = "gpt-4o-mini"
llm = ChatOpenAI(model_name=modelName, api_key= openAi8key, temperature=0.5)
# Initialiser le modèle LLM
GROQ_TOKEN = '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'

@st.cache_resource
def llm_generation(modelName, GROQ_TOKEN):         
    llm = ChatGroq(model_name=modelName, api_key=GROQ_TOKEN, temperature=0)
    return llm


url="https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io"
api_key= "lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA"
collection_name="OpenAI_SELFQ_collection"
FILE_TYPES= ['png', 'jpeg', 'jpg', 'pdf', 'docx', 'xlsx','msg','eml']
modelName2='gemma2-9b-it'
modelName = "gpt-4o-mini"
#llm2= llm_generation(modelName2,GROQ_TOKEN)
llm2 = ChatOpenAI(model_name=modelName, api_key= openAi8key, temperature=0.3)

pdf_prompt_instruct =  """Tu es un **Assistant Extraction de Descriptions :**
1. **Extraction** : Identifie toutes les descriptions de produits dans le contexte.
2. **Reformulation** : Si nécessaire, reformule les descriptions tout en restant fidèle à l'original.
3. **Contexte Vide** : Ne génère aucune réponse si le contexte est vide ou insuffisant.
4. **Format** : un produit decrit par ligne et une ligne suffit un produit, numérotée.
5. **Réponse Brute** : Retourne uniquement les descriptions sans commentaire.
6. **Nombre**: N'oublie aucun produit.
{contexte}
Reponse:"""
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
pdf_chain=  pdf_prompt | llm2 | StrOutputParser()
image_chain= image_prompt | llm | StrOutputParser()

# Créer un template de prompt pour la reformulation
prompt_template = """Tu es un Assistant de Reformulation spécialisé dans les requêtes produit :
   *Base de connaissance *:
    - voici quelque marque de laptops : Dell, HP, Lenovo, Acer, Asus, Microsoft, MSI, Razer, Samsung, Toshiba, Sony (Vaio), Alienware, Gigabyte, Huawei, LG, Xiaomi, Fujitsu, Chuwi, Clevo, Eurocom
   *Instructions*:
    1. Vérifie si la question inclut une marque. Si oui, reformule la question selon les instructions ci-dessous ; sinon, retourne la question telle qu'elle est.
    2. Objectif : Reformuler chaque requête de manière complète, auto-suffisante, en excluant  la marque mentionnée s'il y en a une. Fournis trois reformulations pour chaque question, en gardant les reformulations proches de l'originale.
        Exemple 1:
        - Question : "Laptop Lenovo i7 16GB RAM 512GB SSD 14" "
        - Reformulations : 
          1. "Laptop i7 16GB RAM 512GB SSD 14" 
          2. "Laptop i7 16GB RAM avec 512GB SSD 14 pouces"
          3. "Laptop i7 16GB RAM 512GB SSD 14 pouces"
        Exemple 2:
        - Question : " 21D60011FR lenovo ThinkPad P16 Gen 1 intel i7 16 GB 1 TB"
        - Reformulations : 
          1. "21D60011FR ThinkPad P16 Gen 1 intel i7 16 GB 1 TB"
          2. "21D60011FR ThinkPad P16 Gen 1 intel i7 16 GB 1 To"
          3. "21D60011FR ThinkPad P16 Gen 1 intel i7 16 GB avec 1 To"
    3. Pas de Réponse : Ne réponds jamais à la question, reformule-la uniquement si nécessaire.
    4. Met chaque question reformulée sur une ligne, en indiquant les reformulations numérotées.
    5. N'ajoute pas des saut de ligne entre les questions reformulées.
    Questions:
    {questions}
"""
prompt_template_rech= """Tu es un Assistant de Reformulation spécialisé dans les requêtes produit :
   *Base de connaissance *:
    - Voici quelques marques de laptops : Dell, HP, Lenovo, Acer, Asus, Microsoft, MSI, Razer, Samsung, Toshiba, Sony (Vaio), Alienware, Gigabyte, Huawei, LG, Xiaomi, Fujitsu, Chuwi, Clevo, Eurocom
   *Instructions*:
    1. Reformule chaque question pour la rendre plus complète ou plus claire, sans enlever la marque mentionnée.
    2. Objectif : Fournis trois reformulations pour chaque question. Chaque reformulation doit rester fidèle à l’original tout en offrant une légère variation dans la phrasing ou la structure.
        Exemple 1:
        - Question : "Laptop Lenovo i7 16GB RAM 512GB SSD 14"
        - Reformulations : 
          1. "Laptop Lenovo avec processeur i7, 16GB de RAM et stockage de 512GB SSD, écran de 14 pouces"
          2. "Laptop Lenovo i7, 16GB RAM, 512GB SSD, taille de l'écran 14 pouces"
          3. "Laptop Lenovo, i7, 16GB de RAM, SSD de 512GB, écran 14 pouces"

        Exemple 2:
        -"21D60011FR lenovo ThinkPad P16 Gen 1 intel i7 16 GB 1 TB"
        - "21D60011FR Lenovo ThinkPad P16 Gen 1 avec Intel i7, 16GB de RAM et disque dur de 1TB","21D60011FR ThinkPad P16 Gen 1 Lenovo, processeur Intel i7, 16GB RAM, stockage 1TB","21D60011FR Lenovo ThinkPad P16 Gen 1, Intel i7, 16GB de RAM, avec 1TB de stockage"

        Exemple 3:
        - "HP Pavilion x360 Intel Core i5 8GB RAM 256GB SSD"
        -"HP Pavilion x360 avec processeur Intel Core i5, 8GB RAM, et disque SSD de 256GB","HP Pavilion x360 Intel Core i5, 8GB de RAM et 256GB SSD","HP Pavilion x360, Core i5, 8GB RAM, 256GB SSD"

        Exemple 4:
        - "MacBook Pro M1 16GB 512GB SSD"
        -"MacBook Pro avec puce M1, 16GB de RAM et 512GB SSD","MacBook Pro M1, 16GB RAM, 512GB SSD","MacBook Pro M1 avec 16GB de RAM et 512GB de stockage SSD"
    Ne réponds pas directement aux questions ; reformule uniquement en suivant les instructions ci-dessus.
    Mets chaque question reformulée sur une seule ligne pour garantir une lisibilité optimale.
    N'ajoute pas de saut de ligne entre les questions reformulées pour maintenir la cohérence.
    Questions:
    {questions}
"""
prompt_template_metier = """Assistant Spécialisé en Reformulation de Requêtes metier 
    *Base de connaissance* :
    Marques de Laptops : Apple, Dell, HP, Lenovo, Acer, Asus, Microsoft, MSI, Razer, Samsung, Toshiba, Sony (Vaio), Alienware, Gigabyte, Huawei, LG, Xiaomi, Fujitsu, Chuwi, Clevo, Eurocom.

    *Synonymes de catégories* :

    Laptop : Laptop, Ordinateur Portable, PC, PC portable.
    All in One : All-in-One, PC Tout-en-Un.
    Poste de Travail : Poste de Travail, PC, Ordinateur, Desktop, unité centrale.
    Station Mobile : Laptop gaming, laptop haute performance.
    Station de Travail : Station de Travail, Workstation, desktop haute performance, desktop gaming, unité centrale gaming.
    Écran : Écran, Moniteur, Monitor.
    Téléphone : Téléphone, Smartphone.
    Imprimante : Imprimante, Printer.
    Instructions de Reformulation :

    *Remplacement des synonymes* :

    Distingue clairement entre les catégories de produits pour éviter toute confusion. Par exemple, "desktop" se réfère à un poste de travail standard, tandis que "desktop haute performance" désigne une station de travail.
    Remplace les synonymes de catégories par la catégorie principale correspondante lors de la reformulation. Par exemple, remplace "PC portable" par "Laptop". Applique cette règle à toutes les catégories listées ci-dessus.

    *Reformulation basée sur un métier spécifique* :

    Si la question concerne plusieurs types de produits, reformule chaque produit séparément en décrivant ses caractéristiques essentielles. Utilise des phrases concises et informatives.
    Conserve les catégories, marques ou modèles spécifiques mentionnés dans la question.
    Exemple : "Pouvez-vous nous recommander des laptops pour nos développeurs de logiciels, des stations mobiles pour nos graphistes, et des moniteurs pour les postes de travail de notre équipe technique ?"
    Reformulation :

    "Laptop avec processeur i7, 16GB de RAM, 1TB SSD et carte graphique dédiée."
    "Station mobile équipée d'un processeur i7, 32GB de RAM, 1TB SSD et carte graphique dédiée avec un écran 4K."
    "Moniteur de 27 pouces, résolution 1440p et temps de réponse rapide."
    *Conseils supplémentaires* :

    Assure-toi que chaque reformulation soit clairement séparée et autonome.
    Présente chaque reformulation sur une ligne distincte pour une clarté maximale.
    *Instructions supplémentaires* :

    Ne réponds pas directement aux questions ; reformule uniquement selon les instructions ci-dessus.
    Mets chaque reformulation sur une ligne séparée.
    Assure-toi que chaque description de produit est claire et distincte.
    Questions : {questions}



"""

def reformulate_queries_with_llm(queries: list, llm ,prompt_template) -> list:
    # Join the queries into a single string with newline separation
    joined_queries = "\n".join([f"Question: {query}" for query in queries])
    
    # Prepare the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["questions"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Execute the chain to get the reformulations
    reformulated_queries = chain.run(questions=joined_queries)
    
    # Split the reformulated queries into a list
    reformulated_list = reformulated_queries.strip().split('\n')
    
    return reformulated_list

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

#import extract_msg
def extract_text_and_attachments_from_msg(path, temp_dir):
    msg = extract_msg.Message(path)
    # Extraire le corps du message
    msg_message = "le contenu de l'email est : \n\n " + msg.body
    
    msg.close()
    return msg_message
def extract_eml_file(file_path,):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Extract body
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get("Content-Disposition")

            # Extract the plain text part
            if content_type == "text/plain" and disposition is None:
                extracted_text = part.get_payload(decode=True).decode(part.get_content_charset())
    else:
        # For non-multipart emails
        extracted_text = msg.get_payload(decode=True).decode(msg.get_content_charset())

    return extracted_text
def extract_text_from_file(file):
    _, ext = os.path.splitext(file)
    ext= ext.strip('.')
    assert ext in FILE_TYPES, f'wrong file type. The file must be one of the following {FILE_TYPES}'
        # print(ext)
    if ext in ['docx', 'xlsx']:
            return extract_text_from_docx_xlsx(file)
    if ext == 'pdf':
            return extract_text_from_pdf(file)
    if ext == 'msg':
        temp_dir = os.path.join('temp_attachments', uuid.uuid4().hex)
        os.makedirs(temp_dir, exist_ok=True)
        msg_text = extract_text_and_attachments_from_msg(file, temp_dir)
        # Nettoyer les fichiers temporaires
        for root, dirs, files in os.walk(temp_dir):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(temp_dir)
        return msg_text
    if ext =='eml':
        return extract_eml_file(file)
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
    
    if ext in ['pdf', 'docx', 'xlsx','msg','eml']:
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
prompt_similarite = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
               *Base de savoir*:
                1. *Synonymes de catégories* :
                    - *Laptop* : Laptop, Ordinateur Portable, PC
                    - *All in One*: All-in-One, PC Tout-en-Un
                    - *Poste de Travail* : Poste de Travail, PC, Ordinateur, Desktop, unité centrale
                    - *Station Mobile*: Laptop gaming, laptop haute performance
                    - *Station de Travail*: workstation, desktop haute performance, desktop gaming, unité centrale gaming
                    - *Écran* : Écran, Moniteur, Monitor
                    - *Téléphone* : Téléphone, Smartphone
                    - *Imprimante* : Imprimante, Printer

                2. *Priorités de classement* selon les catégories :

                | Catégorie | Priorité 1 | Priorité 2 | Priorité 3 | Priorité 4 | Priorité 5 | Priorité 6 | Priorité 7 | Priorité 8 |
                |---|---|---|---|---|---|---|---|---|
                | Laptops, All-in-One | Part Number | Modèle | Marque | Écran (taille, résolution, tactile) | CPU (modèle, famille, génération) | RAM / Stockage | Autres | N/A |
                | Station Mobile | Part Number | Modèle | Marque | Écran (taille, résolution, tactile) | CPU (modèle, famille, génération) | GPU | RAM / Stockage | Autres |
                | Desktop | Part Number | Modèle | Marque | CPU (modèle, famille, génération) | Format (tour, SFF, mini, etc.) | RAM / Stockage | Autres | N/A |
                | Workstation | Part Number | Modèle | Marque | CPU (modèle, famille, génération) | Format (tour, SFF, mini, etc.) | GPU | RAM / Stockage | Autres |
                | Imprimante | Part Number | Modèle | Marque | Type (couleur, noir et blanc) | Fonctionnalité (numérisation, recto-verso, photocopie, etc.) | Vitesse (PPM) | Papier (capacité, formats A4, A3, type) | Connectivité (Bluetooth, Wi-Fi, USB) |

                ---

                *Instructions*:
                0. dites combien de produit dans le contexte
                1. Utilisez le contexte fourni entre triple crochets pour répondre à la requête en listant tous les produits. 
                2. Identifiez la catégorie appropriée en utilisant la liste des synonymes.
                3. Classez les produits par ordre décroissant de similarité avec la description de référence donnée, en suivant l'ordre des priorités de 1 à 8, comme le montre l'example ci-dessous entre triple parenthèses.
                4. Si plusieurs produits sont à égalité pour une priorité donnée, passez à la suivante pour affiner le classement.
                5. En cas d'égalité après application de toutes les priorités, maintenez leur ordre d'origine ou utilisez des critères supplémentaires si disponibles.
                6. Utilisez le contexte fourni entre triple accolades pour répondre à la requête en listant les produits. Si le contexte est vide ou que les instructions ne s'appliquent pas, répondez "pas d'équivalents".
                7. Format de réponse : tableau avec les colonnes suivantes:
                    - Référence : Part number
                    - Catégorie : Type de produit (ex. ordinateur, téléphone)
                    - Marque : Marque du produit
                    - Description : Description complète du produit
                8. Si le contexte est vide ou que qu'il ne contient pas les produits equivalents a la question : répondez "pas de produits équivalents".
                *Pas de commentaire*


                ---

                *Exemple* :
                (((la requete est trouver des equivalents pour :\
                    8A4H6EA, HP Dragonfly 13.5 G4. Type de produit: Ordinateur portable, Format: Clapet. Famille de processeur: Intel® Core™ i7, Modèle de processeur: i7-1355U. Taille de l'écran: 34,3 cm (13.5"), Type HD: WUXGA+, Résolution de l'écran: 1920 x 1280 pixels, Écran tactile. Mémoire interne: 16 Go, Type de mémoire interne: LPDDR5-SDRAM. Capacité totale de stockage: 512 Go, Supports de stockage: SSD. Modèle d'adaptateur graphique inclus: Intel Iris Xe Graphics. Système d'exploitation installé: Windows 11 Pro. Couleur du produit: Bleu

                Les priorités seront :

                1. Part Number: - Si un part number (référence) est présent, les produits avec le même part number que la référence sont prioritaires. Donc : Si un produit HP Dragonfly 13.5 G4 avec le part number "8A4H6EA" est trouvé, il sera classé en premier.

                2. Modèle: Ensuite classez les produits avec le même modèle ("Dragonfly G4") en priorité. Donc : Tous les HP Dragonfly G4 seront classés ici.

                3. Marque: Ensuite, classez les produits de la même marque (HP), même si le modèle diffère. Donc : Les autres Laptops HP (par exemple, HP Spectre, HP EliteBook) seront classés ici.

                4. Écran (taille, résolution, tactile): Si aucun produit ne correspond à la marque, listez les produits avec un écran similaire (13.5 pouces, WUXGA+, tactile), même si la marque et le modèle diffèrent. Donc: Laptops d'autres marques avec un écran de 13.5 pouces, WUXGA+ tactile, comme un Dell XPS 13.

                5. CPU (famille, modèle, génération): Si aucun produit ne correspond à l'écran, classez les produits avec le même CPU (Intel Core i7, i7-1355U), même si la marque, le modèle, et l'écran diffèrent. Donc : Laptops avec un Intel Core i7, i7-1355U, indépendamment de la marque et du modèle.

                6. RAM et Stockage: Ensuite, passez aux produits ayant une RAM de 16 Go et un SSD de 512 Go, même si tous les autres critères ne correspondent pas. Donc: Laptops avec 16 Go RAM et 512 Go SSD, sans se soucier de la marque, du modèle, ou du CPU.

                7. Autres: Enfin, considérez les autres spécifications, comme la carte graphique intégrée (Intel Iris Xe Graphics), le système d'exploitation (Windows 11 Pro), ou la couleur (bleu). Donc: Laptops avec Intel Iris Xe Graphics, même s'ils n'ont pas de CPU i7, 16 Go RAM, ou 512 Go SSD.
                )))


                Contexte: {{{context}}}
                historique :{historique}
                Question: {question}

                Réponse :Voici une liste de produits équivalents au vôtre :
                """
            ),
        ]
)
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
   
from langchain_core.messages import AIMessage, HumanMessage 
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

if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False  # Variable pour gérer le clic du bouton de recherche

if "similaire_clicked" not in st.session_state:
    st.session_state.similaire_clicked = False  # Variable pour gérer le clic du bouton de produit similaire

if "proposition_clicked" not in st.session_state:
    st.session_state.proposition_clicked = False  # Variable pour gérer le clic du bouton de proposition selon metier

# conversation
def chat_display():
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
            
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
with st.sidebar:
    
    st.session_state.query = st.text_area(label='recherche manuelle', placeholder='Comment puis-je vous aider?', height=150)
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
    
    with col1:
        # Bouton désactivé
        st.button("Disponibilité", disabled=True, help="Ce bouton est temporairement désactivé.", key="disabled_button_1")

    with col2:
        # Bouton désactivé
        st.button("Prix", disabled=True, help="Ce bouton est temporairement désactivé.", key="disabled_button_2")
        if st.button("proposition selon un metier"):
            st.session_state.proposition_clicked=True

    with col3:
        if st.button("Produits Similaires"):
                st.session_state.similaire_clicked=True

    # Ajouter un conteneur pour le bouton
    with st.container():
        if st.button('Rechercher '):
            st.session_state.search_clicked = True

chat_display()

if st.session_state.proposition_clicked :
     
    full_query= f"{st.session_state.query}{st.session_state.extracted_text}"
    if not full_query.strip() :
        st.warning("La requête est vide. Veuillez entrer une requête valide.")
    else :
        # Append the user's input to the chat history
        st.session_state.messages.append(HumanMessage(content=full_query))
        with st.chat_message("Human"):
            st.markdown(st.session_state.query)

        queries = full_query.strip('\n').split('\n')
        # Delete the temporary file
        reformulated_queries = reformulate_queries_with_llm(queries, llm,prompt_template_metier)
        print(reformulated_queries)

        # Get the bot's response
        with st.chat_message("AI"): 
            # Obtenir la réponse du bot sous forme de générateur
            start_time =time.time()
            result_generator = st.write_stream(batch_query_bot(retriever, reformulated_queries, prompt_similarite))
            exec_time=time.time() - start_time
        result_generator += f'\n\n---\n\nexecution time: {exec_time:.2f}s'
        st.session_state.messages.append(AIMessage(content=result_generator))

        # Append the bot's response to the chat history
        st.session_state.proposition_clicked = False
        st.session_state.extracted_text= ""
        st.session_state.query = "" 


if st.session_state.similaire_clicked :
     
    full_query= f"{st.session_state.query}{st.session_state.extracted_text}"
    if not full_query.strip() :
        st.warning("La requête est vide. Veuillez entrer une requête valide.")
    else :
        # Append the user's input to the chat history
        st.session_state.messages.append(HumanMessage(content=full_query))
        with st.chat_message("Human"):
            st.markdown(st.session_state.query)

        queries = full_query.strip('\n').split('\n')
        # Delete the temporary file
        reformulated_queries = reformulate_queries_with_llm(queries, llm,prompt_template)
        print(reformulated_queries)

        # Get the bot's response
        with st.chat_message("AI"): 
            # Obtenir la réponse du bot sous forme de générateur
            start_time =time.time()
            result_generator = st.write_stream(batch_query_bot(retriever, reformulated_queries, prompt_similarite))
            exec_time=time.time() - start_time
        result_generator += f'\n\n---\n\nexecution time: {exec_time:.2f}s'
        st.session_state.messages.append(AIMessage(content=result_generator))

        # Append the bot's response to the chat history
        st.session_state.similaire_clicked = False
        st.session_state.extracted_text= ""
        st.session_state.query = "" 


if st.session_state.search_clicked :
     
    full_query= f"{st.session_state.query}{st.session_state.extracted_text}"
    if not full_query.strip() :
        st.warning("La requête est vide. Veuillez entrer une requête valide.")
    else :
        # Append the user's input to the chat history
        st.session_state.messages.append(HumanMessage(content=full_query))
        with st.chat_message("Human"):
            st.markdown(st.session_state.query)

        queries = full_query.strip('\n').split('\n')
        # Delete the temporary file
        reformulated_queries = reformulate_queries_with_llm(queries, llm,prompt_template_rech)
        print(reformulated_queries)

        # Get the bot's response
        with st.chat_message("AI"): 
            # Obtenir la réponse du bot sous forme de générateur
            start_time =time.time()
            result_generator = st.write_stream(batch_query_bot(retriever, reformulated_queries, prompt))
            exec_time=time.time() - start_time
        result_generator += f'\n\n---\n\nexecution time: {exec_time:.2f}s'
        st.session_state.messages.append(AIMessage(content=result_generator))

        # Append the bot's response to the chat history
        st.session_state.search_clicked = False
        st.session_state.extracted_text= ""
        st.session_state.query = "" 



