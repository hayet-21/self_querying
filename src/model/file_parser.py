import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings # embeddings
from langchain_qdrant import QdrantVectorStore as qd # indexing
from langchain_groq import ChatGroq # llm
from langchain_core.output_parsers import StrOutputParser # parser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # template
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from IPython.display import Markdown, HTML # display answer
import pymupdf, pymupdf4llm, pytesseract as ocr
from langchain_core.output_parsers import StrOutputParser # parser

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000

COLLECTION_NAME = 'collection_icecat'
MBD_MODEL = 'intfloat/multilingual-e5-large'
FILE_TYPES= ['.png', '.jpeg', '.jpg', '.pdf']

# embedding model 
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)

# vector database for indexing + retriever
vectordb = qd.from_existing_collection(embedding=embeddings,
        url='https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333',
        prefer_grpc=True,
        api_key= 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA',
        collection_name= COLLECTION_NAME,
        vector_name='',)

retriever= vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 250, 'fetch_k': 50})

# LLM Model
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)

# parser
# output_parser= StrOutputParser()

from langchain_core.prompts import PromptTemplate
pdf_prompt_instruct= """Tu es Un assistant AI super helpful. Etant donnee un contexte qui est le texte brute issue d'image ou de pdf, ton travail est simple. il consiste a: \
1- Extraire toutes les decriptions et les descriptions des produits qui se trouvent à l'interieur du contexte. \
2- Reformuler, si besoin, les descriptions en etant le plus fidele possible à la description originale. \
3- NE JAMAIS GENERER de reponse de ta part si le contexte est vide ou y a pas assez d'info. \
4- Mettre chaque description sur une ligne. \
5- la reponse sera l'entree pour un autre llm qui va chercher c'est produits , demande lui de trouver les produits qui corresponts aux descriptions dans le pdf

{contexte}
------------
Reponse:"""

pdf_prompt= PromptTemplate.from_template(pdf_prompt_instruct)

pdf_chain=  pdf_prompt | llm | StrOutputParser()

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

print(ocr)