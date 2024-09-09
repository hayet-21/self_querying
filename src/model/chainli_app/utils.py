import os
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Qdrant
import qdrant_client
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_cohere import CohereRerank
from constants import FILE_TYPES
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import extract_msg
from email.parser import BytesParser
from email import policy
import pymupdf, pymupdf4llm, base64
from PIL import Image
from io import BytesIO
import uuid, base64

def init_cohere(cohere_api_key,top_n):
    return CohereRerank(model="rerank-multilingual-v3.0",cohere_api_key=cohere_api_key,top_n=top_n)


def init_embeddings(model_name, api_key):
    return OpenAIEmbeddings(model= model_name, api_key=api_key)


def init_llm(type, model_name, api_key,temperature):
    if type == 'Groq' :
        return ChatGroq(model_name= model_name, api_key=api_key, temperature=temperature) 
    elif type == 'gpt' :
        return ChatOpenAI(model= model_name, temperature= temperature, api_key= api_key)
    

def initialize_vectorstore(embeddings, QDRANT_URL, QDRANT_API_KEY, collection_name):
    qdrantClient = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)
    return Qdrant(qdrantClient, collection_name, embeddings) #, vector_name='vector_params'

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
              
def parse_file(filepath, parser1, parser2, size=(320, 320)):
    _, ext= os.path.splitext(filepath)
    ext= ext.strip('.').lower()
    assert ext in FILE_TYPES, f'wrong file type. the file must be on of following {FILE_TYPES}'
    
    if ext in ['pdf', 'docx', 'xlsx','msg','eml']:
        text = extract_text_from_file(filepath)
        return parser1.invoke(text)
    else:
        return parse_image(filepath, parser2, size)

# Unique union of retrieved docs
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]