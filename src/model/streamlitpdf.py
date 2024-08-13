import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
import pymupdf4llm

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
MBD_MODEL = 'intfloat/multilingual-e5-large'

# Initialize the embedding model and language model
embeddings_model = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)

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


# Streamlit UI
st.title("PDF Query System")

query = st.chat_input("Ask a question about the PDF:")

if query:
    # Load the PDF and generate embeddings
    file_path = "../data/pdf_query_sample.pdf"
    #pdf_path = "../data/pdf_query_sample.pdf"
    #pdf_path = "../data/pdef2.pdf"
    file_extension = os.path.splitext(file_path)[1].lower()
   
    if file_extension == '.pdf':
        #documents = load_pdf(pdf_path)
        #documents_text = ' '.join([doc.page_content for doc in documents])
        documents_text = pymupdf4llm.to_markdown(file_path)
    elif file_extension == '.docx' :
        doc=load_doc(file_path)
        documents_text= ' '.join([doc.page_content for doc in doc])

    prompt_template = PromptTemplate.from_template(
    "On se basant sur le context  : {documents_text} repond moi a la question: {query} , si la question est sur les produit alors dans la reponse tu doit me fournir que les caracteristique technique des produit ne me donne pas la data de livraison , la quantite etc. Si le context est vide dis moi que tu nas pas trouve"
)
    # Create the prompt    
    #prompt = prompt_template.format(documents_text=documents_text, query=query)
    
    # Get response from the LLM
    chain = prompt_template | llm
    response=chain.invoke({"documents_text": {documents_text},"query":{query}})
    st.markdown(response.content)
