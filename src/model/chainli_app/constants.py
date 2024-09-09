import os
from dotenv import load_dotenv
load_dotenv()

if 'openAi8key' in os.environ: 
    API_TOKEN = os.environ['HF_API']
    GROQ_TOKEN = os.environ['GROQ_TOKEN']
    OPENAI_TOKEN = os.environ['openAi8key']

    QDRANT_URL= os.environ['QDRANT_URL']
    QDRANT_API = os.environ['QDRANT_API']
    COLLECTION_NAME = 'OpenAI_SELFQ_collection'

    COHERE_API_KEY = os.environ['COHERE_API_KEY']

    MBD_MODEL = 'text-embedding-3-large'
    LLM_NAME_1= 'gemma2-9b-it'
    LLM_NAME_2= 'llama3-8b-8192'
    LLM_NAME_3= 'llama-3.1-70b-versatile'
    GPT= 'gpt-4o-mini'

    FILE_TYPES= ['pdf', 'docx', 'jpeg', 'png', 'jpg', 'webp']
    