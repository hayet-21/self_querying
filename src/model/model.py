from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API Hugging Face
HF_TOKEN = os.getenv('API_TOKEN')
'''
# Initialiser le modèle Hugging Face via HuggingFaceHub
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-2-70b-hf",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    add_to_git_credential=True,
)
'''

GROQ_TOKEN='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'
llm = ChatGroq(model_name='llama3-8b-8192', api_key= GROQ_TOKEN,temperature=0)

question = "who won the world cup in 1994? "

# Créer le template de prompt amélioré
template = """
You are an AI model that provides brief, direct and concise answers to factual questions. 
Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

llm_chain = prompt | llm
print(llm_chain.invoke({"question": question}))
