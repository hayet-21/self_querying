import os
import traceback
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Charger les variables d'environnement
load_dotenv()

HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')

# Fonction pour charger le modèle BLOOM via l'API Hugging Face
def load_model():
    try:
        model_name = "bigscience/bloom"
        # Configuration du modèle Hugging Face BLOOM via API
        llm = OpenAI(model=model_name, api_key=HF_TOKEN, temperature=0.5)
        print(f"Model {model_name} loaded successfully via API.")
        return llm
    except Exception as e:
        print("Error loading the model:", e)
        print(traceback.format_exc())
        return None

# Charger l'embedding model de Hugging Face
def load_embedding_function():
    try:
        embedding_function = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="intfloat/multilingual-e5-large"
        )
        print("Embedding function loaded successfully.")
        return embedding_function
    except Exception as e:
        print("Error loading embedding function:", e)
        print(traceback.format_exc())
        return None

# Initialiser la base de données vectorielle Chroma
def initialize_chroma(embedding_function):
    try:
        chroma_db = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_CSV,
            embedding_function=embedding_function
        )
        print("ChromaDB initialized successfully.")
        return chroma_db
    except Exception as e:
        print("Error initializing ChromaDB:", e)
        print(traceback.format_exc())
        return None

# Créer le récupérateur SelfQueryRetriever
def create_self_query_retriever(chroma_db: Chroma, llm) -> BaseRetriever:
    try:
        retriever = SelfQueryRetriever.from_llm(
            vectorstore=chroma_db,
            llm=llm,
            search_kwargs={"k": 5}
        )
        print("SelfQueryRetriever created successfully.")
        return retriever
    except Exception as e:
        print("Error creating SelfQueryRetriever:", e)
        print(traceback.format_exc())
        return None

# Créer la chaîne de questions-réponses
def create_qa_chain(llm, retriever) -> RetrievalQA:
    try:
        # Template de prompt pour le modèle BLOOM
        template = """
        Given the following question, please find the most relevant documents from the database and answer the question.
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate.from_template(template, input_variables=["question"])
        qa_chain = RetrievalQA(
            retriever=retriever,
            llm_chain=LLMChain(llm=llm, prompt=prompt)
        )
        print("QA Chain created successfully.")
        return qa_chain
    except Exception as e:
        print("Error creating QA Chain:", e)
        print(traceback.format_exc())
        return None

# Fonction principale pour gérer le QA
def main():
    # Charger le modèle et l'embedding function
    llm = load_model()
    embedding_function = load_embedding_function()

    if not llm or not embedding_function:
        print("Failed to load necessary components.")
        return

    # Initialiser ChromaDB
    chroma_db = initialize_chroma(embedding_function)
    if not chroma_db:
        print("Failed to initialize ChromaDB.")
        return

    # Créer SelfQueryRetriever
    retriever = create_self_query_retriever(chroma_db, llm)
    if not retriever:
        print("Failed to create SelfQueryRetriever.")
        return

    # Créer la chaîne de QA
    qa_chain = create_qa_chain(llm, retriever)
    if not qa_chain:
        print("Failed to create QA Chain.")
        return

    # Exemple de question
    question = "give me 3 laptops of different brands that has i7 RAM 16GB ?"
    
    try:
        # Utiliser la chaîne de QA pour répondre à la question
        answer = qa_chain.run(question)
        print(f"Question: {question}\nAnswer: {answer}")
    except Exception as e:
        print("Error running the QA Chain:", e)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
