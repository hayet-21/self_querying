import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()
# Charger les variables d'environnement
# Récupérer la clé API Hugging Face
HF_TOKEN = os.getenv('API_TOKEN')
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')

def transform(embedding_function):
    loader = CSVLoader(file_path=DATA_PATH_CSV,
    metadata_columns=["Marque","Categorie"],   # Optional: Include 'Marque' as metadata
    csv_args={"delimiter": ","},  # Specify delimiter if needed
    encoding="utf-8"
    )
    documents = loader.load()
    chroma_instance = Chroma(embedding_function= embedding_function, persist_directory=CHROMA_PATH, collection_name=COLLECTION_CSV)
    # chroma_instance.delete(chroma_instance.get()['ids'])
    chroma_instance.add_documents(documents)
    chroma_instance.persist()
    print("There are", chroma_instance._collection.count(), "documents in the collection")
    
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
        return None

embedding_function = load_embedding_function()


transform(embedding_function)
        


