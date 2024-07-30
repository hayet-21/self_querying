from dotenv import load_dotenv
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from utils_processing import load_documents, split_text, save_to_chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))  # Ajustez cette valeur selon votre système
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
HF_TOKEN = os.getenv('API_TOKEN')

def generate_data_store():
    try:
        loader = CSVLoader(file_path=DATA_PATH_CSV,source_column="Ref_produit",metadata_column=["Marque"], encoding='UTF-8')
        documents = load_documents(loader)
        if not documents:
            return

        chunks = split_text(documents)
        if not chunks:
            return
              # Assurez-vous de définir embedding_function avant de l'utiliser
        embedding_function = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="intfloat/multilingual-e5-large")
        save_to_chroma(documents, CHROMA_PATH, COLLECTION_CSV, embedding_function)
    except Exception as e:
        print(f"An error occurred in generate_data_store: {e}")
    except Exception as e:
        print(f"An error occurred in generate_data_store: {e}")

if __name__ == "__main__":
    generate_data_store()
