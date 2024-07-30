from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
import traceback
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

# Charger les variables d'environnement
load_dotenv()

HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')

embedding_function = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="intfloat/multilingual-e5-large"
        )

chroma_db = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_CSV,
            embedding_function=embedding_function
        )

query = "give 3 products of brand hp noir "

answer = chroma_db.similarity_search(query)

print(answer)