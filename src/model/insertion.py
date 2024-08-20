import pandas as pd
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore as qd
from langchain.document_loaders import CSVLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Constants
API_TOKEN = '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'
MBD_MODEL = 'intfloat/multilingual-e5-large'
QDRANT_URL = "https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
COLLECTION_NAME = "lenovoHP_collection"
API_KEY = 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'

# Charger les données du CSV avec CSVLoader
loader = CSVLoader(
    file_path='/content/lenovoHP_clean (1).csv',  # Assurez-vous que le chemin du fichier est correct
    metadata_columns=["Marque", "Categorie"],
    csv_args={"delimiter": ","},
    encoding="utf-8"
)
documents = loader.load()
import pandas as pd
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore as qd
from langchain.document_loaders import CSVLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Constants
API_TOKEN = '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'
MBD_MODEL = 'intfloat/multilingual-e5-large'
QDRANT_URL = "https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
COLLECTION_NAME = "lenovoHP_collection"
API_KEY = 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'

# Charger les données du CSV avec CSVLoader
loader = CSVLoader(
    file_path='/content/lenovoHP_clean (1).csv',  # Assurez-vous que le chemin du fichier est correct
    metadata_columns=["Marque", "Categorie"],
    csv_args={"delimiter": ","},
    encoding="utf-8"
)
documents = loader.load()

document1 = documents[:5000]
document2=documents[5001:10000]
document3=documents[10001:15000]
document4=documents[15001:20000]
document5=documents[20001:25000]
document6=documents[25001:30000]
document7=documents[30001:35000]
document8=documents[35001:40000]
document9=documents[40001:45000]
document10=documents[45001:50000]
document11=documents[50001:55000]
document12=documents[55001:58877]
import openai
# Configurer la clé API
import os
def load_embedding_function():
    try:
        # Set your OpenAI API key
        openai.api_key = os.getenv('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')
        
        class OpenAIEmbedding:
            def __init__(self, model_name="text-embedding-3-large"):
                self.model_name = model_name

            def embed_query(self, text):
                response = openai.Embedding.create(input=[text], model=self.model_name)
                return response['data'][0]['embedding']

        embedding_function = OpenAIEmbedding()
        return embedding_function
    except Exception as e:
        print(f"Error loading embedding function: {e}")
        return None
embeddings = load_embedding_function()
# Initialisez les embeddings
COLLECTION_NAME = "OpenAI_SELFQ_collection"
vector_params = VectorParams(
    size=1024,  # Remplacez par la taille des vecteurs que vous utilisez
    distance=Distance.COSINE
)
# Connectez-vous à Qdrant et créer ou recréer la collection
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,  # Remplacez par la taille des vecteurs que vous utilisez
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created.")
db = qd.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=API_KEY,
    collection_name=COLLECTION_NAME,
    prefer_grpc=True,
    vector_name=''  # Utilisez un vecteur sans nom
)

# Ajouter les documents à la collection
db.add_documents(document1)
print("Documents ajoutés à la collection Qdrant.")