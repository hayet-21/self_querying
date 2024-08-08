import pandas as pd
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore as qd
from langchain.document_loaders import CSVLoader
from qdrant_client.models import Distance, VectorParams

# Constants
API_TOKEN = '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'
MBD_MODEL = 'intfloat/multilingual-e5-large'
QDRANT_URL = "https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
COLLECTION_NAME = "icecat_collection"
API_KEY = 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'

# Charger les données du CSV avec CSVLoader
loader = CSVLoader(
    file_path='../data/icecat.csv',
    metadata_columns=["Marque", "Categorie"],
    csv_args={"delimiter": ","},
    encoding="utf-8"
)
documents = loader.load()

# Initialisez les embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
# Définir les paramètres de la collection
vector_params = VectorParams(
    size=1024,  # Remplacez par la taille des vecteurs que vous utilisez
    distance=Distance.COSINE
)
db = qd.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=API_KEY,
    collection_name=COLLECTION_NAME,
    prefer_grpc=True,
    vector_name='vector_params'  # Utilisez le nom du vecteur dense configuré
)

# Ajouter les documents à la collection
db.add_documents(documents)
print("Documents ajoutés à la collection Qdrant.")