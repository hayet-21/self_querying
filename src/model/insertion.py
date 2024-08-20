from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant 
import qdrant_client
from langchain_qdrant import QdrantVectorStore as qd
from langchain.document_loaders import CSVLoader
from qdrant_client.models import Distance, VectorParams

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openAi8key)
COLLECTION_NAME = "OpenAI_SELFQ_collection"
QDRANT_URL="https://31c7da1f-d357-4163-9da5-51008923630d.us-east4-0.gcp.cloud.qdrant.io:6333"
API_KEY ="aWrMYkukRIX9UdHZqhxWsWGuWM9RstNh1temmPB3OFt0inPCrf3vQw"
# Charger les données du CSV avec CSVLoader
loader = CSVLoader(
    file_path='../data/lenovoHP_clean (1).csv',
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

client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=API_KEY)
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,  # Remplacez par la taille des vecteurs que vous utilisez
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