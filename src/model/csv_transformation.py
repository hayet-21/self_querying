import csv
from typing import List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
from insert_data import insert_chroma
from langchain_community.document_loaders import CSVLoader

# Charger les variables d'environnement
load_dotenv()

# Définir le chemin vers le fichier CSV à partir des variables d'environnement
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")

# Exemple d'utilisation
csv_loader = CSVLoader(
    file_path=DATA_PATH_CSV,  # Chemin vers votre fichier CSV
)

# Charger les documents depuis le fichier CSV
documents = csv_loader.load()

# Afficher les documents chargés
for doc in documents:
    print("Page Content:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 80)

# Insérer les documents dans Chroma
insert_chroma(documents)

