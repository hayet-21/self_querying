import csv
from typing import List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
from insert_data import insert_chroma

# Charger les variables d'environnement
load_dotenv()

# Définir le chemin vers le fichier CSV à partir des variables d'environnement
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")

class CSVLoader(BaseLoader):
    """Charge un fichier CSV dans une liste de documents.

    Chaque document représente une ligne du fichier CSV. Chaque ligne est convertie en
    une paire clé/valeur et sortie sur une nouvelle ligne dans le contenu de la page du document.

    La source de chaque document chargé depuis le CSV est définie par défaut comme la valeur de 
    l'argument file_path pour tous les documents. Vous pouvez remplacer cela en définissant l'argument 
    source_column comme le nom d'une colonne dans le fichier CSV. La source de chaque document sera alors 
    définie comme la valeur de la colonne dont le nom est spécifié dans source_column.

    Exemple de sortie:
        .. code-block:: txt

            Ref_produit: value1
            Marque: value2
            Categorie: value3
            Description: value4
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        csv_args: Optional[dict] = None,
        encoding: Optional[str] = None,
    ):
        # Initialiser les paramètres
        self.file_path = file_path
        self.source_column = source_column
        self.csv_args = csv_args or {}
        self.encoding = encoding or "utf-8"

    def load(self) -> List[Document]:
        """Charge les données dans des objets document."""
        
        docs = []  # Liste pour stocker les documents
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            # Utiliser DictReader pour lire les lignes en tant que dictionnaires
            csv_reader = csv.DictReader(csvfile, **self.csv_args)
            for i, row in enumerate(csv_reader):
                # Vérifier si les colonnes requises sont présentes
                try:
                    ref_produit = row["Ref_produit"].strip()
                    marque = row["Marque"].strip()
                    categorie = row["Categorie"].strip()
                    description = row["Description"].strip()
                except KeyError as e:
                    raise ValueError(f"Colonne attendue manquante: {e}")
                
                # Préparer le contenu du document
                content = (
                    f"Ref_produit: {ref_produit}\n"
                    f"Description: {description}"
                )
                
                # Définir la source du document
                source = self.file_path
                
                # Créer des métadonnées avec seulement la marque, la catégorie, la source, et la ligne
                metadata = {
                    "Marque": marque,
                    "Categorie": categorie,
                    "source": source,
                    "row": i + 1  # Numéro de ligne du produit
                }
                
                # Créer et ajouter le Document
                doc = Document(page_content=content, metadata=metadata, source=source)
                docs.append(doc)

        return docs

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
