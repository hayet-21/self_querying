import os
from dotenv import load_dotenv

load_dotenv()

# Chemin absolu
DATA_PATH_CSV = os.path.abspath(f"../{os.getenv('DATA_PATH_CSV')}")

# Vérification et affichage du chemin
print("Chemin absolu du fichier CSV :", DATA_PATH_CSV)

# Vérifier si le fichier existe
if not os.path.exists(DATA_PATH_CSV):
    print("Le fichier CSV spécifié n'existe pas.")
else:
    print("Le fichier CSV spécifié a été trouvé.")
