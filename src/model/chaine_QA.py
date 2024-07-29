from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API Hugging Face
HF_TOKEN = os.getenv('API_TOKEN')

# Description du contenu du document
document_content_description = "Information about the product, including part number, supplier, description, brand, quantity, and price. Do not use the word 'contains' or 'contain' as filters. "

# Métadonnées des attributs
metadata_field_info = [
    AttributeInfo(
        name="part",
        description="The part number of the product.",
        type="string",
    ),
    AttributeInfo(
        name="quantity",
        description="The quantity available of the product.",
        type="integer",
    ),
    AttributeInfo(
        name="description",
        description="The description of the product.",
        type="string",
    ),
    AttributeInfo(
        name="fournisseur",
        description="The supplier of the product.",
        type="string",
    ),
    AttributeInfo(
        name="prix", 
        description="The price of the product.",
        type="float",
    ),
    AttributeInfo(
        name="marque",
        description="The brand of the product.",
        type="string",
    ),
]

 # Define allowed comparators list
allowed_comparators = [
    "$eq",  # Equal to (number, string, boolean)
    "$ne",  # Not equal to (number, string, boolean)
    "$gt",  # Greater than (number)
    "$gte",  # Greater than or equal to (number)
    "$lt",  # Less than (number)
    "$lte",  # Less than or equal to (number)
]

examples = [
    (
        "Recommend some products of brand hp i7 8go de ram.",
        {
            "query": "products i7 8go de ram",
            "filter": 'and(eq("brand", "hp") ,in("description", "i7 8go de ram"))',
        },
    ),
    (
        "Show me products with at least 16GB of RAM.",
        {
            "query": "products 16GB RAM",
            "filter": 'gte("description", "16GB RAM")',
        },
    ),
    (
        "Find me the latest model of Dell laptops with a price below 1000 euros.",
        {
            "query": "latest Dell laptops below 1000 euros",
            "filter": 'and(and(eq("brand", "Dell") , lte("prix", 1000)) , eq("description", "laptop"))',
        },
    ),
    (
        "List all available products from Asus with at least 500 units in stock.",
        {
            "query": "Asus products with stock",
            "filter": 'and(eq("brand", "Asus") , gte("quantity", 500))',
        },
    ),
    (
        "Show me all Samsung monitors priced between 150 and 300 euros.",
        {
            "query": "Samsung monitors priced between 150 and 300 euros",
            "filter": 'and(and(and(eq("brand", "Samsung") , gte("prix", 150)) , lte("prix", 300)) , eq("description", "monitor"))',
        },
    ),
    (
        "Recommend high-performance laptops with at least 32GB of RAM and a price above 2000 euros.",
        {
            "query": "high-performance laptops with 32GB RAM above 2000 euros",
            "filter": 'and(gte("description", "32GB RAM") , gte("prix", 2000))',
        },
    ),
    (
        "Find products from Lenovo with a description mentioning 'SSD' and a quantity less than 50.",
        {
            "query": "Lenovo products with SSD",
            "filter": 'and(and(eq("brand", "Lenovo") , eq("description", "SSD")) , lte("quantity", 50))',
        },
    )
]

GROQ_TOKEN='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'
llm = ChatGroq(model_name='llama3-8b-8192', api_key= GROQ_TOKEN,temperature=0)

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

# Initialiser Chroma pour la récupération
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
COLLECTION_CSV = os.getenv('COLLECTION_CSV')

# Charger la collection Chroma
vectorstore = Chroma(persist_directory=CHROMA_PATH, 
                     embedding_function=embedding_function, 
                     collection_name=COLLECTION_CSV)

# Créer le prompt pour le constructeur de requêtes
prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
            allowed_comparators=allowed_comparators,
            examples=examples,
        )


output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser

# Initialiser le SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    enable_limit=True
   
)


# Construire le template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des infos si elles ne sont pas dans le contexte.
            Répond seulement si tu as la réponse. Affiche les produits trouvés dans un tableau avec comme colonnes: Référence, Marque, et Description.
            Ne formate pas le contexte.si le context est vide dis moi que tu nas pas trouve
            je veux que la reponse soit clair
            avec des saut de ligne.
            ne me donne pas des produit qui ne sont pas dans le context
            
            Contexte:
            {context}
            
            Question: {question}
            
            Réponse :
            """
        ),
    ]
)
# Fonction pour formater les documents
def format_docs(docs):
    """Formate chaque document avec un affichage lisible"""
    return "\n\n".join(
        f"{doc.page_content}\n\nMetadata: {doc.metadata}"
        for doc in docs
    )

# Création de la chaîne de traitement
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)

# Chaîne parallèle pour documents et question
rag_chain_with_source = (
    RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
)

# Requête de l'utilisateur
query = "donne moi 2 produits de marque samsung "

# Exécution de la requête
result = rag_chain_with_source.invoke(query)

# Afficher la réponse générée
print(result)

