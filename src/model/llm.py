import time
import os
import re
from dotenv import load_dotenv
from helper_utils import word_wrap
from langchain.chains import LLMChain
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
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

CHROMA_PATH = "../data/chroma"
GROQ_TOKEN = os.getenv('GROQ_TOKEN')
HF_TOKEN = os.getenv('API_TOKEN')

# Prompt template
PROMPT_TEMPLATE = """
Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des infos si elles ne sont pas dans le contexte.
Il faut répondre seulement si tu as la réponse. Accompagne chaque réponse du numéro de pièce, marque, et description du produit tels qu'ils sont dans le contexte.
Affiche autant de lignes que les produits trouvés dans le contexte. Réponds à la question de l'utilisateur en français.
Tu es obligé de répondre dans un tableau avec comme colonnes: Référence, Marque, et la Description.

{context}
Question: {question}
Réponse:"""

# Description of the document's content
document_content_description = (
    "Information about the product, including part number, supplier, "
    "description, brand, quantity, and price."
)

# Metadata fields description for structured query
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
    AttributeInfo(
        name="description",
        description="Detailed description of the product including specifications.",
        type="string",
    ),
]

def get_embedding_function(model_name: str = "intfloat/multilingual-e5-large"):
    """Get Hugging Face embeddings function."""
    return HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name=model_name)

def query_rag(query_text: str) -> str:
    """Queries the retrieval-augmented generation system with a structured query.

    Args:
        query_text (str): The user's input query.

    Returns:
        str: The formatted response from the LLM based on retrieved documents.
    """
    try:
        start_time = time.time()

        # Prepare the embedding function
        embedding_function = get_embedding_function()
        print(f"Embedding function initialized: {embedding_function}")

        # Load the vector database with the appropriate collection
        collection_name = "csv_collection"
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        print("Chroma vector store loaded.")

        # Setup the prompt template for the LLM
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # Initialize the language model with Groq
        llm = ChatGroq(model_name='llama3-8b-8192', api_key=GROQ_TOKEN, temperature=0)
        print("Language model initialized.")

        # Prepare the query constructor prompt
        query_constructor_prompt = get_query_constructor_prompt(
            llm=llm,
            document_contents=document_content_description,
            attribute_info=metadata_field_info,
        )

        # Parse the structured query output
        output_parser = StructuredQueryOutputParser.from_components()

        # Define the query constructor
        query_constructor = query_constructor_prompt | llm | output_parser

        # Initialize SelfQueryRetriever
        retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=db,
            structured_query_translator=ChromaTranslator(),
        )

        # Modify the query text to avoid using unsupported comparators
        # Split query text into different components
        def modify_query_for_supported_comparators(query: str):
            """Modifies query text for compatibility with supported comparators.

            Args:
                query (str): The original query string.

            Returns:
                str: Modified query string.
            """
            # This function should extract and transform the query for supported filters
            # Assuming the input query is structured like: "brand: HP, price: < 1000, description contains: 'i3 8Go RAM'"
            # We can implement a regex-based solution to convert these into valid expressions
            queries = []

            # Example for extracting conditions and transforming them
            brand_match = re.search(r"marque:\s*(\w+)", query)
            if brand_match:
                brand = brand_match.group(1)
                queries.append(f"marque EQ '{brand}'")

            price_match = re.search(r"prix\s*<\s*(\d+)", query)
            if price_match:
                price = price_match.group(1)
                queries.append(f"prix LT {price}")

            description_match = re.search(r"description contient: '([^']+)'", query)
            if description_match:
                # Use EQ instead and make the system robust by checking multiple exact matches
                description_keywords = description_match.group(1).split()
                # Use multiple EQ for different keywords, handling string separately since Chroma doesn't support CONTAIN
                for keyword in description_keywords:
                    queries.append(f"description EQ '{keyword}'")

            return " AND ".join(queries)

        # Modify the user query
        modified_query = modify_query_for_supported_comparators(query_text)

        # Perform retrieval based on the self-querying mechanism
        results = retriever.invoke(modified_query)

        if not results:
            print("No documents found.")
            return "No documents found."

        # Extracting relevant documents and forming the context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        print(f"Context Text: {context_text[:500]}...")

        # Format the prompt with the context and the user's question
        formatted_prompt = prompt_template.format(context=context_text, question=query_text)

        # Use the LLM to generate a response based on the prompt
        response_text = llm(formatted_prompt)
        print(f"Response Text: {response_text}")

        # Parse JSON response from the model output
        try:
            response_json = json.loads(response_text)
            # Assuming the expected output is a JSON object with specific keys
            formatted_response = f"Réponse: {response_json['response']}\nSources: {response_json['sources']}"
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            formatted_response = "Error: Invalid JSON response received from LLM."

        # Extract document sources
        sources = [doc.metadata.get("id", None) for doc in results]
        print(f"Sources: {sources}")

        # Format the response
        formatted_response += f"\nSources: {sources}"
        print(f"Formatted Response: {formatted_response}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for query: {elapsed_time:.2f} seconds")

        return formatted_response

    except Exception as e:
        import traceback
        print(f"An error occurred in query_rag: {e}")
        print(traceback.format_exc())

# Example usage
if __name__ == "__main__":
    query_text = "Trouver un produit de marque HP avec un prix inférieur à 1000 et une description contenant 'i3 8Go RAM'."
    response = query_rag(query_text)
    print(response)
