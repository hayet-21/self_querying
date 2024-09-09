import os, time
import chainlit as cl

from utils import *
from prompts import *
from constants import *
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

# INITIALISATION

def get_file_ext(filename: str):
    return filename.split('.')[-1].lower()

@cl.on_chat_start
def start():
    memory = ConversationBufferWindowMemory(k=0)
    embeddings = init_embeddings(model_name=MBD_MODEL, api_key=OPENAI_TOKEN)
    llm = init_llm(type='gpt', model_name=GPT, api_key=OPENAI_TOKEN, temperature=0.5)
    llm2 = init_llm(type='gpt', model_name=GPT, api_key=OPENAI_TOKEN, temperature=0.3)
    vectordb = initialize_vectorstore(embeddings, QDRANT_URL, QDRANT_API, COLLECTION_NAME)
    retriever = initialize_retriever(llm, vectordb, metadata_field_info, document_content_description)
    compressor_cohere = init_cohere(COHERE_API_KEY, 10)
    
    # CHAINS
    async def batch_query_bot(questions: list[str]):
        re_rank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor_cohere, base_retriever=retriever
        )
        context = [[]]
        
        # Vérifier si 'questions' est une liste de chaînes
        if not isinstance(questions, list):
            questions = [questions]
        
        print('questions =', questions)
        
        # Vérifier si la liste est vide ou ne contient que des chaînes vides
        if not questions or all(q.strip() == '' for q in questions):
            return "Please provide at least one valid question."
        else:
            context = await re_rank_retriever.abatch(questions)
        
        if not context:
            return "Je n'ai pas trouvé de produits correspondants."
        
       # print('length de context :', sum(len(sublist) for sublist in context))
        liste = get_unique_union(context)
        print(f"Nombre total de liste : {len(liste)}")
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Charger l'historique des conversations
        conversation_history = memory.load_memory_variables({})
        result = await document_chain.ainvoke(
            {
                "context": liste,
                "historique": conversation_history.get('history', ""),
                "question": questions  # Utiliser 'questions' au lieu de 'messages'
            }
        )
        
        # Sauvegarder le contexte
        memory.save_context({"question": questions}, {"response": result})
        
        return result
    
    pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)
    pdf_chain = pdf_prompt | llm2 | StrOutputParser()
    image_chain = image_prompt | llm | StrOutputParser()

    # Session
    cl.user_session.set('llm', llm)
    cl.user_session.set('batch_query_bot', batch_query_bot)
    cl.user_session.set('pdf_chain', pdf_chain)
    cl.user_session.set('image_chain', image_chain)

@cl.on_message
async def chat(message: str):
    # Récupérer les chaînes depuis la session utilisateur
    llm = cl.user_session.get('llm')
    batch_query_bot = cl.user_session.get("batch_query_bot")
    pdf_chain = cl.user_session.get('pdf_chain')
    image_chain = cl.user_session.get('image_chain')

    # Vérifier que batch_query_bot est correctement récupéré
    if not batch_query_bot:
        raise ValueError("batch_query_bot n'est pas défini dans la session utilisateur.")
    
    # Initialiser les variables
    extracted_text = ''
    
    # Traiter les fichiers joints
    if message.elements:
        extracted_text = '\n'.join(
            parse_file(file.path, pdf_chain, image_chain)
            for file in message.elements if get_file_ext(file.name) in FILE_TYPES
        )
    
    # Reformuler les requêtes
    queries = extracted_text.strip().split('\n')
    queries[0] = f"{message.content}: {queries[0]}" 
    
    reformulated_queries = reformulate_queries_with_llm(queries, llm, prompt_template_rech)

    # Appeler 'batch_query_bot' avec les requêtes reformulées et le prompt
    response = await batch_query_bot(reformulated_queries)
    
    # Mettre à jour le message avec les résultats
    await cl.Message(response).send()