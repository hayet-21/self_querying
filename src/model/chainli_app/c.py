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

# When the chat starts, prompt the user to pick an action
@cl.on_chat_start
def start():
    async def ask_user():
        return await cl.AskActionMessage(
            content="Pick an action!",
            actions=[
                cl.Action(name="simple_search", value="simple_search", label="🔍 Recherche"),
                cl.Action(name="similarity_search", value="similarity_search", label="🔎 Produits Similaires"),
                cl.Action(name="job_products", value="job_products", label="🔎 produits selon un metier"),
            ],
        ).send()

    memory = ConversationBufferWindowMemory(k=0)
    embeddings = init_embeddings(model_name=MBD_MODEL, api_key=OPENAI_TOKEN)
    llm = init_llm(type='gpt', model_name=GPT, api_key=OPENAI_TOKEN, temperature=0.5)
    llm2 = init_llm(type='gpt', model_name=GPT, api_key=OPENAI_TOKEN, temperature=0.3)
    vectordb = initialize_vectorstore(embeddings, QDRANT_URL, QDRANT_API, COLLECTION_NAME)
    retriever = initialize_retriever(llm, vectordb, metadata_field_info, document_content_description)
    compressor_cohere = init_cohere(COHERE_API_KEY, 10)
    
    # CHAINS
    async def batch_query_bot(retriever, questions: list[str] | str, prompt):
        re_rank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor_cohere, base_retriever=retriever
        )
        context = [[]]
        
        if not isinstance(questions, list):
            questions = [questions]
        
        if not questions or all(q.strip() == '' for q in questions):
            yield "Please provide at least one valid question."
            return
        
        context = await retriever.abatch(questions)
        
        if not context:
            yield "Je n'ai pas trouvé de produits correspondants."
            return
        
        liste = get_unique_union(context)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        conversation_history = memory.load_memory_variables({})
        result = await document_chain.ainvoke(
            {
                "context": liste,
                "historique": conversation_history.get('history', ""),
                "question": questions
            }
        )
        
        memory.save_context({"question": questions}, {"response": result})
        
        # Instead of returning the result, yield tokens
        for token in result:
            yield token

    
    pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)
    pdf_chain = pdf_prompt | llm2 | StrOutputParser()
    image_chain = image_prompt | llm | StrOutputParser()

    # Session
    cl.user_session.set('llm', llm)
    cl.user_session.set('batch_query_bot', batch_query_bot)
    cl.user_session.set('pdf_chain', pdf_chain)
    cl.user_session.set('image_chain', image_chain)
    cl.user_session.set('retriever', retriever)
    cl.user_session.set('ask_user',ask_user)

# Handle user messages based on selected functionality
@cl.on_message
async def chat(message: str):
    # Retrieve chains from the user session
    llm = cl.user_session.get('llm')
    batch_query_bot = cl.user_session.get('batch_query_bot')
    pdf_chain = cl.user_session.get('pdf_chain')
    image_chain = cl.user_session.get('image_chain')
    retriever = cl.user_session.get('retriever')
    ask_user=cl.user_session.get('ask_user')

    # If batch_query_bot is not found, re-initialize it
    if not batch_query_bot:
        msg = cl.Message(content="batch_query_bot is not set, re-initializing...")
        await msg.send()
        # Set it back in the session
        cl.user_session.set('batch_query_bot', batch_query_bot)

    extracted_text = ''

    # Process any attached files
    if message.elements:
        extracted_text = '\n'.join(
            parse_file(file.path, pdf_chain, image_chain)
            for file in message.elements if get_file_ext(file.name) in FILE_TYPES
        )

    usr_res= await ask_user()

    # Handle different functionalities based on user selection
    if usr_res and usr_res.get("value") == "simple_search":
        queries = extracted_text.strip().split('\n')
        queries[0] = f"{message.content}: {queries[0]}"
        msg = cl.Message(content="")
        await msg.send()

        reformulated_queries = reformulate_queries_with_llm(queries, llm, prompt_template_rech)
        async for token in batch_query_bot(retriever, reformulated_queries, prompt):
            await msg.stream_token(token)

    elif usr_res and usr_res.get("value") == "similarity_search":
        queries = extracted_text.strip().split('\n')
        queries[0] = f"{message.content}: {queries[0]}"
        msg = cl.Message(content="")
        await msg.send()

        reformulated_queries = reformulate_queries_with_llm(queries, llm, prompt_template)
        async for token in batch_query_bot(retriever, reformulated_queries, prompt_similarite):
            await msg.stream_token(token)

    elif usr_res and usr_res.get("value") == "job_products":
        queries = extracted_text.strip().split('\n')
        queries[0] = f"{message.content}: {queries[0]}"
        msg = cl.Message(content="")
        await msg.send()

        reformulated_queries = reformulate_queries_with_llm(queries, llm, prompt_template_metier)
        async for token in batch_query_bot(retriever, reformulated_queries, prompt_similarite):
            await msg.stream_token(token)

    else:
        response = "Unknown functionality selected."
        await msg.stream_token(response)

    msg = cl.Message(content="")
    await msg.send()