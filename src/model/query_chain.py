from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import time

# CONSTS
API_TOKEN='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'
GROQ_TOKEN='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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000
RAW_DATA_PATH= '../data/data_base.csv'
FEATURES= ['part', 'marque', 'description']
DATA_PATH_CSV= 'data/first_1000_lines.csv'
DATA_PATH_TXT= 'data/sample_db.txt'
COLLECTION_TXT= 'txt_collection'
COLLECTION_CSV= 'csv_collection'
MBD_MODEL= 'intfloat/multilingual-e5-large'

t1= time.time()
persist_directory = CHROMA_PATH
# vectordb.delete_collection()
embedding = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL, )
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=COLLECTION_CSV)

#initiate model
llm = ChatGroq(model_name='llama3-8b-8192', api_key= GROQ_TOKEN,temperature=0)

# Build prompt
template = """tu es un assistant vendeur, tu as acces au context seulement. ne generes pas des infos si ell ne sont pas dans le context il faut repondre seulement si tu as la reponse. accompagne chaque reponse du part, marque et description du produit tel qu'ils snont dans le context. affiche autant de lignes que les produit trouve dans le context. repond a la question de l'utilisateur en francais. tu est obliger de repondre dans un tableau avec comme colonnes: reference, marque et la description
{context}
Question: {question}
Reponse:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Build chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k':10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    verbose= True
)

# Run chain: 
question = "Je veux u pc de 8gb de ram i7"
result = qa_chain.invoke({"query": question})
print('THE CHAIN RESULTS: ')
print(result["result"])
# res= vectordb.similarity_search(question, k=10)
print(f"THE 5 MMR RESULTS:\n{'\n--------------\n'.join([doc.page_content for doc in vectordb.max_marginal_relevance_search(question, k= 20, fetch_k= 5)])}\n")
print(f"THE 5 SS RESULTS:\n{'\n--------------\n'.join([doc.page_content for doc in vectordb.similarity_search(question, k=5)])}")
print(f'response in only: {time.time()- t1: .2f}')
# print(vectordb._collection.count())