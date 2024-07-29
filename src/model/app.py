from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
from utils_query import query_rag

app = FastAPI()

# Définition d'un modèle Pydantic pour valider les données d'entrée
class QueryRequest(BaseModel):
    query_text: str
    file_type: str

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/query/")
async def handle_query(request: QueryRequest):
    # Validation du type de fichier
    if request.file_type not in ["txt", "csv"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be 'txt' or 'csv'.")
    
    try:
        response_text = query_rag(request.query_text, request.file_type)
        return {"response_text": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("file_type", type=str, choices=["txt", "csv"], help="The type of query file (txt or csv).")
    args = parser.parse_args()
    query_text = args.query_text
    file_type = args.file_type
    response_text = query_rag(query_text, file_type)
    print(response_text)
