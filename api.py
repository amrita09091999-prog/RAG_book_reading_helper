from fastapi import FastAPI,Path,HTTPException, Query, File, UploadFile,HTTPException
import numpy as np
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel, EmailStr, AnyUrl,Field, field_validator,model_validator,computed_field,conint
from typing import Any, List, Dict,Optional, Annotated
import pickle
import pandas as pd
import os
from main import RAGOrchestration

app = FastAPI()

UPLOAD_DIR = "/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/uploaded_pdf"
NOVEL_PATH  = None
QUERY = None
os.makedirs(UPLOAD_DIR,exist_ok = True)

class QuestionRequest(BaseModel):
    query : str

@app.get("/")
def intro():
    return {
        "Introduction Message":"Hey Reader! This is your Novel Reading Assistant, please go to docs to upload and ask questions related to the current book youre reading!"
    }
@app.post("/novel_upload")
async def upload_file(file: UploadFile = File(...)):
    global NOVEL_PATH

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid content type")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(await file.read())

        NOVEL_PATH = file_path

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "File uploaded successfully"
        }

    NOVEL_PATH = file_path
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File already exists..proceed to ask questions please"
    }


@app.post("/ask_questions")
async def ask_questions(request:QuestionRequest):
    global QUERY
    if not NOVEL_PATH:
        raise HTTPException(
            status = 400,
            detail = "Please first upload the novel - use /upload_file endpoint"
        )
    else:
        input_json = {
            'doc_pdf':NOVEL_PATH,
            'query':request.query
        }
        QUERY = request.query
        rag_orchestra = RAGOrchestration(input_json)
        rag_orchestra.get_llm_response()

        with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/rag_response.json",'r') as f:
            llm_response = json.load(f)

            return JSONResponse(
                status_code = 200,
                content = {"User Query":request.query,
                "AI Respone":llm_response['response']
            })

@app.post("/evaluate")
async def evaluate():
    if not os.path.exists("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/rag_response.json"):
        raise HTTPException(
            status_code=400,
            detail="No query or novel found for evaluation"
        )
    input_json = {
            'doc_pdf':NOVEL_PATH,
            'query':QUERY
        }
    rag_orchestra = RAGOrchestration(input_json)
    rag_orchestra.create_evaluation()
    with open ("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/evaluation/evaluation.json",'r') as f:
        evaluated_doc = json.load(f)
    return JSONResponse(
        status_code= 200,
        content = {
            'Evaluation':evaluated_doc
        }
    )




