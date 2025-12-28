import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import os
import json
from dotenv import load_dotenv

class RAG:
    def __init__(self,book_name):
        load_dotenv()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs = {'normalize_embeddings':True}
        )
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            task="text-generation",
            temperature = 0.2
            )
        self.generation_model = ChatHuggingFace(llm=llm)
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-14B-GGUF",
            task="text-generation",
            temperature = 0.1
            )
        self.judge_model = ChatHuggingFace(llm=llm)
        self.chunks_dir = "/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/chunk_store"
        self.vector_store = "/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/vector_store"
        self.book_name = book_name
        os.makedirs("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",exist_ok =True)
        rag_response = {
            'query':None,
            'retrieved_context':None,
            'response':None
        }
        if not os.path.exists(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json')):
            with open(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json'),'w') as f:
                json.dump(rag_response,f)
    
    def chunk_novel(self,doc_pdf):
        loader = PyPDFLoader(f"/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/uploaded_pdf/{doc_pdf}")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,       
        chunk_overlap=200,  
        separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        chapter_number = 0
        chapter_name = "prefactual"
        for chunk in chunks:
            if 'chapter' in chunk.page_content.lower():
                splited = chunk.page_content.split("\n")
                chapter_name = splited[2]
                chapter_number = splited[1]
            
            chunk.metadata['chapter_number'] = chapter_number
            chunk.metadata['chapter_name'] = chapter_name
        os.makedirs(self.chunks_dir, exist_ok=True)
        book_name = doc_pdf.replace(".pdf","")
        os.makedirs(os.path.join(self.chunks_dir,self.book_name),exist_ok = True)
        with open(os.path.join(self.chunks_dir,self.book_name,'chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)
    
    def initialise_empty_vector_store(self):
        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("hello world")))
        vector_store = FAISS(
        embedding_function=self.embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        )
        return vector_store
    
    def add_documents_in_vector_store(self,vector_store):
        with open(f"{self.chunks_dir}/{self.book_name}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        vector_store.add_documents(chunks)
        os.makedirs(self.vector_store,exist_ok = True)
        os.makedirs(os.path.join(self.vector_store,self.book_name),exist_ok = True)
        vector_store.save_local(os.path.join(self.vector_store,self.book_name,'faiss_index'))
    
    def retrieve(self, query):
        with open(f"{self.chunks_dir}/{self.book_name}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        
        vector_store = FAISS.load_local(
        os.path.join(self.vector_store,self.book_name,'faiss_index'),
        self.embedding_model,
        allow_dangerous_deserialization=True
        )

        bm25_retriever = BM25Retriever.from_documents(
        documents=chunks,  
        k= 10)

        semantic_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":40}
        )

        bm25_docs = bm25_retriever.invoke(query)
        faiss_docs = semantic_retriever.invoke(query)
        
        combined_docs = bm25_docs + faiss_docs
        unique_docs_dict = {id(doc): doc for doc in combined_docs}
        deduped_docs = list(unique_docs_dict.values())
        pairs = [[query, doc.page_content] for doc in deduped_docs]
        scores = self.cross_encoder.predict(pairs)
        top_docs = [doc for _, doc in sorted(zip(scores, deduped_docs), key=lambda x: x[0], reverse=True)][:20]
    
        with open(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json'),'r') as f:
            rag_response = json.load(f)
        rag_response['query']= query
        with open(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json'),'w') as f:
            json.dump(rag_response,f)
        return top_docs 
    
    def format_docs(self, top_docs):
        doc_info = []

        for i, doc in enumerate(top_docs, start=1):
            formatted_doc = (
                f"Document {i}\n"
                f"Chapter Number: {doc.metadata.get('chapter_number', 'N/A')}\n"
                f"Chapter Name: {doc.metadata.get('chapter_name', 'N/A')}\n"
                f"Page Number: {doc.metadata.get('page', 'N/A')}\n\n"
                f"Content:\n{doc.page_content.strip()}"
            )

            doc_info.append(formatted_doc)

        formatted_docs  = "\n\n---\n\n".join(doc_info)

        with open(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json'),'r') as f:
            rag_response = json.load(f)

        rag_response['retrieved_context'] = formatted_docs

        with open(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json'),'w') as f:
            json.dump(rag_response,f)
    
    def create_prompt(self):
        with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/prompts/rag_response_prompt_template.txt",'r',encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = PromptTemplate(
            template = prompt_template,
            input_variables = ['context','query']
        )

        with open(os.path.join("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle",'rag_response.json'),'r') as f:
            rag_response = json.load(f)
        
        context = rag_response['retrieved_context']
        query = rag_response['query']
        if query is None:
            query = "who is rand al thor?"

        formatted_prompt = prompt.format(
            context = context,
            query = query
        )
        with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/prompts/rag_response_prompt.txt", "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
    
    def create_llm_response(self):
        with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/prompts/rag_response_prompt.txt", "r", encoding="utf-8") as f:
            input_prompt = f.read()
        
        with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle/rag_response.json",'r') as f:
            rag_response = json.load(f)

        response = self.generation_model.invoke(input_prompt)
        rag_response['response'] = response.content

        with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/rag_response_bundle/rag_response.json",'w') as f:
            json.dump(rag_response,f)





        

    

    





