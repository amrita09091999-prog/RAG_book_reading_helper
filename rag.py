import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import os

class RAG:
    def __init__(self,book_name):
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
        self.chunks_dir = "/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/chunk_store"
        self.vector_store = "/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/vector_store"
        self.book_name = book_name
    
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
        k= 25)

        semantic_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":25}
        )

        bm25_docs = bm25_retriever.invoke(query)
        faiss_docs = semantic_retriever.invoke(query)
        
        combined_docs = bm25_docs + faiss_docs
        unique_docs_dict = {id(doc): doc for doc in combined_docs}
        deduped_docs = list(unique_docs_dict.values())
        pairs = [[query, doc.page_content] for doc in deduped_docs]
        scores = self.cross_encoder.predict(pairs)
        top_docs = [doc for _, doc in sorted(zip(scores, deduped_docs), key=lambda x: x[0], reverse=True)][:15]
        return top_docs 
    
    





