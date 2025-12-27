import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAG:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs = {'normalize_embeddings':True}
        )
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
        os.makedirs(os.path.join(self.chunks_dir,book_name),exist_ok = True)
        with open(os.path.join(self.chunks_dir,book_name,'chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)

