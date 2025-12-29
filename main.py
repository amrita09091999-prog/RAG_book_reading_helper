from pathlib import Path
import traceback
from rag import RAG
from evaluation import Evaluate

class RAGOrchestration:
    def __init__(self,input_json):
        self.doc_pdf = input_json['doc_pdf']
        self.query = input_json['query']
        filename = Path(self.doc_pdf).name.replace(".pdf","")
        self.bookname = filename   
        self.rag = RAG(self.bookname)
    def get_llm_response(self):
        try:

            vector_store_path = Path("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/vector_store")
            if not (vector_store_path/self.bookname).is_dir():
                print(f"Documents havent been added for this book - {self.bookname}, proceeding to chunk and vectorise stage...\n")
                self.rag.chunk_novel(self.doc_pdf)
                print("Chunking completed\n")
                vector_store = self.rag.initialise_empty_vector_store()
                print("Empty vector store inititalised\n")
                self.rag.add_documents_in_vector_store(vector_store)
                print("Documents added into vector store\n")
            else:
                print("documents already uploaded and chunked, skipping this stage\n")

            print("proceeding for retrieval stage...\n")
            top_docs = self.rag.retrieve(self.query)
            self.rag.format_docs(top_docs)
            print("Relevant documents retrieved\n")
            self.rag.create_prompt()
            print("Prompts created\n")
            self.rag.create_llm_response()
            print("LLM response received and saved in - /Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/rag_response.json")
        except Exception as e:
            print(f"Error in pipeline - {e}")
            traceback.print_exc()
    
    def create_evaluation(self):
        try:
            vector_store_path = Path("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/vector_store")
            if not (vector_store_path/self.bookname).is_dir():
                print("No RAG response to evaluate , please run get_llm_response first")
            else:
                self.evaluate = Evaluate()
                print("RAG evalaution done and saved in - /Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/evaluation/evaluation.txt")
        except Exception as e:
            print(f"Error in pipeline - {e}")
            traceback.print_exc()

input_json = {
    'doc_pdf':'/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/uploaded_pdf/Wheel Of Time - 1The Eye Of The World - PDF Room.pdf',
    'query':'Summarise chapter 10 and explain how is chapter 10 related to chapter 9'
}

orchestra = RAGOrchestration(input_json)
#orchestra.get_llm_response()
orchestra.create_evaluation()

        