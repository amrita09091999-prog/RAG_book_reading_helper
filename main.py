from pathlib import Path
import traceback
from rag import RAG
from evaluation import Evaluate

class RAGOrchestration:
    def get_llm_response(self,input_json):
        try:
            doc_pdf = input_json['doc_pdf']
            query = input_json['query']
            filename = Path(doc_pdf).name.replace(".pdf","")
            bookname = filename   
            rag = RAG(bookname)
            vector_store_path = Path("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/chunks_and_vectors/vector_store")
            if not (vector_store_path/bookname).is_dir():
                print(f"Documents havent been added for this book - {bookname}, proceeding to chunk and vectorise stage...\n")
                rag.chunk_novel(doc_pdf)
                print("Chunking completed\n")
                vector_store = rag.initialise_empty_vector_store()
                print("Empty vector store inititalised\n")
                rag.add_documents_in_vector_store(vector_store)
                print("Documents added into vector store\n")
            else:
                print("documents already uploaded and chunked, skipping this stage\n")

            print("proceeding for retrieval stage...\n")
            top_docs = rag.retrieve(query)
            rag.format_docs(top_docs)
            print("Relevant documents retrieved\n")
            rag.create_prompt()
            print("Prompts created\n")
            rag.create_llm_response()
            print("LLM response received and saved in - /Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/rag_response.json")
        except Exception as e:
            print(f"Error in pipeline - {e}")
            traceback.print_exc()
    
    def create_evaluation(self):
        try:
            self.evaluate = Evaluate()
            print("RAG evalaution done and saved in - /Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/evaluation/evaluation.json")
        except Exception as e:
            print(f"Error in pipeline - {e}")
            traceback.print_exc()

# input_json = {
#     'doc_pdf':'/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/uploaded_pdf/Wheel Of Time - 1The Eye Of The World - PDF Room.pdf',
#     'query':'Summarise chapter 10 and explain how is chapter 10 related to chapter 9'
# }

# orchestra = RAGOrchestration(input_json)
# #orchestra.get_llm_response()
# orchestra.create_evaluation()

        