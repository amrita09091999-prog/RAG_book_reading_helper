class RAG:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs = {'normalize_embeddings':True}
        )
        