import documentreader_pinecone as reader

data = reader.load_documents("files/Enlightenment-Manual.pdf")
chunks = reader.chunk_data(data)

index_name = "enlightenment-manual"

vector_store = reader.insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)
