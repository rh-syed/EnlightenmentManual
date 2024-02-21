import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def load_documents(file):
    name, extension = os.path.splitext(file)
    print(f"Loading {file}")
    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file)
    else:
        print("Document format is not supported!")
        return None

    data = loader.load()
    return data


def load_From_WikiPedia(query, lang="en", load_max_dox=2):
    from langchain_community.document_loaders import WikipediaLoader

    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_dox)
    data = loader.load()
    return data


def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def print_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total Tokens: {total_tokens}")
    print(f"Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}")


def insert_or_fetch_embeddings(index_name, chunks):
    from pinecone import PodSpec, Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone = Pinecone(api_key=pinecone_key)

    if index_name in pinecone.list_indexes():
        print(f"Index {index_name} already exists. Loading embeddings ... ", end="")
        vector_store = Pinecone(index_name, embeddings)
        print("Ok")
    else:
        print(f"Creating index {index_name} and embeddings ...", end="")
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )
        print("Ok")

    return vector_store


def delete_pinecone_index(index_name="all"):
    import pinecone

    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"))

    if index_name == "all":
        indexes = pinecone.list_indexes()
        print("Deleting all indexes...")
        for index in indexes:
            pinecone.delete_index(index)
            print("Indexes Deleted!")
    else:
        print(f"Deleting index {index_name}...", end="")
        pinecone.delete_index(index_name)
        print(f"{index_name} deleted!")


def ask_and_get_answer(vectore_store, question):
    from langchain.chains import RetrievalQA
    from langchain_community.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperatures=1)

    retreiver = vectore_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retreiver=retreiver
    )

    answer = chain.run(question)
    return answer


def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))

    return result, chat_history
