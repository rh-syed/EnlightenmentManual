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


def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.00002
    # print(f"Total Tokens: {total_tokens}")
    # print(f"Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}")


def create_embeddings(chunks):
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.run(q)
    return answer
