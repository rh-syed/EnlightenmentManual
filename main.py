import os
import streamlit as st

import documentreader_chroma as reader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


# clear the chat history from streamlit session state
def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]


st.subheader("Ask Your Burning Questions About Enlightenment")
with st.sidebar:
    # text_input for the OpenAI API key (alternative to python-dotenv and .env)
    api_key = st.text_input("OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # file uploader widget
    uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])

    # chunk size number widget
    chunk_size = st.number_input(
        "Chunk size:", min_value=100, max_value=2048, value=512, on_change=clear_history
    )

    # k number input widget
    k = st.number_input(
        "k", min_value=1, max_value=20, value=3, on_change=clear_history
    )

    # add data button widget
    add_data = st.button("Add Data", on_click=clear_history)

    if uploaded_file and add_data:  # if the user browsed a file
        with st.spinner("Reading, chunking and embedding file ..."):

            # writing the file from RAM to the current directory on disk
            bytes_data = uploaded_file.read()
            file_name = os.path.join("./", uploaded_file.name)
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            data = reader.load_documents(file_name)
            chunks = reader.chunk_data(data, chunk_size=chunk_size)
            st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

            tokens, embedding_cost = reader.calculate_embedding_cost(chunks)
            st.write(f"Embedding cost: ${embedding_cost:.4f}")

            # creating the embeddings and returning the Chroma vector store
            vector_store = reader.create_embeddings(chunks)

            # saving the vector store in the streamlit session state (to be persistent between reruns)
            st.session_state.vs = vector_store
            st.success("File uploaded, chunked and embedded successfully.")

# user's question text input widget

q = st.text_input("Ask a question about the content of your file:")
if q:  # if the user entered a question and hit enter
    if (
        "vs" in st.session_state
    ):  # if there's the vector store (user uploaded, split and embedded a file)
        vector_store = st.session_state.vs
        st.write(f"k: {k}")
        answer = reader.ask_and_get_answer(vector_store, q, k)

        # text area widget for the LLM answer
        st.text_area("LLM Answer: ", value=answer)

        st.divider()

        # if there's no chat history in the session state, create it
        if "history" not in st.session_state:
            st.session_state.history = ""

        # the current question and answer
        # the current question and answer
        value = f"Q: {q} \nA: {answer}"

        st.session_state.history = (
            f'{value} \n {"-" * 100} \n {st.session_state.history}'
        )
        h = st.session_state.history

        # text area widget for the chat history
        st.text_area(label="Chat History", value=h, key="history", height=400)
