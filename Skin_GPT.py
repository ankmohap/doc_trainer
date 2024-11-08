import streamlit as st
import time
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time
from constants import CHROMA_SETTINGS
import random
model = os.environ.get("MODEL", "llama3")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
thread = random.random()

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
callbacks = [StreamingStdOutCallbackHandler()]
llm = Ollama(model=model, callbacks=callbacks)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True )

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

st.set_page_config(page_title="SkinGPT", page_icon=":speech_balloon:")
st.sidebar.title("Skin GPT")
st.sidebar.markdown(
"""
Self-service platform to
- Answer all your queries on skin-care
- Get suggestions
    - Skin care routines 
    - Diet advice
    - Do's and Dont's 
- Check out latest market products 
- Latest AI Guidelines and Compliance queries 
"""
)

if st.sidebar.button("Start Chat"):
    st.session_state.start_chat = True
    st.session_state.thread_id = thread

st.title("Skin GPT")

if st.button("Exit Chat"):
    st.session_state.messages = []  # Clear the chat history
    st.session_state.start_chat = False  # Reset the chat state
    st.session_state.thread_id = None

if st.session_state.start_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Please enter your query here?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        #Display assistant response in chat message container 
        with st.chat_message("assistant"):
            my_spinner= st.spinner("Thinking")
            with my_spinner:
                    # it is printing response again after a new prompt when using spinner()
                    response = response=qa(prompt)
                    st.write(response['result'], response['source_documents'])

        # Add assistant response to chat histpry 
        st.session_state.messages.append({"role": "assistant", "content": "response"})
else:
    st.write("Click 'Start Chat' to begin.")
