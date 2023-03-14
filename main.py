import pinecone
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAIChat
from langchain.chains import VectorDBQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=st.secrets["pinecone"]
    environment=st.secrets["env"]
)
index_name = "acc-municode"

@st.cache_resource
acc_pinecone = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings)
query = st.text_input("`Please ask a question about ACC's Code of Ordinances:` ","How many parking spaces does a bowling alley need")
info = " In which section can I find this information?"
qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), acc_pinecone, return_source_documents=True)
result = qa({"question": query + info, "chat_history": ""})
st.info("`%s`"%result['answer'])
