import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import SystemMessage
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



st.title('Simple Chatbot')

if 'store' not in st.session_state:
        st.session_state.store = {}

session_id = st.text_input("Session ID", value="default_session")


with st.sidebar:
    st.subheader('Adjust params!')
    temperature = st.slider('Temperature', 0.0, 2.0, value=0.2, step=0.1, format='%0.1f')
    max_tokens = st.slider('Max Tokens', 10, 500, value=300, step=1)


llm = ChatGroq(model="llama-3.1-8b-instant", 
                       temperature=temperature, 
                       max_tokens=max_tokens)

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
summarize_button = st.button('Summarize')

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


if summarize_button and uploaded_file:
    def load_docs(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        return docs
        
    docs = load_docs(uploaded_file)

    def split_docs(docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)
    
    chunks = split_docs(docs)

    
    chain = load_summarize_chain(llm, chain_type='map_reduce')

    callback = StreamlitCallbackHandler(st.container())
    summary = chain.invoke(chunks, config={'callbacks': [callback]})

    get_session_history(session_id)
    history = st.session_state.store[session_id]
    system_message = f"""Human had asked to summarize a pdf file.
    You summarized it. Its summary was {summary['output_text']}
    """
    
    history.add_message(SystemMessage(content=system_message))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_db = FAISS.from_documents(chunks, embeddings)
    faiss_db.save_local("faiss_doc_db")

    st.subheader("🧾 Summary")
    st.success(summary['output_text'])

    faiss_db = FAISS.load_local("faiss_doc_db",embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_db.as_retriever(search_kwargs={'k':3})
    



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

runnable = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)



if query:= st.chat_input('Ask me anything!'):
    response = runnable.invoke(
        {"input":query},
        config={"configurable":{"session_id":session_id}}
    )
    st.chat_message("assistant").write(response.content)
    
    # for msg in st.session_state.store[session_id].messages:
    #     st.success(f"{msg.type}: {msg.content}")

