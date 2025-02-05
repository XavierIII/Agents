import streamlit as st
import asyncio
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_community.document_loaders import SeleniumURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # Persistent storage
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.agents import AgentGraph

# Initialize Embeddings and Vector Store
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = Chroma(embedding_function=embeddings)
model = OllamaLLM(model="llama3.2")

def extract_relevant_content(html):
    """Extract only meaningful content from the HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join([p.text for p in soup.find_all("p")])  # Extracting paragraphs

def load_dynamic_page(url):
    """Load JavaScript-rendered pages with Selenium."""
    driver = webdriver.Chrome()
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "p")))
    html = driver.page_source
    driver.quit()
    return extract_relevant_content(html)

async def async_load_page(url):
    return await asyncio.to_thread(load_dynamic_page, url)

async def async_split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return await asyncio.to_thread(text_splitter.split_documents, documents)

async def async_index_docs(documents):
    texts = [doc.page_content for doc in documents]
    await asyncio.to_thread(vector_store.add_texts, texts)

def retrieve_docs(query):
    retriever = vector_store.as_retriever()
    return retriever.get_relevant_documents(query)

def filter_retrieved_docs(query, docs):
    query_words = set(query.lower().split())
    def relevance_score(doc):
        doc_words = set(doc.page_content.lower().split())
        return len(query_words & doc_words)
    return sorted(docs, key=relevance_score, reverse=True)[:3]  # Return top 3

def format_answer(answer):
    return f"### 📝 Answer:\n\n{answer}"

def get_source_links(docs):
    return "\n\n**Sources:**\n" + "\n".join([f"- [{doc.metadata['source']}]({doc.metadata['source']})" for doc in docs])

# LangGraph Agent Integration
agent_graph = AgentGraph(model)

st.title("AI Multi-Site Crawler")
urls = st.text_area("Enter URLs (one per line):").split("\n")

if st.button("Crawl Websites"):
    with st.spinner("Fetching and indexing webpages..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for url in urls:
            if url.strip():
                documents = loop.run_until_complete(async_load_page(url.strip()))
                chunked_documents = loop.run_until_complete(async_split_text(documents))
                loop.run_until_complete(async_index_docs(chunked_documents))
        
        st.success("Webpages indexed successfully!")

uploaded_file = st.file_uploader("Upload PDF for Analysis", type=["pdf"])
if uploaded_file:
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    asyncio.run(async_index_docs(documents))
    st.success("PDF Indexed!")

question = st.chat_input()
if question:
    st.chat_message("user").write(question)
    retrieved_documents = retrieve_docs(question)
    filtered_documents = filter_retrieved_docs(question, retrieved_documents)
    context = "\n\n".join([doc.page_content for doc in filtered_documents])
    
    # Use LangGraph Agent for answering
    answer = agent_graph.invoke({"question": question, "context": context})
    
    formatted_answer = format_answer(answer)
    st.chat_message("assistant").write(formatted_answer + get_source_links(filtered_documents))
    
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((question, formatted_answer))

    for q, a in st.session_state.history[::-1]:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
