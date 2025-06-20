
import streamlit as st
from langchain.document_loaders import WebBaseLoader, WikipediaLoader, UnstructuredPDFLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title="MYDRATUM", page_icon="🧠", layout="centered")
st.title("🧠 MYDRATUM - IA Aprendiz da Web")

st.sidebar.title("🔧 Fontes de Conhecimento")
source = st.sidebar.selectbox("Escolha a fonte:", ["Website", "Wikipedia", "YouTube", "PDF"])

query = st.text_input("Digite sua pergunta ou tópico:")

# Carregamento de fonte
if source == "Website":
    url = st.text_input("Informe a URL do site:")
    if url and query:
        loader = WebBaseLoader(url)
        docs = loader.load()

elif source == "Wikipedia":
    topic = st.text_input("Tópico da Wikipedia:")
    if topic and query:
        loader = WikipediaLoader(query=topic, lang="pt")
        docs = loader.load()

elif source == "YouTube":
    yt_url = st.text_input("Link do vídeo do YouTube:")
    if yt_url and query:
        loader = YoutubeLoader.from_youtube_url(yt_url)
        docs = loader.load()

elif source == "PDF":
    pdf_file = st.file_uploader("Envie seu PDF", type=["pdf"])
    if pdf_file is not None and query:
        loader = UnstructuredPDFLoader(pdf_file)
        docs = loader.load()

else:
    docs = []

if docs and query:
    with st.spinner("🧠 Processando e aprendendo..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_split = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings()
        db = Chroma.from_documents(docs_split, embeddings)

        results = db.similarity_search(query, k=3)

        st.subheader("🔍 Resposta da IA:")
        for res in results:
            st.write(res.page_content)

st.markdown("---")
st.markdown("🚀 Criado por **MYDRATUM** | IA Aprendiz da Web")
