"""
RAG con DeepSeek-R1 y FAISS
Tarea individual - IA Aplicada (ESIC)
Basado en: Annie Wellington, "Developing RAG Systems with DeepSeek R1 & Ollama"

EJECUCION EN LOCAL (con Ollama)
  1. Ollama instalado y abierto, con el modelo descargado:
       ollama run deepseek-r1:1.5b
  2. Librerias de Python:
       py -m pip install streamlit langchain langchain-community langchain-experimental langchain-huggingface pdfplumber faiss-cpu sentence-transformers torchvision
  3. Ejecutar la aplicacion:
       py -m streamlit run app.py

VERSION ONLINE (Streamlit Community Cloud)
  https://TU-APP.streamlit.app
  En la nube no existe Ollama, por eso alli DeepSeek-R1 se usa a traves de su API
  (modelo deepseek-v4-flash, sucesor de deepseek-reasoner; clave DEEPSEEK_API_KEY en los Secrets de Streamlit Cloud).
"""

# 1. Import Libraries
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("RAG con DeepSeek-R1 y FAISS")

# 2. Upload & Process PDFs
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load PDF text
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # 3. Chunking: split text into semantic chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # 4. Create a Searchable Knowledge Base
    # Generate embeddings and store them in a FAISS index
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Connect retriever (fetch top 3 chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    st.success("¡PDF procesado e índice FAISS creado con éxito!")

    # 5. Configure DeepSeek R1
    # AÑADIDO para Streamlit Cloud: si existe la clave de la API se usa DeepSeek-R1 en la nube;
    # si no (ordenador local), se usa Ollama exactamente como en el articulo.
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        api_key = None

    if api_key:
        from langchain_openai import ChatOpenAI  # la API de DeepSeek es compatible con OpenAI
        llm = ChatOpenAI(
            model="deepseek-v4-flash",  # sucesor de deepseek-reasoner (R1), razonamiento activado por defecto
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    else:
        llm = Ollama(model="deepseek-r1:1.5b")  # Our 1.5B parameter model

    # Craft the prompt template
    prompt = """
    1. Use ONLY the context below.
    2. If unsure, say "I don't know".
    3. Keep answers under 4 sentences.

    Context: {context}

    Question: {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # 6. Assemble the RAG Chain
    # Combine document chunks
    document_prompt = PromptTemplate(
        template="Context:\ncontent:{page_content}\nsource:{source}",
        input_variables=["page_content", "source"],
    )

    def format_docs(docs):
        return "\n\n".join(
            document_prompt.format(page_content=d.page_content, source=d.metadata["source"])
            for d in docs
        )

    # Final RAG pipeline
    qa = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )

    # 7. Launch the Web Interface (Streamlit UI)
    user_input = st.text_input("Ask your PDF a question:")

    if user_input:
        with st.spinner("Thinking..."):
            response = qa.invoke(user_input)
            st.write(response)
