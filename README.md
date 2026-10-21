# RAG con DeepSeek-R1 y FAISS

Chat que responde preguntas sobre un PDF mediante Retrieval-Augmented Generation (RAG).
Tarea individual de **IA Aplicada (ESIC)**, basada en el artículo de Annie Wellington
*Developing RAG Systems with DeepSeek R1 & Ollama*.

**Demo online:** https://TU-APP.streamlit.app

## Cómo funciona

PDF → PDFPlumber → SemanticChunker → embeddings Hugging Face → FAISS (top 3 chunks) → DeepSeek-R1 → respuesta

| Dónde se ejecuta | Modelo |
|---|---|
| Ordenador local | `deepseek-r1:1.5b` con Ollama |
| Streamlit Cloud | `deepseek-reasoner` (DeepSeek-R1) vía API |

## Ejecutar en local

```bash
ollama run deepseek-r1:1.5b
py -m pip install -r requirements.txt
py -m streamlit run app.py
```
