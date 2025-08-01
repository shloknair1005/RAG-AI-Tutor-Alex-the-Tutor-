from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("rag_python_tutor_dataset.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        topic = row.get("Topic", "").strip()
        explanation = row.get("Explanation", "").strip()
        example = row.get("Example", "").strip() if pd.notnull(row.get("Example")) else ""

        content = f"{topic}:\n{explanation}"

        document = Document(
            page_content=content,
            metadata={"Example": example},
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="Python_tutorial",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
