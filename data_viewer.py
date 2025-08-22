import streamlit as st
import pandas as pd
import chromadb

client = chromadb.PersistentClient()
face_database_collection = client.get_collection(name="face_database")
facial_database = face_database_collection.get()
data = {
    'ids': facial_database['ids'],
    'metadatas':  facial_database['metadatas'],
    'embeddings': facial_database['embeddings']
}
df = pd.DataFrame(data)
st.title('Face Image Vector Database')
st.table(df)
