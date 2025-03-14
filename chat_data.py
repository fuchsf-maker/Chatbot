import streamlit as st
import pandas as pd
import faiss
import numpy as np
import openai
import os

# OpenAI API Key aus den Umgebungsvariablen
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.sidebar.warning("Bitte API Key in .env oder Streamlit Secrets speichern!")
else:
    openai.api_key = api_key

st.title("DNB-Datenset-Suche mit KI")

# Datei-Upload
uploaded_file = st.sidebar.file_uploader(
    "Excel-Datei hochladen", 
    type=["xlsx"],
    help="Nur Excel-Dateien (.xlsx) mit der korrekten Struktur"
)

if uploaded_file:
    @st.cache_data
    def load_data(file):
        """Lädt Excel-Datei und erstellt Volltextindex"""
        try:
            df = pd.read_excel(file, header=0)
            df.fillna("", inplace=True)
            df["Volltext"] = df.apply(lambda row: " | ".join(row.astype(str)), axis=1)
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")
            return None

    df = load_data(uploaded_file)

    if df is not None:
        st.success(f"Erfolgreich geladen: {len(df)} Zeilen")
        
# OpenAI Embeddings für den Index erstellen
@st.cache_resource
def build_vector_index(df):
    """Erstellt einen FAISS-Vektorspeicher mit OpenAI-Embeddings"""
    texts = df["Volltext"].tolist()
    embeddings_response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Hier das richtige Modell verwenden
        input=texts
    )
    
    embeddings = embeddings_response['data']
    vectors = np.array([embedding['embedding'] for embedding in embeddings]).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

        index, vectors = build_vector_index(df)

        # Suchabfrage
        user_query = st.text_input("Frage stellen (z.B. 'Welche Hochschulschriften habt ihr?'):")

        if user_query:
            # Vektor für die Frage erzeugen
            query_vector = openai.embeddings.create(
                model="text-embedding-ada-002", 
                input=[user_query]
            )["data"][0]["embedding"]

            # Ähnlichste Einträge suchen
            D, I = index.search(np.array([query_vector]).astype("float32"), k=5)
            results = df.iloc[I[0]]

            # Treffer anzeigen
            st.subheader("Relevante Datensätze:")
            st.write(results[["datensetname", "datenformat", "kategorie 1", "kategorie 2"]])

            # ChatGPT für Antwort nutzen
            context = results.to_string(index=False)
            prompt = f"Basierend auf diesen Datensätzen: {context}\n\n{user_query}"

            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("ChatGPT Antwort:")
            st.write(response["choices"][0]["message"]["content"])
