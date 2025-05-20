import streamlit as st
import pandas as pd
from pathlib import Path
from agent import synthesize_answer
from utils import load_transcripts
from data_model import create_db_and_tables, add_response, get_session
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

DB_PATH = "data/database.db"
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
create_db_and_tables(DB_PATH)

st.sidebar.title("Survey Agent MVP")

# 1 — Upload transcripts
uploaded_file = st.sidebar.file_uploader(
    "Upload transcripts CSV/Parquet", type=["csv", "parquet"]
)
if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    transcripts_df = load_transcripts(file_path)
    st.session_state["transcripts"] = transcripts_df
    st.sidebar.success(f"Loaded {len(transcripts_df)} transcripts")

# 2 — Survey builder
if "transcripts" in st.session_state:
    st.header("Transcript preview")
    st.dataframe(st.session_state["transcripts"].head())

    st.subheader("Survey builder")
    question = st.text_area(
        "Survey question", "Who will you vote for in the upcoming election?"
    )
    num_respondents = st.slider("Synthetic respondents", 1, 200, 10)
    persona = st.radio(
        "Persona", ["pollster", "marketer", "product manager"], horizontal=True
    )
    run = st.button("Generate synthetic answers")

    if run and question:
        with st.spinner("Generating answers…"):
            session = get_session(DB_PATH)
            transcripts = st.session_state["transcripts"]
            results = []
            for _ in range(num_respondents):
                record = transcripts.sample(1).iloc[0]
                answer, conf, usage = synthesize_answer(record, question, persona)
                add_response(session, record, question, answer, conf)
                results.append(
                    {
                        "respondent_id": record["respondent_id"],
                        "answer": answer,
                        "confidence": conf,
                    }
                )
            df = pd.DataFrame(results)
        st.success("Generation complete!")
        st.dataframe(df)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="synthetic_responses.csv",
            mime="text/csv",
        )

