import streamlit as st
import pandas as pd
from pathlib import Path
from agent import synthesize_answer
from utils import load_transcripts
from data_model import create_db_and_tables, add_response, get_session
import openai

# -------------------------------------------------------------------------
# Config & setup
# -------------------------------------------------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

DB_PATH = "data/database.db"
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
create_db_and_tables(DB_PATH)

st.sidebar.title("Survey Agent MVP")

# -------------------------------------------------------------------------
# 1 — Choose transcript data source
# -------------------------------------------------------------------------
data_source = st.sidebar.radio(
    "Transcript data:",
    ("Use bundled sample", "Upload my own"),
    index=0,
)

if data_source == "Upload my own":
    uploaded_file = st.sidebar.file_uploader(
        "Upload transcripts CSV or Parquet", type=["csv", "parquet"]
    )
    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        transcripts_df = load_transcripts(file_path)
        st.session_state["transcripts"] = transcripts_df
        st.sidebar.success(f"Loaded {len(transcripts_df)} transcripts")
else:
    sample_path = Path(__file__).parent / "sample_transcripts.csv"
    transcripts_df = load_transcripts(sample_path)
    st.session_state["transcripts"] = transcripts_df
    st.sidebar.info(f"Using bundled sample data ({len(transcripts_df)} rows)")

# -------------------------------------------------------------------------
# 2 — Demographic filters (sidebar)
# -------------------------------------------------------------------------
if "transcripts" in st.session_state:
    df = st.session_state["transcripts"]

    # --- Age slider -------------------------------------------------------
    if "age" in df.columns:
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.sidebar.slider(
            "Age range", age_min, age_max, (age_min, age_max)
        )
        age_mask = df["age"].between(*age_range)
    else:
        age_mask = True  # no age column present

    # --- Gender filter ----------------------------------------------------
    if "gender" in df.columns:
        genders = sorted(df["gender"].dropna().unique())
        gender_sel = st.sidebar.multiselect(
            "Gender", options=genders, default=genders
        )
        gender_mask = df["gender"].isin(gender_sel)
    else:
        gender_mask = True

    # --- Region filter ----------------------------------------------------
    if "region" in df.columns:
        regions = sorted(df["region"].dropna().unique())
        region_sel = st.sidebar.multiselect(
            "Region", options=regions, default=regions
        )
        region_mask = df["region"].isin(region_sel)
    else:
        region_mask = True

    # Apply masks ----------------------------------------------------------
    filtered_df = df[age_mask & gender_mask & region_mask]
    st.sidebar.markdown(f"**Matched transcripts:** {len(filtered_df)}")
    st.session_state["filtered"] = filtered_df

# -------------------------------------------------------------------------
# 3 — Preview & survey builder (main pane)
# -------------------------------------------------------------------------
if "filtered" in st.session_state and len(st.session_state["filtered"]) > 0:
    st.header("Transcript preview (after filters)")
    st.dataframe(st.session_state["filtered"].head())

    st.subheader("Survey builder")
    question = st.text_area(
        "Survey question",
        "Who will you vote for in the upcoming election?",
        height=100,
    )
    num_respondents = st.slider("Synthetic respondents", 1, 200, 10)
    persona = st.radio(
        "Persona",
        ["pollster", "marketer", "product manager"],
        horizontal=True,
    )

    generate = st.button(
        "Generate synthetic answers",
        disabled=len(st.session_state["filtered"]) == 0,
    )

    if generate and question:
        with st.spinner("Generating answers…"):
            session = get_session(DB_PATH)
            results = []

            for _ in range(num_respondents):
                record = st.session_state["filtered"].sample(1).iloc[0]
                answer, conf, _usage = synthesize_answer(record, question, persona)
                add_response(session, record, question, answer, conf)
                results.append(
                    {
                        "respondent_id": record["respondent_id"],
                        "answer": answer,
                        "confidence": conf,
                    }
                )

            df_out = pd.DataFrame(results)

        st.success("Generation complete!")
        st.dataframe(df_out)
        st.download_button(
            "Download CSV",
            data=df_out.to_csv(index=False),
            file_name="synthetic_responses.csv",
            mime="text/csv",
        )

elif "transcripts" in st.session_state and len(st.session_state["filtered"]) == 0:
    st.warning("No transcripts match the selected demographic filters.")
