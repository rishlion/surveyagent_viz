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

    # Age
    if "age" in df.columns:
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
        age_mask = df["age"].between(*age_range)
    else:
        age_mask = True

    # Gender
    if "gender" in df.columns:
        genders = sorted(df["gender"].dropna().unique())
        gender_sel = st.sidebar.multiselect("Gender", genders, default=genders)
        gender_mask = df["gender"].isin(gender_sel)
    else:
        gender_mask = True

    # Region
    if "region" in df.columns:
        regions = sorted(df["region"].dropna().unique())
        region_sel = st.sidebar.multiselect("Region", regions, default=regions)
        region_mask = df["region"].isin(region_sel)
    else:
        region_mask = True

    filtered_df = df[age_mask & gender_mask & region_mask]
    st.sidebar.markdown(f"**Matched transcripts:** {len(filtered_df)}")
    st.session_state["filtered"] = filtered_df

# -------------------------------------------------------------------------
# 3 — Preview & survey builder
# -------------------------------------------------------------------------
if "filtered" in st.session_state and len(st.session_state["filtered"]) > 0:
    st.header("Transcript preview (after filters)")
    st.dataframe(st.session_state["filtered"].head())

    st.subheader("Survey builder")

    questions_raw = st.text_area(
        "Enter one question per line",
        "Who will you vote for in the upcoming election?",
        height=150,
    )
    questions = [q.strip() for q in questions_raw.splitlines() if q.strip()]
    num_questions = len(questions)

    num_respondents = st.slider("Synthetic respondents per question", 1, 200, 10)
    persona = st.radio(
        "Persona",
        ["pollster", "marketer", "product manager"],
        horizontal=True,
    )

    generate = st.button(
        f"Generate ({num_questions * num_respondents} answers)",
        disabled=(num_questions == 0 or len(st.session_state["filtered"]) == 0),
    )

    if generate:
        with st.spinner("Generating answers…"):
            session = get_session(DB_PATH)
            results = []

            total_iters = num_questions * num_respondents
            progress = st.progress(0)
            counter_placeholder = st.empty()
            counter = 0

            for q in questions:
                for _ in range(num_respondents):
                    record = st.session_state["filtered"].sample(1).iloc[0]
                    answer, conf, _usage = synthesize_answer(record, q, persona)
                    add_response(session, record, q, answer, conf)
                    results.append(
                        {
                            "respondent_id": record["respondent_id"],
                            "question": q,
                            "answer": answer,
                            "confidence": conf,
                        }
                    )
                    counter += 1
                    progress.progress(counter / total_iters)
                    counter_placeholder.text(f"{counter}/{total_iters} answers done")

            progress.empty()
            counter_placeholder.empty()
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
