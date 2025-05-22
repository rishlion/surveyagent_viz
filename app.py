import streamlit as st
import pandas as pd
from pathlib import Path
from agent import synthesize_answer
from utils import load_transcripts
from data_model import create_db_and_tables, add_response, get_session
import openai, json, math

# -------------------------------------------------------------------------
# Config & setup
# -------------------------------------------------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

DB_PATH = "data/database.db"
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
create_db_and_tables(DB_PATH)

st.sidebar.title("Data & Filters")

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

    age_mask, gender_mask, region_mask = True, True, True

    # Age
    if "age" in df.columns:
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
        age_mask = df["age"].between(*age_range)

    # Gender
    if "gender" in df.columns:
        genders = sorted(df["gender"].dropna().unique())
        gender_sel = st.sidebar.multiselect("Gender", genders, default=genders)
        gender_mask = df["gender"].isin(gender_sel)

    # Region
    if "region" in df.columns:
        regions = sorted(df["region"].dropna().unique())
        region_sel = st.sidebar.multiselect("Region", regions, default=regions)
        region_mask = df["region"].isin(region_sel)

    filtered_df = df[age_mask & gender_mask & region_mask]
    st.sidebar.markdown(f"**Matched transcripts:** {len(filtered_df)}")
    st.session_state["filtered"] = filtered_df

# -------------------------------------------------------------------------
# 3 — Question list builder (main pane)
# -------------------------------------------------------------------------
if "filtered" in st.session_state and len(st.session_state["filtered"]) > 0:
    st.header("Survey Agent")
    st.caption("Instantly generate realistic, demographic-matched survey answers with an LLM-powered synthetic panel.")

    st.subheader("Step 1: 👥 Review sample personas")
    st.dataframe(st.session_state["filtered"].head())

    st.subheader("Step 2: 📋 Build your question list")

    # Persistent list
    if "questions" not in st.session_state:
        st.session_state.questions = []

    # --- Text input + callback -------------------------------------------
    def add_question():
        q = st.session_state.new_q_input.strip()
        if q:
            st.session_state.questions.append(q)
            st.session_state.new_q_input = ""

    st.text_input("Type a question and click “Add”", key="new_q_input")
    st.button("Add question", on_click=add_question)

    # Show existing questions with delete
    for i, q in enumerate(st.session_state.questions):
        cols = st.columns((10, 1))
        cols[0].markdown(f"**Q{i+1}.** {q}")
        if cols[1].button("✖︎", key=f"del_{i}"):
            st.session_state.questions.pop(i)
            st.experimental_rerun()

    num_q = len(st.session_state.questions)

    # ---------------------------------------------------------------------
    # Generation controls
    # ---------------------------------------------------------------------
    st.subheader("Step 3: 📊 Select number of survey responses")
    num_resp = st.slider("Respondents per question", 1, 200, 10)
    persona = st.radio(
        "Persona",
        ["pollster", "marketer", "product manager"],
        horizontal=True,
    )

    gen_label = (
        f"Generate {num_resp} answers" if num_q == 1
        else f"Generate {num_q * num_resp} answers"
    )
    generate = st.button(gen_label, disabled=num_q == 0)

    

    if generate:
        filtered_df = st.session_state["filtered"]

        if num_resp > len(filtered_df):
            st.warning(
                "Requested more respondents than available transcripts – sampling with replacement."
            )

        respondent_pool = (
            filtered_df.sample(n=num_resp, replace=num_resp > len(filtered_df))
            .to_dict("records")
        )

        with st.spinner("Generating answers…"):
            session = get_session(DB_PATH)
            results = []

            total_iters = num_q * num_resp
            progress = st.progress(0)
            counter_placeholder = st.empty()
            counter = 0

            for record in respondent_pool:
                for q in st.session_state["questions"]:
                    try:
                        answer, conf, _usage = synthesize_answer(record, q, persona)
                    except (json.JSONDecodeError, KeyError):
                        answer, conf = "ERROR: malformed response", 0
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
                    counter_placeholder.text(f"{counter}/{total_iters} answers")

            progress.empty()
            counter_placeholder.empty()
            df_out = pd.DataFrame(results)

        st.success("Generation complete!")
        
        st.subheader("Result: 📈 Review survey responses!")

        st.dataframe(df_out)
        
        st.download_button(
            "Download CSV",
            data=df_out.to_csv(index=False),
            file_name="synthetic_responses.csv",
            mime="text/csv",
        )

elif "transcripts" in st.session_state and len(st.session_state["filtered"]) == 0:
    st.warning("No transcripts match the selected demographic filters.")
