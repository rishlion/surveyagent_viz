import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

st.title("📊 Survey Results & Insights")

# ------------------------------------------------------------------
# 1 — Load the latest dataframe from session_state or DB
# ------------------------------------------------------------------
if "latest_df" not in st.session_state:
    st.info("Run a survey on the Home page first ⬅︎")
    st.stop()

df = st.session_state["latest_df"]

# ------------------------------------------------------------------
# 2 — High-level stats
# ------------------------------------------------------------------
st.metric("Total synthetic answers", len(df))
st.metric("Unique synthetic personas", df["respondent_id"].nunique())

# ------------------------------------------------------------------
# 3 — Choose a question to visualise
# ------------------------------------------------------------------
questions = df["question"].unique().tolist()
q_selected = st.selectbox("Select a question", questions)

subset = df[df["question"] == q_selected]

# ------------------------------------------------------------------
# 4 — Visual 1: answer distribution (bar or pie)
# ------------------------------------------------------------------
answer_counts = subset["answer"].value_counts().reset_index()
answer_counts.columns = ["answer", "count"]

chart_type = st.radio("Chart type", ["Bar", "Pie"], horizontal=True)

if chart_type == "Bar":
    fig = px.bar(answer_counts, x="answer", y="count", title="Answer counts")
else:
    fig = px.pie(answer_counts, names="answer", values="count", title="Answer share")

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 5 — Visual 2: word cloud for open-ended answers
# ------------------------------------------------------------------
st.subheader("Word cloud (open-ended only)")
text_blob = " ".join(subset["answer"].astype(str).tolist())

if len(text_blob.split()) < 5:
    st.write("Not enough text to generate a word cloud.")
else:
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# ------------------------------------------------------------------
# 6 — Download filtered data
# ------------------------------------------------------------------
csv = subset.to_csv(index=False)
st.download_button(
    "Download this question’s answers (CSV)",
    data=csv,
    file_name="survey_subset.csv",
    mime="text/csv",
)
