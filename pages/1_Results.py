import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Survey Results & Demographic Insights")

# ------------------------------------------------------------
# 0 — Load dataframe from session_state
# ------------------------------------------------------------
if "latest_df" not in st.session_state:
    st.info("Run a survey on the Home page first ⬅︎")
    st.stop()

df = st.session_state["latest_df"]

# ------------------------------------------------------------
# 1 — Global metrics
# ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total answers", len(df))
c2.metric("Unique personas", df["respondent_id"].nunique())
c3.metric("Questions asked", df["question"].nunique())

st.divider()

# ------------------------------------------------------------
# 2 — Interactive question selector
# ------------------------------------------------------------
questions = df["question"].unique().tolist()
q_selected = st.selectbox("Select a question to analyse", questions)
subset = df[df["question"] == q_selected]

# ------------------------------------------------------------
# 3 — Answer distribution
# ------------------------------------------------------------
st.subheader("Answer distribution")
answer_counts = subset["answer"].value_counts().reset_index()
answer_counts.columns = ["answer", "count"]

tab_bar, tab_pie = st.tabs(["Bar chart", "Pie chart"])
with tab_bar:
    fig_bar = px.bar(answer_counts, x="answer", y="count",
                     labels={"answer": "Answer", "count": "Count"})
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_pie:
    fig_pie = px.pie(answer_counts, names="answer", values="count")
    st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------
# 4 — Demographic breakdown
# ------------------------------------------------------------
st.subheader("Demographic breakdown")

demo_tab1, demo_tab2, demo_tab3 = st.tabs(["Gender", "Age", "Region"])

# -- gender
with demo_tab1:
    if subset["gender"].notna().any():
        fig = px.histogram(subset, x="gender", color="answer",
                           barmode="group",
                           category_orders={"gender": sorted(subset["gender"].dropna().unique())})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No gender data available.")

# -- age
with demo_tab2:
    if subset["age"].notna().any():
        fig = px.histogram(subset, x="age", nbins=10, color="answer",
                           marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No age data available.")

# -- region
with demo_tab3:
    if subset["region"].notna().any():
        region_counts = subset.groupby(["region", "answer"]).size().reset_index(name="count")
        fig = px.bar(region_counts, x="region", y="count", color="answer",
                     category_orders={"region": sorted(region_counts["region"].unique())})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No region data available.")

# ------------------------------------------------------------
# 5 — Word cloud for open-ended answers
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 6 — Download filtered dataset
# ------------------------------------------------------------
csv = subset.to_csv(index=False)
st.download_button("Download this question’s answers (CSV)",
                   data=csv,
                   file_name="survey_subset.csv",
                   mime="text/csv")
