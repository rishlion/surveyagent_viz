# Survey Agent – MVP

This is a Streamlit application that demonstrates how to generate **synthetic survey responses** from an LLM prompted with real interview transcripts.

## Quick start

```bash
# 1) install deps
pip install -r requirements.txt

# 2) set your OpenAI key (or fill .streamlit/secrets.toml)
export OPENAI_API_KEY="sk-..."

# 3) launch
streamlit run app.py
