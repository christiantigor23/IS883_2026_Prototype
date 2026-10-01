import streamlit as st
from google import genai
from google.genai import types

### Load your API Key
try:
    gemini_api_key = st.secrets['MyGeminiKey']  # Info: https://docs.streamlit.io/develop/api-reference/connections/st.secrets
except (KeyError, FileNotFoundError):
    st.error("No Gemini key found. Add `MyGeminiKey` under **Manage app → ⋮ → Settings → Secrets**, then refresh this page.")
    st.stop()

client = genai.Client(api_key=gemini_api_key)
MODEL = "gemini-3.1-flash-lite"

### User input
query = st.text_area("Enter your question", placeholder="e.g., Who is the president of the USA?")

if st.button("Ask Gemini"):
    if not query.strip():
        st.warning("Type a question first, then press Ask Gemini.")
    else:
        with st.spinner("Thinking..."):
            response = client.models.generate_content(model=MODEL, contents=query)
        st.write(response.text)
