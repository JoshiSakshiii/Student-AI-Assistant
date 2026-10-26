```python
import os
import ast
import operator
import json
from io import StringIO

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide",
)

load_dotenv()

MODEL_NAME = os.getenv("GROK_MODEL", "grok-3")
GROK_BASE_URL = "https://api.x.ai/v1"

# Streamlit Cloud Secrets first; local .env as fallback.
try:
    API_KEY = st.secrets.get(
        "XAI_API_KEY",
        os.getenv("XAI_API_KEY", "")
    )
except Exception:
    API_KEY = os.getenv("XAI_API_KEY", "")


# --------------------------------------------------
# 2. SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_PROMPT = """
You are Student AI Assistant, a helpful and friendly AI tutor.

Your responsibilities:
1. Explain computer science and engineering concepts.
2. Help students understand programming and debugging.
3. Explain difficult topics in simple language.
4. Provide step-by-step explanations when requested.
5. Give examples relevant to students.
6. Adapt your answer to the user's requested response style.
7. Use relevant earlier messages in the current conversation.
8. If uncertain, explain the uncertainty instead of inventing facts.
9. Do not claim that you executed code unless you actually did.
10. Use the calculator tool for arithmetic when appropriate.
"""


# --------------------------------------------------
# 3. INITIALIZE GROK CLIENT
# --------------------------------------------------
@st.cache_resource
def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url=GROK_BASE_URL,
    )


# --------------------------------------------------
# 4. SAFE CALCULATOR
# --------------------------------------------------
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator
```
