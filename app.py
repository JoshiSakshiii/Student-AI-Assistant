
import os
import ast
import operator
from io import StringIO

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# Read credentials from Streamlit Secrets or local .env.
try:
    API_KEY = st.secrets.get(
        "GROK_API_KEY",
        os.getenv("GROK_API_KEY", "")
    )
    SECRET_MODEL = st.secrets.get(
        "GROK_MODEL",
        os.getenv("GROK_MODEL", "")
    )
except Exception:
    API_KEY = os.getenv("GROK_API_KEY", "")
    SECRET_MODEL = os.getenv("GROK_MODEL", "")

# The model name must be available to your xAI API account.
MODEL_NAME = SECRET_MODEL or "grok-4-1-fast"

GROK_BASE_URL = "https://api.x.ai/v1"


# =====================================================
# 2. API CLIENT
# =====================================================

@st.cache_resource
def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url=GROK_BASE_URL,
        timeout=45.0,
        max_retries=1
    )


# =====================================================
# 3. SAFE CALCULATOR
# =====================================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):
    """Evaluate basic arithmetic without using eval()."""

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > 10 or abs(left) > 1_000_000:
                    raise ValueError("Exponent or number is too large.")

            result = ALLOWED_OPERATORS[op_type](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            return ALLOWED_OPERATORS[op_type](evaluate(node.operand))

        raise ValueError("Invalid mathematical expression.")

    return evaluate(tree.body)


# =====================================================
# 4. SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# 5. SIDEBAR
# =====================================================

with st.sidebar:
    st.title("🎓 Student AI Assistant")

    st.write(
        "Your AI study companion for academic learning, "
        "programming, mathematics, and exam preparation."
    )

    st.divider()

    response_style = st.selectbox(
        "Choose Answer Style",
        ["Simple", "Detailed", "Exam Answer"]
    )

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        chat_text = StringIO()

        for message in st.session_state.messages:
            chat_text.write(
                f"{message['role'].upper()}:\n"
                f"{message['content']}\n\n"
            )

        st.download_button(
            "📥 Download Chat",
            data=chat_text.getvalue(),
            file_name="student_ai_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()
    st.caption("AI provider: Grok by xAI")


# =====================================================
# 6. MAIN INTERFACE
# =====================================================

st.title("🎓 Student AI Assistant")

st.markdown(
    "Ask questions, understand difficult concepts, solve "
    "mathematical problems, and prepare for examinations."
)

st.caption(
    "Examples: Explain cloud computing | What is Python? | "
    "Calculate: 25 * 4"
)


# =====================================================
# 7. DISPLAY PREVIOUS MESSAGES
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# 8. RESPONSE STYLE INSTRUCTIONS
# =====================================================

STYLE_INSTRUCTIONS = {
    "Simple": (
        "Explain in simple English. Use short sentences, "
        "easy words, and examples suitable for a student."
    ),
    "Detailed": (
        "Provide a detailed and organized explanation. "
        "Include examples, important concepts, and practical uses."
    ),
    "Exam Answer": (
        "Prepare an exam-ready answer with a definition, "
        "headings, key points, examples where useful, and a "
        "conclusion when appropriate."
    )
}


# =====================================================
# 9. CHAT PROCESSING
# =====================================================

user_prompt = st.chat_input("Ask your study question here...")

if user_prompt:

    # Save and display the question.
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ---------------------------------------------
    # Calculator
    # ---------------------------------------------

    if user_prompt.lower().startswith("calculate:"):

        expression = user_prompt.split(":", 1)[1].strip()

        try:
            result = safe_calculate(expression)
            answer = (
                f"**Expression:** `{expression}`\n\n"
                f"**Result:** {result}"
            )

        except Exception as error:
            answer = f"Unable to calculate: {error}"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    # ---------------------------------------------
    # Missing API key
    # ---------------------------------------------

    elif not API_KEY:

        answer = (
            "⚠️ **Grok API key is missing.**\n\n"
            "Open Streamlit Cloud → your app → Settings → Secrets "
            "and add `GROK_API_KEY`."
        )

        with st.chat_message("assistant"):
            st.warning(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    # ---------------------------------------------
    # Grok API request
    # ---------------------------------------------

    else:

        with st.chat_message("assistant"):
            with st.spinner("Grok is generating your answer..."):

                try:
                    client = get_client(API_KEY)

                    system_prompt = (
                        "You are Student AI Assistant, a helpful "
                        "educational assistant. Help students with "
                        "academic subjects, programming, mathematics, "
                        "science, and examination preparation. "
                        "Be accurate, clear, and honest about uncertainty. "
                        "Do not invent facts or references. "
                        f"{STYLE_INSTRUCTIONS[response_style]}"
                    )

                    # Include recent conversation context.
                    recent_messages = st.session_state.messages[-12:]

                    api_messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        }
                    ] + recent_messages

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = (
                            "Grok returned an empty response. "
                            "Please try asking your question again."
                        )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                # ---------------------------------
                # Detailed API error handling
                # ---------------------------------

                except Exception as error:

                    error_details = str(error)
                    error_code = getattr(error, "status_code", None)

                    if error_code == 400:
                        answer = (
                            "❌ **Grok rejected the API request (400).**\n\n"
                            "Possible causes include an unsupported model "
                            "or an invalid request. Check the model name "
                            "in Streamlit Secrets.\n\n"
                            f"**Actual error:** `{error_details[:1500]}`"
                        )

                    elif error_code == 401:
                        answer = (
                            "❌ **Authentication failed (401).**\n\n"
                            "Check that `GROK_API_KEY` contains a valid "
                            "xAI API key. Do not use an OpenAI API key here."
                        )

                    elif error_code == 403:
                        answer = (
                            "❌ **Access denied (403).**\n\n"
                            "Your account may not have permission to use "
                            "this model or API service. Check your xAI "
                            "Console account and model access."
                        )

                    elif error_code == 404:
                        answer = (
                            "❌ **Resource not found (404).**\n\n"
                            "Check the model name and API endpoint."
                        )

                    elif error_code == 429:
                        answer = (
                            "⚠️ **API usage limit or credit issue (429).**\n\n"
                            "Check your xAI Console credits, billing, "
                            "and rate limits."
                        )

                    else:
                        answer = (
                            "❌ **Unable to get a response from Grok.**\n\n"
                            "Check your API key, internet connection, "
                            "account access, and model availability.\n\n"
                            f"**Error code:** `{error_code}`\n\n"
                            f"**Actual error:** `{error_details[:1500]}`"
                        )

                    st.error(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

