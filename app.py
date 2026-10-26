
import os
import ast
import operator
from io import StringIO

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide",
)

# ==================================================
# 2. AI/ML API CONFIGURATION
# ==================================================

try:
    API_KEY = st.secrets.get(
        "AIMLAPI_API_KEY",
        os.getenv("AIMLAPI_API_KEY", ""),
    )
    SECRET_MODEL = st.secrets.get(
        "AIMLAPI_MODEL",
        os.getenv("AIMLAPI_MODEL", ""),
    )
except Exception:
    API_KEY = os.getenv("AIMLAPI_API_KEY", "")
    SECRET_MODEL = os.getenv("AIMLAPI_MODEL", "")

BASE_URL = "https://api.aimlapi.com/v1"

# Change this if the model is not available in your account.
MODEL_NAME = SECRET_MODEL or "openai/gpt-5.6-sol"


# ==================================================
# 3. API CLIENT
# ==================================================

@st.cache_resource
def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=1,
    )


# ==================================================
# 4. SAFE CALCULATOR
# ==================================================

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
    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            return ALLOWED_OPERATORS[op_type](
                evaluate(node.operand)
            )

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

        raise ValueError("Invalid mathematical expression.")

    return evaluate(tree.body)


# ==================================================
# 5. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:
    st.title("🎓 Student AI Assistant")

    st.write(
        "Your AI study companion for learning, programming, "
        "mathematics, science, and examination preparation."
    )

    st.divider()

    answer_style = st.selectbox(
        "Answer Style",
        ["Simple", "Detailed", "Exam Answer"],
    )

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
            use_container_width=True,
        )

    st.divider()
    st.caption("Powered by AI/ML API")


# ==================================================
# 7. MAIN INTERFACE
# ==================================================

st.title("🎓 Student AI Assistant")

st.write(
    "Ask questions, understand difficult concepts, "
    "solve mathematical problems, and prepare for exams."
)

st.caption(
    "Try: Explain cloud computing | Explain Python loops | "
    "Calculate: 25 * 4"
)


# ==================================================
# 8. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==================================================
# 9. RESPONSE STYLES
# ==================================================

STYLE_INSTRUCTIONS = {
    "Simple": (
        "Use simple English, short sentences, and easy examples. "
        "Explain technical words clearly."
    ),
    "Detailed": (
        "Provide a detailed explanation with headings, examples, "
        "important concepts, and practical applications."
    ),
    "Exam Answer": (
        "Write an exam-ready answer using a definition, headings, "
        "numbered points, examples where useful, and a conclusion "
        "when appropriate."
    ),
}


# ==================================================
# 10. PROCESS USER QUESTION
# ==================================================

user_prompt = st.chat_input("Ask your study question...")

if user_prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ----------------------------------------------
    # A. CALCULATOR — NO API REQUIRED
    # ----------------------------------------------

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
            "content": answer,
        })

    # ----------------------------------------------
    # B. CHECK API KEY
    # ----------------------------------------------

    elif not API_KEY:

        answer = (
            "The AI/ML API key is missing.\n\n"
            "Open Streamlit Cloud → App Settings → Secrets "
            "and add `AIMLAPI_API_KEY`."
        )

        with st.chat_message("assistant"):
            st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

    # ----------------------------------------------
    # C. CALL AI/ML API
    # ----------------------------------------------

    else:

        with st.chat_message("assistant"):
            with st.spinner("Preparing your answer..."):

                try:
                    client = get_client(API_KEY)

                    system_prompt = (
                        "You are Student AI Assistant, an educational "
                        "assistant helping students with academic subjects, "
                        "programming, mathematics, science, and examinations. "
                        "Give clear, accurate, well-organized answers. "
                        "Do not invent facts or references. "
                        "Admit uncertainty when necessary. "
                        f"Answer style: {STYLE_INSTRUCTIONS[answer_style]}"
                    )

                    recent_messages = st.session_state.messages[-12:]

                    api_messages = [
                        {
                            "role": "system",
                            "content": system_prompt,
                        }
                    ] + recent_messages

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages,
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = (
                            "The model returned an empty answer. "
                            "Please try again."
                        )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                except Exception as error:

                    status = getattr(error, "status_code", None)
                    body = getattr(error, "body", None)

                    if body is not None:
                        details = str(body)
                    else:
                        details = str(error)

                    if status == 400:
                        heading = (
                            "The request was rejected. Check the model ID "
                            "and the model's supported request format."
                        )
                    elif status == 401:
                        heading = (
                            "Authentication failed. Check your AI/ML API key."
                        )
                    elif status == 403:
                        heading = (
                            "Access denied. Your account may not have "
                            "permission to use this model."
                        )
                    elif status == 404:
                        heading = (
                            "The model or endpoint was not found. "
                            "Verify the model ID in the AI/ML API catalogue."
                        )
                    elif status == 429:
                        heading = (
                            "The API usage limit was reached. "
                            "Check your account credits and rate limits."
                        )
                    else:
                        heading = (
                            "The API request failed. Check your credentials, "
                            "account access, network, and model configuration."
                        )

                    answer = (
                        f"**API Error (HTTP {status}):**\n\n"
                        f"{heading}\n\n"
                        f"**Error details:**\n```text\n"
                        f"{details[:2000]}\n```"
                    )

                    st.error(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

