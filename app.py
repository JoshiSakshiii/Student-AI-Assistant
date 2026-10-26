
import os
import ast
import operator
from io import StringIO

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------- GROQ CONFIG ----------------

try:
    API_KEY = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY", "")
    )

    MODEL_NAME = st.secrets.get(
        "GROQ_MODEL",
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    )

except Exception:
    API_KEY = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

BASE_URL = "https://api.groq.com/openai/v1"


# ---------------- API CLIENT ----------------

@st.cache_resource
def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=45.0,
        max_retries=1
    )


# ---------------- SAFE CALCULATOR ----------------

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
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
            operation = type(node.op)

            if operation not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            return ALLOWED_OPERATORS[operation](
                evaluate(node.operand)
            )

        if isinstance(node, ast.BinOp):
            operation = type(node.op)

            if operation not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > 10 or abs(left) > 1_000_000:
                    raise ValueError("Exponent or number is too large.")

            result = ALLOWED_OPERATORS[operation](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        raise ValueError("Invalid mathematical expression.")

    return evaluate(tree.body)


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🎓 Student AI Assistant")

    st.write(
        "Your AI study companion for learning, programming, "
        "mathematics, and examination preparation."
    )

    st.divider()

    answer_style = st.selectbox(
        "Answer Style",
        ["Simple", "Detailed", "Exam Answer"]
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
            use_container_width=True
        )

    st.divider()
    st.caption("Powered by Groq")


# ---------------- MAIN INTERFACE ----------------

st.title("🎓 Student AI Assistant")

st.write(
    "Ask questions, learn difficult concepts, solve "
    "mathematical problems, and prepare for exams."
)

st.caption(
    "Try: Explain cloud computing | Explain Python loops | "
    "Calculate: 25 * 4"
)


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- ANSWER STYLES ----------------

STYLE_INSTRUCTIONS = {

    "Simple": (
        "Use simple English, short sentences, and easy examples. "
        "Explain technical terms clearly."
    ),

    "Detailed": (
        "Provide a detailed explanation with headings, examples, "
        "important concepts, and practical applications."
    ),

    "Exam Answer": (
        "Write an exam-ready answer using a definition, headings, "
        "numbered points, examples where appropriate, and a conclusion."
    )
}


# ---------------- CHAT INPUT ----------------

user_prompt = st.chat_input("Ask your study question...")


if user_prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # -------- CALCULATOR --------

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

    # -------- CHECK API KEY --------

    elif not API_KEY:

        answer = (
            "⚠️ **Groq API key is missing.**\n\n"
            "Open Streamlit Cloud → App Settings → Secrets "
            "and add `GROQ_API_KEY`."
        )

        with st.chat_message("assistant"):
            st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    # -------- GROQ REQUEST --------

    else:

        with st.chat_message("assistant"):

            with st.spinner("Preparing your answer..."):

                try:

                    client = get_client(API_KEY)

                    system_prompt = (
                        "You are Student AI Assistant, a helpful "
                        "educational assistant. Help students with "
                        "academic subjects, programming, mathematics, "
                        "science, and exam preparation. "
                        "Be accurate, clear, and honest about uncertainty. "
                        "Do not invent facts or references. "
                        + STYLE_INSTRUCTIONS[answer_style]
                    )

                    api_messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        }
                    ] + st.session_state.messages[-12:]

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = (
                            "The model returned an empty response. "
                            "Please try again."
                        )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:

                    status = getattr(error, "status_code", None)
                    body = getattr(error, "body", None)

                    details = (
                        str(body)
                        if body is not None
                        else str(error)
                    )

                    if status == 400:
                        explanation = (
                            "The request was rejected. Check the model ID "
                            "and the API's supported request parameters."
                        )

                    elif status == 401:
                        explanation = (
                            "Authentication failed. Check your Groq API key."
                        )

                    elif status == 403:
                        explanation = (
                            "Access denied. Check your account and model access."
                        )

                    elif status == 404:
                        explanation = (
                            "Model or endpoint not found. Verify the model ID."
                        )

                    elif status == 429:
                        explanation = (
                            "Rate limit or usage limit reached. "
                            "Check your Groq account limits."
                        )

                    else:
                        explanation = (
                            "Check your API key, model ID, network, "
                            "and Groq account access."
                        )

                    answer = (
                        f"**Groq API error: HTTP {status}**\n\n"
                        f"{explanation}\n\n"
                        f"**Details:**\n```text\n"
                        f"{details[:2000]}\n```"
                    )

                    st.error(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

