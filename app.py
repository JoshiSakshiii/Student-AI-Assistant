```python
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

# ---------------- GROK CONFIGURATION ----------------

try:
    API_KEY = st.secrets.get(
        "GROK_API_KEY",
        os.getenv("GROK_API_KEY", "")
    )
except Exception:
    API_KEY = os.getenv("GROK_API_KEY", "")

MODEL_NAME = os.getenv("GROK_MODEL", "grok-4.1-fast")

# ---------------- API CLIENT ----------------

@st.cache_resource
def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
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
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):
    """Evaluate basic mathematical expressions safely."""

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            # Prevent excessively large exponent calculations.
            if isinstance(node.op, ast.Pow):
                if abs(evaluate(node.right)) > 10:
                    raise ValueError("Exponent is too large.")

                left = evaluate(node.left)
                right = evaluate(node.right)

                if abs(left) > 1_000_000:
                    raise ValueError("Number is too large.")

                return operator.pow(left, right)

            return ALLOWED_OPERATORS[op_type](
                evaluate(node.left),
                evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            return ALLOWED_OPERATORS[op_type](
                evaluate(node.operand)
            )

        raise ValueError("Invalid mathematical expression.")

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🎓 Student AI Assistant")

    st.write(
        "Your AI study companion for learning, "
        "programming, mathematics, and exam preparation."
    )

    st.divider()

    response_style = st.selectbox(
        "Answer Style",
        ["Simple", "Detailed", "Exam Answer"]
    )

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        chat_text = StringIO()

        for message in st.session_state.messages:
            role = message["role"].upper()
            chat_text.write(f"{role}:\n{message['content']}\n\n")

        st.download_button(
            "📥 Download Chat",
            data=chat_text.getvalue(),
            file_name="student_ai_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()
    st.caption("Powered by Grok API")


# ---------------- MAIN INTERFACE ----------------

st.title("🎓 Student AI Assistant")

st.write(
    "Ask questions, understand difficult topics, "
    "solve mathematical problems, and prepare for exams."
)

# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------

user_prompt = st.chat_input(
    "Ask your study question here..."
)

if user_prompt:
    # Display and save the user's question.
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Check whether the user is asking for a calculation.
    if user_prompt.lower().startswith("calculate:"):
        expression = user_prompt.split(":", 1)[1].strip()

        try:
            result = safe_calculate(expression)
            answer = f"**Calculation:** `{expression}`\n\n**Result:** {result}"
        except Exception as error:
            answer = f"Could not calculate the expression: {error}"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    elif not API_KEY:
        answer = (
            "⚠️ **Grok API key is missing.**\n\n"
            "Open your Streamlit Cloud app's Settings → Secrets "
            "and add `GROK_API_KEY`."
        )

        with st.chat_message("assistant"):
            st.warning(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    else:
        with st.chat_message("assistant"):
            with st.spinner("Grok is preparing your answer..."):
                try:
                    client = get_client(API_KEY)

                    style_instructions = {
                        "Simple": (
                            "Explain in simple English, using short "
                            "sentences and examples suitable for a student."
                        ),
                        "Detailed": (
                            "Give a detailed, well-organized explanation "
                            "with examples and important points."
                        ),
                        "Exam Answer": (
                            "Write an exam-ready answer with a definition, "
                            "headings, key points, and a conclusion where "
                            "appropriate. Match the question's expected length."
                        )
                    }

                    system_prompt = (
                        "You are a helpful Student AI Assistant. "
                        "Help students understand academic topics, "
                        "programming, mathematics, science, and exam questions. "
                        "Be accurate, clear, and honest when uncertain. "
                        "Do not invent facts or sources. "
                        f"Response style: {style_instructions[response_style]}"
                    )

                    # Send recent conversation context.
                    recent_messages = st.session_state.messages[-12:]

                    api_messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        }
                    ] + recent_messages

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages,
                        temperature=0.4
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = "Grok returned an empty response. Please try again."

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    error_message = str(error)

                    if "401" in error_message:
                        answer = (
                            "❌ **Authentication failed.** "
                            "Check that `GROK_API_KEY` contains a valid "
                            "xAI API key."
                        )
                    elif "429" in error_message:
                        answer = (
                            "⚠️ **API usage limit or credit issue.** "
                            "Check your xAI Console usage, credits, and "
                            "rate limits."
                        )
                    elif "model" in error_message.lower():
                        answer = (
                            "⚠️ The selected Grok model may be unavailable "
                            "for your account. Check the available models "
                            "in the xAI Console and update `GROK_MODEL`."
                        )
                    else:
                        answer = (
                            "❌ Unable to get a response from Grok.\n\n"
                            "Check your internet connection, API key, "
                            "API credits, and model availability.\n\n"
                            f"Error details: `{error_message[:500]}`"
                        )

                    st.error(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
```
