
import os
import ast
import operator
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

try:
    API_KEY = st.secrets.get(
        "XAI_API_KEY",
        os.getenv("XAI_API_KEY", ""),
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
6. Adapt your answer to the requested response style.
7. Use earlier messages in the current conversation.
8. If uncertain, explain the uncertainty.
9. Never claim you executed code unless you actually did.
10. Help students learn rather than simply memorize answers.
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
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_calculate(expression):
    """Evaluate basic arithmetic without using eval()."""

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Operator not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = operation(left, right)

            if not isinstance(result, (int, float)):
                raise ValueError("Invalid calculation.")

            if abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unary operator not allowed.")

            return operation(evaluate(node.operand))

        raise ValueError("Invalid mathematical expression.")

    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("Enter a shorter mathematical expression.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)

# --------------------------------------------------
# 5. PAGE HEADER
# --------------------------------------------------
st.title("🎓 Student AI Assistant")
st.caption("Your AI-powered learning companion, powered by Grok.")

with st.sidebar:
    st.header("⚙️ Settings")

    response_style = st.selectbox(
        "Response Style",
        ["Simple", "Detailed", "Exam Answer"],
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Model: {MODEL_NAME}")

# --------------------------------------------------
# 6. INITIALIZE CHAT HISTORY
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# 7. DISPLAY CHAT HISTORY
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# 8. CHAT INPUT AND GROK RESPONSE
# --------------------------------------------------
user_input = st.chat_input(
    "Ask a question about programming, computer science, or your studies..."
)

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    if not API_KEY:
        with st.chat_message("assistant"):
            st.error(
                "Grok API key is missing. Add XAI_API_KEY "
                "in Streamlit Cloud → Settings → Secrets."
            )
    else:
        style_instructions = {
            "Simple": "Explain in simple language with a clear example.",
            "Detailed": "Provide a detailed explanation with examples and steps.",
            "Exam Answer": (
                "Give a well-organized exam-style answer with a definition, "
                "key points, an example, and a conclusion where appropriate."
            ),
        }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n\nResponse style: "
                + style_instructions[response_style],
            }
        ]

        messages.extend(
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in st.session_state.messages
        )

        try:
            with st.chat_message("assistant"):
                with st.spinner("Grok is thinking..."):
                    client = get_client(API_KEY)

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.7,
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = "Grok returned an empty response. Please try again."

                    st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            with st.chat_message("assistant"):
                st.error(
                    "Unable to get a response from Grok. "
                    "Check your API key, model access, and internet connection."
                )
                st.caption(f"Error details: {error}")

# --------------------------------------------------
# 9. EXPORT CONVERSATION
# --------------------------------------------------
if st.session_state.messages:
    export_text = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in st.session_state.messages
    )

    st.sidebar.download_button(
        label="📥 Download Conversation",
        data=StringIO(export_text).getvalue(),
        file_name="student_ai_conversation.txt",
        mime="text/plain",
        use_container_width=True,
    )

