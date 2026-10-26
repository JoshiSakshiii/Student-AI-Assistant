import os
import ast
import operator
from io import StringIO

import streamlit as st
import ollama
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
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# --------------------------------------------------
# 2. SYSTEM PROMPT: PROMPT ENGINEERING
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
# 3. INITIALIZE LOCAL OLLAMA CLIENT
# --------------------------------------------------
@st.cache_resource
def get_client(host):
    return ollama.Client(host=host)


# --------------------------------------------------
# 4. SAFE CALCULATOR TOOL
# --------------------------------------------------
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def safe_calculate(expression):
    """Evaluate basic arithmetic safely without using eval()."""
    if not isinstance(expression, str) or len(expression) > 100:
        raise ValueError("Expression must be text of at most 100 characters.")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numbers are allowed.")
            if abs(node.value) > 1e12:
                raise ValueError("Number is too large.")
            return node.value

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)
            if operator_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operator_type is ast.Pow and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = ALLOWED_OPERATORS[operator_type](left, right)
            if not isinstance(result, (int, float)) or abs(result) > 1e100:
                raise ValueError("Result is too large.")
            return result

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)
            if operator_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")
            return ALLOWED_OPERATORS[operator_type](evaluate(node.operand))

        raise ValueError("Invalid arithmetic expression.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. 25 * 4 + 10.",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


# --------------------------------------------------
# 5. GENERATE RESPONSE AND HANDLE TOOL CALLING
# --------------------------------------------------
def generate_response(client, messages, response_style):
    style_instructions = {
        "Simple": "Use simple language and short explanations.",
        "Detailed": "Provide a detailed, structured explanation.",
        "Exam Answer": (
            "Use an academic, exam-oriented answer with a definition, "
            "important points, and a short conclusion where appropriate."
        ),
    }
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
        + "\n\nResponse style: "
        + style_instructions[response_style],
    }

    # Make a separate working copy so tool-call messages do not pollute
    # the Streamlit conversation history.
    working_messages = [system_message] + [
        {"role": item["role"], "content": item["content"]}
        for item in messages
        if item.get("role") in ("user", "assistant")
    ]

    for _ in range(4):
        response = client.chat(
            model=MODEL_NAME,
            messages=working_messages,
            tools=OLLAMA_TOOLS,
        )
        assistant_message = response.message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return assistant_message.content or ""

        # Include the assistant's tool-call message in the next request.
        working_messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments or {}

            if function_name != "calculate":
                tool_result = "Unsupported tool."
            else:
                try:
                    expression = arguments.get("expression", "")
                    tool_result = str(safe_calculate(expression))
                except Exception as error:
                    tool_result = f"Calculation error: {error}"

            working_messages.append({
                "role": "tool",
                "name": function_name,
                "content": tool_result,
            })

    return (
        "The assistant reached its calculator tool-call limit. "
        "Please simplify your request and try again."
    )


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------
st.sidebar.title("🎓 Student AI Assistant")
st.sidebar.write(
    "A local AI learning assistant with conversation memory "
    "and calculator tool calling."
)

response_style = st.sidebar.selectbox(
    "Response Style",
    ["Simple", "Detailed", "Exam Answer"],
)
st.sidebar.caption(f"Local model: {MODEL_NAME}")
st.sidebar.caption("Powered by Ollama — no OpenAI API credits required.")

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Features")
st.sidebar.write("✅ Local LLM integration")
st.sidebar.write("✅ Conversational memory")
st.sidebar.write("✅ Prompt engineering")
st.sidebar.write("✅ Calculator tool calling")
st.sidebar.write("✅ Conversation export")


# --------------------------------------------------
# 7. MAIN INTERFACE
# --------------------------------------------------
st.title("🤖 Student AI Assistant")
st.write(
    "Ask questions about programming, artificial intelligence, "
    "computer science, mathematics, or other study topics."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

client = get_client(OLLAMA_HOST)


# --------------------------------------------------
# 8. DISPLAY CONVERSATION HISTORY
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 9. CHAT INPUT AND RESPONSE GENERATION
# --------------------------------------------------
question = st.chat_input("Ask your question here...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {MODEL_NAME}..."):
            try:
                answer = generate_response(
                    client,
                    st.session_state.messages,
                    response_style,
                )
                if not answer:
                    answer = "No text response was returned."

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                st.error(
                    "Unable to connect to Ollama. Check that Ollama is "
                    "running and that the selected model is downloaded."
                )
                st.caption(f"Technical details: {error}")
                # Remove the unanswered question from conversation history.
                st.session_state.messages.pop()


# --------------------------------------------------
# 10. EXPORT CONVERSATION
# --------------------------------------------------
if st.session_state.messages:
    conversation = StringIO()
    conversation.write("STUDENT AI ASSISTANT\n")
    conversation.write("=" * 40 + "\n\n")

    for message in st.session_state.messages:
        role = message["role"].upper()
        conversation.write(f"{role}:\n")
        conversation.write(message["content"] + "\n\n")

    st.sidebar.download_button(
        label="📥 Download Conversation",
        data=conversation.getvalue(),
        file_name="student_ai_conversation.txt",
        mime="text/plain",
    )

st.caption(
    "Educational AI assistant. Verify important information "
    "before using it in assignments or projects."
)
