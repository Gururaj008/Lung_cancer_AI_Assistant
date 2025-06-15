import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import base64
import time

# --- Configuration ---
API_KEYS = [
    st.secrets.get("API_KEY_01"),
    st.secrets.get("API_KEY_02"),
    st.secrets.get("API_KEY_03"),
    st.secrets.get("API_KEY_04"),
    st.secrets.get("API_KEY_05")
]
API_KEYS = [key for key in API_KEYS if key]

MODEL_NAME = st.secrets["MODEL_NAME"]
BACKGROUND_IMAGE_PATH = "garage.jpeg"

if not API_KEYS:
    st.error("Please provide at least one GOOGLE_API_KEY in st.secrets (e.g., API_KEY_01).")
    st.stop()

# --- Agent and Tool Factory Function ---
def get_agent_executor_with_key(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    try:
        genai.configure(api_key=api_key)
        client = genai
    except Exception as e:
        st.warning(f"Error configuring Google Generative AI with a key: {e}")
        return None

    @tool
    def greet_tool(_input: str = "") -> str:
        """Use this tool for greetings or conversation initiation."""
        res = """
        Welcome to Maverick’s IntelliTune Garage!
        * I am your AI service assistant.
        * How can I help with your vehicle today?
        * Type 'help' for available services or 'exit' to quit.
        """
        return res

    @tool
    def search_engine_problems(query: str) -> str:
        """Use this tool to analyze engine-related complaints."""
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"You are Maverick’s IntelliTune Garage AI. User says: \"{query}\". Respond with 3-5 concise bullet points on possible causes or checks. End with: \"Please contact us to get this fixed or for more info.\""
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def schedule_service(query: str) -> str:
        """Use this tool for scheduling or maintenance interval questions."""
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"You are Maverick’s IntelliTune Garage AI. User query: \"{query}\". Respond with 3-5 concise bullet points on recommended maintenance. End with: \"Please contact us to get this fixed or for more info.\""
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def assess_damage(query: str) -> str:
        """Use this tool for accident damage descriptions."""
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"You are Maverick’s IntelliTune Garage AI. User says: \"{query}\". Respond with 3-5 concise bullet points assessing potential damage or advice. End with: \"Please contact us to get this fixed or for more info.\""
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def routine_service(query: str) -> str:
        """Use this tool for routine service check questions."""
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"You are Maverick’s IntelliTune Garage AI. User asks: \"{query}\". Respond with 3-5 concise bullet points on routine checks. End with: \"Please contact us to get this fixed or for more info.\""
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def contact_info(_input: str = "") -> str:
        """Use this tool when asked for contact details."""
        address = """
        *   **Address:** Maverick’s IntelliTune Garage, Hesaraghatta Main Road, Bengaluru
        *   **Hours:** 10 AM – 6 PM (Weekdays)
        *   **Phone:** +91 98765 00000
        *   **Website:** www.intellitune.com
        *   **Email:** intellitune@tuning.com
        Please contact us for appointments or further information.
        """
        return address

    tools = [greet_tool, search_engine_problems, schedule_service, assess_damage, routine_service, contact_info]

    system_prompt_text = "You are Maverick Agentic AI, a helpful AI service assistant for Maverick’s IntelliTune Garage. You have access to tools. Based on the user's message, decide if a tool is appropriate. If so, use it. If the user says hi, hello, or starts the conversation, use greet_tool. If asked for help, list the types of queries you can handle based on your tools. For goodbyes, respond with a polite closing."
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm_agent = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, client=client)
    agent = create_tool_calling_agent(llm_agent, tools, agent_prompt)
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", k=20, return_messages=True)

    executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=False, handle_parsing_errors=True)
    return executor


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
    .custom-title-container { text-align: center !important; width: 100% !important; margin-top: 20px !important; }
    .custom-title-box { display: inline-block !important; background-color: rgba(0, 0, 0, 0.8) !important; padding: 10px 20px !important; border-radius: 5px !important; }
    .custom-title { font-family: 'Agdasima', sans-serif !important; font-size: 50px !important; color: cyan !important; margin: 0 !important; }
    </style>
    <div class="custom-title-container"><div class="custom-title-box"><p class="custom-title">Agentic AI for Maverick's IntelliTune Garage</p></div></div>
""", unsafe_allow_html=True)

def set_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png" if image_file.endswith(".png") else "jpg"};base64,{encoded_string});
                background-size: cover; background-repeat: no-repeat; background-attachment: fixed;
            }}

            /* --- NEW CHAT MESSAGE STYLING --- */
            .stChatMessage {{
                background-color: rgba(240, 242, 246, 0.95) !important; /* Off-white background */
                border-radius: 15px;
                padding: 16px !important;
                margin-bottom: 12px;
                border: 1px solid #e0e0e0;
            }}
            /* Ensure text inside is dark and readable */
            .stChatMessage p, .stChatMessage li, .stChatMessage div[data-testid="stMarkdownContainer"] > div {{
                color: #1e1e1e !important; /* Dark text */
                font-size: 1rem !important;
            }}

            /* --- NEW AVATAR STYLING --- */
            /* Target the container for the assistant (AI) avatar */
            div[data-testid="chat-avatar-assistant"] div[data-testid="stChatAvatar"] {{
                background-color: #ffc107; /* Yellow */
            }}
            /* Target the SVG icon inside the assistant avatar to make it dark */
            div[data-testid="chat-avatar-assistant"] div[data-testid="stChatAvatar"] svg {{
                fill: #1e1e1e; /* Dark icon */
            }}

            /* Target the container for the user avatar */
            div[data-testid="chat-avatar-user"] div[data-testid="stChatAvatar"] {{
                background-color: #ff4b4b; /* Red */
            }}
            /* Target the SVG icon inside the user avatar to make it white */
            div[data-testid="chat-avatar-user"] div[data-testid="stChatAvatar"] svg {{
                fill: white; /* White icon */
            }}

            /* --- CHAT INPUT AREA STYLING (Unchanged) --- */
            .stChatInputContainer {{
                background-color: rgba(230, 230, 230, 0.90) !important;
                border-top: 1px solid #bbbbbb !important;
            }}
            div[data-testid="stChatInput"] textarea[aria-label="chat input"] {{
                color: #1e1e1e !important;
                background-color: rgba(255, 255, 255, 0.95) !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image '{image_file}' not found. Using default background.")

set_bg_from_local(BACKGROUND_IMAGE_PATH)

# --- Session State Initialization ---
if "session_active" not in st.session_state:
    st.session_state.api_keys = API_KEYS
    st.session_state.current_api_key_index = 0
    st.session_state.agent_executor = None
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", k=20, return_messages=True)
    st.session_state.session_active = False

def initialize_chat():
    if not st.session_state.session_active:
        st.session_state.messages = []
        st.session_state.memory.clear()
        
        current_key = st.session_state.api_keys[st.session_state.current_api_key_index]
        st.session_state.agent_executor = get_agent_executor_with_key(current_key)
        
        if not st.session_state.agent_executor:
             st.error("Failed to initialize the agent with the first API key. Please check your keys.")
             st.stop()

        try:
            with st.spinner("AgenticAI is starting up..."):
                initial_greeting_response = st.session_state.agent_executor.invoke(
                    {"input": "User has just started the chat, greet them."})
            assistant_response = initial_greeting_response.get("output", "Sorry, I couldn't start up correctly.")
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            st.error(f"Error during initial greeting from agent: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "I seem to be having trouble starting. Please try refreshing."})
        
        st.session_state.session_active = True
        st.rerun()

# --- Main Chat Logic ---
if not st.session_state.session_active:
    initialize_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help with your vehicle today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AgenticAI is thinking..."):
            response_content = ""
            if prompt.lower() in {"exit", "quit"}:
                response_content = "Goodbye! We look forward to helping you again."
                st.session_state.session_active = False
            elif prompt.lower() == "help":
                response_content = """
                I can help with:
                *   Analyzing engine complaints
                *   Scheduling services or asking about maintenance
                *   Assessing accident damage
                *   Answering routine service questions
                *   Providing our contact information
                How can I assist you?
                """
            else:
                response = None
                start_index = st.session_state.current_api_key_index
                num_keys = len(st.session_state.api_keys)

                for i in range(num_keys):
                    key_index_to_try = (start_index + i) % num_keys
                    try:
                        st.session_state.agent_executor = get_agent_executor_with_key(st.session_state.api_keys[key_index_to_try])
                        if not st.session_state.agent_executor:
                            st.warning(f"Skipping an invalid API key at index {key_index_to_try}.")
                            continue

                        response = st.session_state.agent_executor.invoke({"input": prompt})
                        st.session_state.current_api_key_index = key_index_to_try
                        break
                    except ResourceExhausted:
                        st.warning(f"API rate limit hit on Key #{key_index_to_try + 1}. Rotating...")
                        time.sleep(1)
                        continue
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        response_content = "I'm having some trouble processing that. Please try rephrasing."
                        break

                if response:
                    response_content = response.get("output", "Sorry, I didn't quite understand that.")
                elif not response_content:
                    st.error("All API keys are currently rate-limited. The system is under heavy load.")
                    response_content = "I'm currently experiencing very high traffic and can't process your request. Please try again in a minute."

            st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.markdown(response_content)
            if not st.session_state.session_active:
                st.rerun()

# --- Session Reset Buttons ---
if not st.session_state.session_active:
    if st.button("Start New Session", key="main_reset_button"):
        st.session_state.current_api_key_index = 0
        st.rerun()
elif st.button("Start New Session", key="manual_reset_button_active_session"):
    st.session_state.session_active = False
    st.session_state.current_api_key_index = 0
    st.rerun()
