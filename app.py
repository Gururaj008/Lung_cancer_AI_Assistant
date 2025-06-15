import streamlit as st
import os
import base64
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from google.api_core.exceptions import ResourceExhausted

# --- Page Config must be the first Streamlit command ---
st.set_page_config(
    page_title="Lung Cancer AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEYS = [st.secrets.get(f"API_KEY_0{i}") for i in range(1, 6)]
API_KEYS = [key for key in API_KEYS if key]

if not API_KEYS:
    st.error("Please provide at least one GOOGLE_API_KEY in st.secrets (e.g., API_KEY_01).")
    st.stop()

EMB_MODEL = "models/embedding-001"
RAG_LLM_MODEL_NAME    = "gemini-1.5-flash-latest"
INTENT_LLM_MODEL_NAME = "gemini-1.5-flash-latest"
REWRITE_LLM_MODEL_NAME = "gemini-1.5-flash-latest"

APP_ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_NAME = "my_faiss_index_artifact"
FAISS_INDEX_PATH = os.path.join(APP_ROOT_FOLDER, FAISS_INDEX_NAME)
BACKGROUND_IMAGE_FILENAME = "hospital.png"

EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
TOP_N_FOR_LLM                 = 5
CONVERSATION_WINDOW_K         = 5
# ==============================================================================


@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


def load_custom_css():
    img_path = os.path.join(APP_ROOT_FOLDER, BACKGROUND_IMAGE_FILENAME)
    img_base64 = get_base64_of_bin_file(img_path)
    background_css = f"background-image: url('data:image/png;base64,{img_base64}');" if img_base64 else "background-color: #0a0b0c;"

    custom_css = f"""
    <style>
        .stApp {{
            {background_css}
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* --- CHAT MESSAGE STYLING --- */
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.88) !important;
            border-radius: 10px;
            padding: 16px !important;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
        }}
        .stChatMessage p, .stChatMessage li, .stChatMessage div[data-testid="stMarkdownContainer"] > div {{
            color: #1e1e1e !important;
            font-size: 1rem !important;
        }}

        /* --- CHAT INPUT AREA STYLING --- */
        .stChatInputContainer {{
            background-color: rgba(230, 230, 230, 0.90) !important;
            border-top: 1px solid #bbbbbb !important;
        }}
        div[data-testid="stChatInput"] textarea[aria-label="chat input"] {{
            color: #1e1e1e !important;
            background-color: rgba(255, 255, 255, 0.95) !important;
        }}

        /* --- NEW: STYLING FOR THE 'SOURCES' EXPANDER --- */
        div[data-testid="stExpander"] {{
            background-color: rgba(0, 0, 0, 0.05) !important; /* Slightly darker background */
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            border-radius: 8px !important;
            margin-top: 16px !important;
        }}
        /* Style for the 'Sources' title text */
        div[data-testid="stExpander"] summary {{
            color: #444444 !important; /* Darker grey for the title */
            font-weight: 600 !important;
        }}
        /* Style for the content inside the expander */
        div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {{
            font-size: 0.9rem !important;
            color: #333333 !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource
def load_faiss_artifact_cached(allow_dangerous_deserialization: bool = True) -> FAISS | None:
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index folder not found at {FAISS_INDEX_PATH}.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMB_MODEL, google_api_key=st.session_state.api_keys[0], task_type=EMBEDDING_TASK_TYPE_QUERY)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None


def semantic_search_st(query_text: str, faiss_index: FAISS) -> list[Document]:
    if not faiss_index: return []
    try:
        return faiss_index.similarity_search(query_text, k=TOP_N_FOR_LLM)
    except Exception as e:
        st.warning(f"Semantic search failed: {e}"); return []


def format_docs_with_sources_st(docs: list[Document]) -> str:
    formatted_strings = []
    for i, doc in enumerate(docs):
        filename = os.path.basename(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        page_num = page + 1 if isinstance(page, int) else page
        formatted_strings.append(f"Source {i+1} (File: {filename}, Page: {page_num}):\n{doc.page_content}")
    return "\n\n".join(formatted_strings)


def create_llm_with_key(api_key: str, model_name: str, temperature: float) -> ChatGoogleGenerativeAI | None:
    try:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, convert_system_message_to_human=True)
    except Exception as e:
        st.warning(f"Failed to create LLM with a key. Error: {e}"); return None

def invoke_chain_with_rotation(prompt_template, model_name, temperature, input_payload):
    start_index = st.session_state.current_api_key_index
    num_keys = len(st.session_state.api_keys)
    for i in range(num_keys):
        key_index_to_try = (start_index + i) % num_keys
        current_key = st.session_state.api_keys[key_index_to_try]
        try:
            llm = create_llm_with_key(current_key, model_name, temperature)
            if not llm: continue
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke(input_payload)
            st.session_state.current_api_key_index = key_index_to_try
            return response, None
        except ResourceExhausted:
            st.warning(f"API rate limit hit on Key #{key_index_to_try + 1}. Rotating...")
            time.sleep(1); continue
        except Exception as e:
            return None, f"An unexpected error occurred: {e}"
    return None, "All API keys are currently rate-limited. Please try again in a minute."


def rewrite_query_with_history(chat_history, user_query):
    if not chat_history:
        return user_query, None

    rewrite_prompt_template = ChatPromptTemplate.from_template(
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow Up Input: {question}\n"
        "Standalone question:"
    )
    rewritten_query, error = invoke_chain_with_rotation(
        rewrite_prompt_template, REWRITE_LLM_MODEL_NAME, 0.1,
        {"chat_history": chat_history, "question": user_query}
    )
    return rewritten_query if not error else user_query, error


# --- Main App Logic ---
load_custom_css()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
    .custom-title-container { text-align: center !important; width: 100% !important; margin-top: 20px !important; }
    .custom-title-box { display: inline-block !important; background-color: rgba(0, 0, 0, 0.8) !important; padding: 10px 20px !important; border-radius: 5px !important; }
    .custom-title { font-family: 'Agdasima', sans-serif !important; font-size: 50px !important; color: cyan !important; margin: 0 !important; }
    </style>
    <div class="custom-title-container"><div class="custom-title-box"><p class="custom-title">Lung Cancer AI Assistant</p></div></div>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant for lung cancer information. How can I help you today?"}]
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferWindowMemory(k=CONVERSATION_WINDOW_K, memory_key="chat_history", input_key="question", return_messages=True)
if "api_keys" not in st.session_state:
    st.session_state.api_keys = API_KEYS
    st.session_state.current_api_key_index = 0

faiss_index = load_faiss_artifact_cached()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                st.markdown(message["sources"])

if st.button("Start New Session"):
    st.session_state.messages = [{"role": "assistant", "content": "New session started. How can I help you with lung cancer information?"}]
    st.session_state.chat_memory.clear()
    st.session_state.current_api_key_index = 0
    st.rerun()

if user_query := st.chat_input("Ask a question about lung cancer..."):
    if not faiss_index:
        st.error("The knowledge base (FAISS index) failed to load. The assistant cannot continue.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        INTENT_LABELS = ["GREETING_SMALLTALK", "LUNG_CANCER_QUERY", "OUT_OF_SCOPE", "EXIT"]
        intent_prompt = ChatPromptTemplate.from_template(
            "Your task is to classify the user's intent. Respond with ONLY one of the following labels: "
            f"{', '.join(INTENT_LABELS)}\nUser input: \"{{user_query}}\"\nClassification:"
        )
        classification_result, error = invoke_chain_with_rotation(
            intent_prompt, INTENT_LLM_MODEL_NAME, 0.1, {"user_query": user_query}
        )
        
        intent = "LUNG_CANCER_QUERY"
        if error: st.error(f"Intent classification failed: {error}")
        elif classification_result and classification_result.strip().upper() in INTENT_LABELS:
            intent = classification_result.strip().upper()

        ai_response_content = ""
        ai_response_sources = ""

        if intent == "GREETING_SMALLTALK":
            ai_response_content = "Hello! How can I assist you with information about lung cancer today?"
        elif intent == "EXIT":
            ai_response_content = "Thank you for using the assistant. Goodbye!"
        elif intent == "OUT_OF_SCOPE":
            ai_response_content = "I apologize, but my expertise is limited to lung cancer. I cannot answer questions on other topics."
        elif intent == "LUNG_CANCER_QUERY":
            with st.spinner("Thinking..."):
                chat_history = st.session_state.chat_memory.load_memory_variables({})['chat_history']
                
                standalone_query, error = rewrite_query_with_history(chat_history, user_query)
                if error: st.warning(f"Query rewriting failed: {error}. Using original query.")

                retrieved_docs = semantic_search_st(standalone_query, faiss_index)
                if not retrieved_docs:
                    ai_response_content = "I could not find specific information for your query in my knowledge base."
                else:
                    context = format_docs_with_sources_st(retrieved_docs)
                    
                    rag_prompt_template_str = (
                        "You are an expert AI assistant for lung cancer information. Your goal is to synthesize a helpful and accurate answer for the user based *only* on the provided 'Context from Retrieved Documents'.\n\n"
                        "Instructions:\n"
                        "1. First, directly address the user's 'Current Question'. Start your response with a clear, direct statement (e.g., 'Yes, according to the provided sources...', 'The documents suggest that...', 'The primary causes listed are...').\n"
                        "2. After the direct statement, provide a comprehensive answer by synthesizing information from all relevant sources in the context. Do not just copy-paste the sources.\n"
                        "3. Use bullet points for lists (e.g., symptoms, risk factors) to improve readability.\n"
                        "4. Do not use any outside knowledge. Your response must be grounded in the provided text.\n"
                        "5. If the context does not contain information to answer the question, state: 'Based on the provided documents, I cannot find specific information about your question.'\n\n"
                        "Context from Retrieved Documents:\n{context}\n\n"
                        "Current Question: {question}\n\n"
                        "Answer:"
                    )
                    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template_str)
                    
                    ai_response_content, error = invoke_chain_with_rotation(
                        rag_prompt, RAG_LLM_MODEL_NAME, 0.3,
                        {"context": context, "question": standalone_query}
                    )
                    if error:
                        st.error(f"Response generation failed: {error}")
                        ai_response_content = "Sorry, I encountered an error while generating the response."
                    else:
                        ai_response_sources = context

        st.session_state.chat_memory.save_context({"question": user_query}, {"answer": ai_response_content})
        
        response_message = {"role": "assistant", "content": ai_response_content}
        if ai_response_sources:
            response_message["sources"] = ai_response_sources
        st.session_state.messages.append(response_message)

        with st.chat_message("assistant"):
            st.markdown(ai_response_content)
            if ai_response_sources:
                with st.expander("Sources"):
                    st.markdown(ai_response_sources)
        
        if intent == "EXIT":
            time.sleep(2)
            st.stop()
