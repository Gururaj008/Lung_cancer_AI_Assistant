import streamlit as st
import os
import base64
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from google.api_core.exceptions import ResourceExhausted # Import for rate limit handling

# --- Page Config must be the first Streamlit command ---
st.set_page_config(
    page_title="Lung Cancer AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Load all API keys from secrets into a list
API_KEYS = [
    st.secrets.get("API_KEY_01"),
    st.secrets.get("API_KEY_02"),
    st.secrets.get("API_KEY_03"),
    st.secrets.get("API_KEY_04"),
    st.secrets.get("API_KEY_05")
]
# Filter out any keys that are not set (are None or empty strings)
API_KEYS = [key for key in API_KEYS if key]

# Validate that at least one key is present
if not API_KEYS:
    st.error("Please provide at least one GOOGLE_API_KEY in st.secrets (e.g., API_KEY_01).")
    st.stop()

EMB_MODEL = "models/embedding-001"
RAG_LLM_MODEL_NAME    = "gemini-1.5-flash-latest"
INTENT_LLM_MODEL_NAME = "gemini-1.5-flash-latest"

APP_ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_NAME = "my_faiss_index_artifact"
FAISS_INDEX_PATH = os.path.join(APP_ROOT_FOLDER, FAISS_INDEX_NAME)
BACKGROUND_IMAGE_FILENAME = "hospital.png"

EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
HYBRID_TOP_N_SEMANTIC         = 25
HYBRID_TOP_N_FOR_LLM          = 5
HYBRID_KEYWORD_BOOST_FACTOR   = 0.05
CONVERSATION_WINDOW_K         = 5
# ==============================================================================


# --- Function to load a local image as base64 (for background) ---
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
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
        .stApp {{ {background_css} background-size: cover; background-repeat: no-repeat; background-attachment: fixed; }}
        .custom-title-container {{ text-align: center !important; width: 100% !important; margin-top: 20px !important; margin-bottom: 20px !important; }}
        .custom-title-box {{ display: inline-block !important; background-color: rgba(0, 0, 0, 0.8) !important; padding: 10px 20px !important; border-radius: 5px !important; }}
        .custom-title {{ font-family: 'Agdasima', sans-serif !important; font-size: 50px !important; color: cyan !important; margin: 0 !important; }}
        div[data-testid="stChatMessage"] > div {{ background-color: rgba(40, 42, 54, 0.85); border-radius: 15px; padding: 12px 15px; margin-bottom: 10px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px); }}
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] code {{ color: #EAEAEA; }}
        div[data-testid="stChatInput"] > div {{ background-color: rgba(10, 10, 12, 0.9); border-radius: 12px; border: 1px solid #00C4FF; padding: 5px 10px; }}
        div[data-testid="stChatInput"] textarea {{ background-color: transparent !important; color: #EAEAEA !important; border: none !important; }}
        div[data-testid="stExpander"] details {{ background-color: rgba(40, 42, 54, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; }}
        div[data-testid="stExpander"] summary {{ color: #B0B0B0; }}
        .stButton>button {{ border-radius: 8px; border: 1px solid #00C4FF; background-color: rgba(0, 196, 255, 0.1); color: #00C4FF; font-weight: bold; padding: 8px 16px; }}
        .stButton>button:hover {{ background-color: rgba(0, 196, 255, 0.2); color: #00C4FF; border: 1px solid #00C4FF; }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# --- MODIFIED NLTK DOWNLOADER ---
@st.cache_data
def download_nltk_resources():
    """
    Downloads necessary NLTK data packages directly.
    The cache ensures this only runs once per container startup.
    """
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")

# Call the function to ensure resources are available
download_nltk_resources()


def pos_tag_keyword_extractor(text: str) -> set[str]:
    if not text: return set()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    return {word.lower() for word, tag in tagged_words if tag in {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'} and word.lower() not in stop_words and word.isalnum()}


@st.cache_resource
def load_faiss_artifact_cached(allow_dangerous_deserialization: bool = True) -> FAISS | None:
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index folder not found at {FAISS_INDEX_PATH}.")
        return None
    try:
        # Use the first available API key for the initial embedding model loading
        embeddings = GoogleGenerativeAIEmbeddings(model=EMB_MODEL, google_api_key=st.session_state.api_keys[0], task_type=EMBEDDING_TASK_TYPE_QUERY)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None


def hybrid_search_with_reranking_st(query_text: str, faiss_index: FAISS) -> list[Document]:
    if not faiss_index: return []
    try:
        semantic_results = faiss_index.similarity_search_with_score(query_text, k=HYBRID_TOP_N_SEMANTIC)
    except Exception as e:
        st.warning(f"Semantic search failed: {e}"); return []
    query_keywords = pos_tag_keyword_extractor(query_text)
    reranked_results = []
    for doc, score in semantic_results:
        common_keywords = len(query_keywords.intersection(set(doc.metadata.get("keywords", []))))
        rerank_score = score - (common_keywords * HYBRID_KEYWORD_BOOST_FACTOR)
        reranked_results.append({"document": doc, "score": rerank_score})
    reranked_results.sort(key=lambda x: x["score"])
    return [item["document"] for item in reranked_results[:HYBRID_TOP_N_FOR_LLM]]


def format_docs_with_sources_st(docs: list[Document]) -> str:
    formatted_strings = []
    for i, doc in enumerate(docs):
        filename = os.path.basename(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        page_num = page + 1 if isinstance(page, int) else page
        formatted_strings.append(f"Source {i+1} (File: {filename}, Page: {page_num}):\n{doc.page_content}")
    return "\n\n".join(formatted_strings)


def create_llm_with_key(api_key: str, model_name: str, temperature: float) -> ChatGoogleGenerativeAI | None:
    """Creates a ChatGoogleGenerativeAI instance using a specific API key."""
    try:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, convert_system_message_to_human=True)
    except Exception as e:
        st.warning(f"Failed to create LLM with a key. Error: {e}")
        return None

def invoke_chain_with_rotation(prompt_template, model_name, temperature, input_payload):
    """
    Invokes a LangChain chain, handling API key rotation on rate limit errors.
    Returns a tuple: (response_text, error_message).
    """
    start_index = st.session_state.current_api_key_index
    num_keys = len(st.session_state.api_keys)

    for i in range(num_keys):
        key_index_to_try = (start_index + i) % num_keys
        current_key = st.session_state.api_keys[key_index_to_try]

        try:
            llm = create_llm_with_key(current_key, model_name, temperature)
            if not llm:
                st.warning(f"Skipping invalid API key at index {key_index_to_try}.")
                continue

            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke(input_payload)

            st.session_state.current_api_key_index = key_index_to_try
            return response, None

        except ResourceExhausted:
            st.warning(f"API rate limit hit on Key #{key_index_to_try + 1}. Rotating...")
            time.sleep(1)
            continue

        except Exception as e:
            return None, f"An unexpected error occurred: {e}"

    error_msg = "All API keys are currently rate-limited. The system is under heavy load. Please try again in a minute."
    return None, error_msg


# --- Main App Logic ---
load_custom_css()

st.markdown("""
    <div class="custom-title-container">
        <div class="custom-title-box"><p class="custom-title">Lung Cancer AI Assistant</p></div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
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
        if error:
            st.error(f"Intent classification failed: {error}")
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
            with st.spinner("Searching knowledge base and generating response..."):
                retrieved_docs = hybrid_search_with_reranking_st(user_query, faiss_index)
                if not retrieved_docs:
                    ai_response_content = "I could not find specific information for your query in my knowledge base."
                else:
                    context = format_docs_with_sources_st(retrieved_docs)
                    chat_history = st.session_state.chat_memory.load_memory_variables({})['chat_history']
                    
                    rag_prompt_template_str = (
                        "You are a helpful AI assistant for lung cancer information. Your goal is to answer the user's question based *only* "
                        "on the provided 'Context from Retrieved Documents' and 'Chat History'.\n"
                        "Instructions:\n"
                        "1. Use bullet points for lists (e.g., symptoms, risk factors).\n"
                        "2. Use paragraphs for explanations.\n"
                        "3. If the context does not contain the answer, state: 'I cannot answer your question based on the provided information.'\n"
                        "4. Do not use outside knowledge.\n\n"
                        "Chat History:\n{chat_history}\n\n"
                        "Context from Retrieved Documents:\n{context}\n\n"
                        "Current Question: {question}\n\n"
                        "Answer:"
                    )
                    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template_str)
                    
                    ai_response_content, error = invoke_chain_with_rotation(
                        rag_prompt, RAG_LLM_MODEL_NAME, 0.3,
                        {"context": context, "question": user_query, "chat_history": chat_history}
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
