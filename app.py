import streamlit as st
import os
import base64
import time
import nltk

# --- NLTK Data Path Configuration (MUST BE NEAR THE TOP) ---
APP_ROOT_FOLDER_FOR_NLTK = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = os.path.join(APP_ROOT_FOLDER_FOR_NLTK, 'nltk_data')

# This print is for debugging deployment path issues for NLTK
print(f"DEBUG: Attempting to use NLTK data path: {NLTK_DATA_PATH}")
print(f"DEBUG: nltk.data.path before modification: {nltk.data.path}")

if os.path.exists(NLTK_DATA_PATH):
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_PATH) # Insert at the beginning to prioritize
        print(f"DEBUG: Added custom NLTK data path: {NLTK_DATA_PATH}")
else:
    print(f"WARNING: Packaged NLTK data path not found: {NLTK_DATA_PATH}. NLTK functions might fail.")
print(f"DEBUG: nltk.data.path after modification: {nltk.data.path}")
# --- END NLTK Data Path Configuration ---

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import getpass # Keep for local API key input if needed, though secrets are preferred

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Try to get API_KEY from Streamlit secrets first, then environment variable
# This variable `API_KEY_CONFIG` is used to set os.environ["GOOGLE_API_KEY"]
API_KEY_CONFIG = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

if not API_KEY_CONFIG:
    # This is a fallback for local development if secrets/env var are not set.
    # In production on Streamlit Cloud, API_KEY_CONFIG should come from st.secrets.
    print("WARNING: GOOGLE_API_KEY not found in Streamlit secrets or environment variables.")
    # For local testing, you might uncomment the getpass line or hardcode temporarily.
    # API_KEY_CONFIG = getpass.getpass("Enter your Google API key for local testing: ")
    API_KEY_CONFIG = "YOUR_FALLBACK_API_KEY_FOR_LOCAL_TESTING_ONLY" # Replace or remove for deployment
    if API_KEY_CONFIG == "YOUR_FALLBACK_API_KEY_FOR_LOCAL_TESTING_ONLY":
        st.sidebar.warning("⚠️ API Key not set via secrets/env. Using placeholder. App may not function.")

# Set the environment variable for LangChain libraries to pick up
if API_KEY_CONFIG and API_KEY_CONFIG != "YOUR_FALLBACK_API_KEY_FOR_LOCAL_TESTING_ONLY":
    os.environ["GOOGLE_API_KEY"] = API_KEY_CONFIG
elif not os.getenv("GOOGLE_API_KEY") and API_KEY_CONFIG == "YOUR_FALLBACK_API_KEY_FOR_LOCAL_TESTING_ONLY":
    # This case means no proper key was found.
    # The app will likely fail when initializing LLMs if this placeholder is used.
    pass


# FOLDER_PATH should point to where the FAISS index SUBFOLDER is located
# If FAISS index subfolder is in the app root:
FOLDER_PATH = APP_ROOT_FOLDER_FOR_NLTK # Assumes FAISS subfolder is in the app root

EMB_MODEL = "models/embedding-001"
RAG_LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Example: "gemini-1.5-flash-latest" or "gemini-pro"
INTENT_LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Example: "gemini-1.5-flash-latest" or "gemini-pro"

FAISS_INDEX_NAME = "my_faiss_index_artifact" # Your FAISS subfolder name
FAISS_INDEX_PATH = os.path.join(FOLDER_PATH, FAISS_INDEX_NAME)

EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
HYBRID_TOP_N_SEMANTIC = 25
HYBRID_TOP_N_FOR_LLM = 5
HYBRID_KEYWORD_BOOST_FACTOR = 0.05
HYBRID_NO_KEYWORD_MATCH_PENALTY = 0.1
CONVERSATION_WINDOW_K = 5
BACKGROUND_IMAGE_FILENAME = "hospital.png"
# ==============================================================================


# --- Helper Functions ---
@st.cache_data
def get_base64_of_bin_file(bin_file_path): # Takes full path
    try:
        with open(bin_file_path, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

def load_custom_css():
    image_path_for_css = os.path.join(APP_ROOT_FOLDER_FOR_NLTK, BACKGROUND_IMAGE_FILENAME)
    img_base64 = get_base64_of_bin_file(image_path_for_css)
    if img_base64:
        background_css = f"background-image: url('data:image/png;base64,{img_base64}');"
    else:
        st.sidebar.warning(f"BG image '{BACKGROUND_IMAGE_FILENAME}' not found at '{image_path_for_css}'. Using default BG.")
        background_css = "background-color: #0a0b0c;"
    custom_css = f"""
    <style>
        .stApp {{ {background_css} background-size: cover; background-repeat: no-repeat; background-attachment: fixed; }}
        .custom-title-container {{ display: flex; justify-content: center; margin-bottom: 20px; margin-top: 15px;}}
        .custom-title {{ background-color: rgba(10, 10, 12, 0.85); width: 70%; max-width: 800px; padding: 8px 20px; border-radius: 12px; text-align: center; border: 1px solid #00C4FF; box-shadow: 0 0 15px rgba(0, 196, 255, 0.3);}}
        .custom-title h1 {{ color: #00C4FF; font-size: 1.6em; font-weight: bold; text-shadow: 0 0 6px #00C4FF, 0 0 10px #00C4FF; margin: 0; line-height: 1.2;}}
        div[data-testid="stChatMessage"] > div {{ background-color: rgba(40, 42, 54, 0.85); border-radius: 15px; padding: 12px 15px; margin-bottom: 10px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px);}}
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] code {{ color: #EAEAEA; }}
        div[data-testid="stChatInput"] > div {{ background-color: rgba(10, 10, 12, 0.9); border-radius: 12px; border: 1px solid #00C4FF; padding: 5px 10px;}}
        div[data-testid="stChatInput"] textarea {{ background-color: transparent !important; color: #EAEAEA !important; border: none !important;}}
        div[data-testid="stExpander"] details {{ background-color: rgba(40, 42, 54, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;}}
        div[data-testid="stExpander"] summary {{ color: #B0B0B0; }}
        .stButton>button {{ border-radius: 8px; border: 1px solid #00C4FF; background-color: rgba(0, 196, 255, 0.1); color: #00C4FF; font-weight:bold; padding: 8px 16px;}}
        .stButton>button:hover {{ background-color: rgba(0, 196, 255, 0.2); color: #00C4FF; border: 1px solid #00C4FF;}}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_data
def verify_nltk_resources(): # Renamed from download_nltk_resources_if_needed
    """Verifies NLTK resources, assuming they are packaged."""
    # This function now primarily serves to check if NLTK can find its data.
    # The actual download should happen locally and files be part of the deployment.
    resources_to_check = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }
    all_found = True
    for resource_path, resource_name in resources_to_check.items():
        try:
            nltk.data.find(resource_path) # Will use nltk.data.path
        except LookupError:
            st.error(f"Packaged NLTK resource missing: '{resource_name}'. Ensure 'nltk_data' folder is deployed correctly.")
            all_found = False
    if not all_found:
        st.warning("Essential NLTK data packages missing. App functionality may be impaired.")
    return all_found


def pos_tag_keyword_extractor(text: str, num_keywords=0, target_pos_tags={'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}) -> set:
    if not text: return set()
    words_from_text = []
    try:
        for sentence in nltk.sent_tokenize(text): words_from_text.extend(nltk.word_tokenize(sentence))
    except LookupError: # This might happen if punkt is still not found despite checks
        st.warning("NLTK 'punkt' tokenizer data not found. Keyword extraction may be basic.")
        words_from_text.extend(nltk.word_tokenize(text)) # Fallback
    try:
        stop_words_set = set(stopwords.words('english'))
        meaningful_words = [w.lower() for w in words_from_text if w.isalnum() and w.lower() not in stop_words_set and len(w) > 2]
        if not meaningful_words: return set()
        tagged_meaningful_words = nltk.pos_tag(meaningful_words)
    except LookupError: # If stopwords or tagger data is missing
        st.warning("NLTK 'stopwords' or 'averaged_perceptron_tagger' data not found. Keyword extraction may be incomplete.")
        return set(meaningful_words[:num_keywords]) if num_keywords > 0 and meaningful_words else set(meaningful_words)
    extracted_keywords_list = [word for word, tag in tagged_meaningful_words if tag in target_pos_tags]
    if num_keywords > 0 and extracted_keywords_list:
        return set(word for word, count in Counter(extracted_keywords_list).most_common(num_keywords))
    return set(extracted_keywords_list)

@st.cache_resource
def load_faiss_artifact_cached(allow_dangerous_deserialization=True) -> FAISS | None:
    faiss_file = os.path.join(FAISS_INDEX_PATH, "index.faiss") # FAISS_INDEX_PATH is the folder
    pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")
    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        st.error(f"FAISS index files ('index.faiss', 'index.pkl') not found in {FAISS_INDEX_PATH}.")
        return None
    try:
        # The API key for embeddings is taken from the environment variable set at the start
        query_embeddings_model = GoogleGenerativeAIEmbeddings(model=EMB_MODEL, task_type=EMBEDDING_TASK_TYPE_QUERY)
        index = FAISS.load_local(FAISS_INDEX_PATH, query_embeddings_model, allow_dangerous_deserialization=allow_dangerous_deserialization)
        return index
    except Exception as e: st.error(f"Error loading FAISS artifact: {e}"); return None

def hybrid_search_with_reranking_st(
    query_text: str, faiss_index: FAISS,
    top_n_semantic: int = HYBRID_TOP_N_SEMANTIC, top_n_final_for_llm: int = HYBRID_TOP_N_FOR_LLM,
    keyword_boost_factor: float = HYBRID_KEYWORD_BOOST_FACTOR, no_match_penalty: float = HYBRID_NO_KEYWORD_MATCH_PENALTY
) -> list[Document]:
    # (Your working hybrid_search_with_reranking function - no changes needed here)
    if not faiss_index: return []
    try:
        semantic_results_with_scores = faiss_index.similarity_search_with_score(query_text, k=top_n_semantic)
    except Exception as e: st.warning(f"Semantic search error: {e}"); return []
    if not semantic_results_with_scores: return []
    query_keywords = pos_tag_keyword_extractor(query_text, num_keywords=0)
    reranked_candidates = []
    for doc, semantic_score in semantic_results_with_scores:
        doc_metadata_keywords = set(doc.metadata.get('keywords', []))
        common_keywords = query_keywords.intersection(doc_metadata_keywords)
        keyword_match_count = len(common_keywords)
        combined_score = semantic_score
        if keyword_match_count > 0: combined_score -= (keyword_match_count * keyword_boost_factor)
        else: combined_score += no_match_penalty
        reranked_candidates.append({"document": doc, "combined_score": combined_score})
    reranked_candidates.sort(key=lambda x: x["combined_score"])
    return [cand["document"] for cand in reranked_candidates[:top_n_final_for_llm]]


def format_docs_with_sources_st(docs: list[Document]) -> str:
    # (Your working format_docs_with_sources function - no changes needed here)
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A'); page = doc.metadata.get('page', 'N/A')
        filename = os.path.basename(source) if source != 'N/A' else 'N/A'
        page_display = page + 1 if isinstance(page, int) else page if page is not None else 'N/A'
        formatted_docs.append(f"Source {i+1} (File: {filename}, Page: {page_display}):\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

INTENT_LABELS = ["GREETING_SMALLTALK", "LUNG_CANCER_QUERY", "OUT_OF_SCOPE", "EXIT"]
@st.cache_resource
def get_llm(_model_name, temperature): # Removed _api_key, relies on os.environ
    try:
        if not os.getenv("GOOGLE_API_KEY"): # Check if API key is set in environment
             st.error(f"GOOGLE_API_KEY not available for LLM ({_model_name}).")
             return None
        llm = ChatGoogleGenerativeAI(model=_model_name, temperature=temperature, convert_system_message_to_human=True)
        return llm
    except Exception as e: st.error(f"Error initializing LLM ({_model_name}): {e}"); return None

def classify_intent_with_llm_st(user_query: str, _intent_llm: ChatGoogleGenerativeAI) -> str:
    # (Your working classify_intent_with_llm function - no changes needed here)
    if not _intent_llm: return "LUNG_CANCER_QUERY"
    prompt_template = f"""Your task is to classify the user's intent based on their input.
Please respond with ONLY one of the following labels: {', '.join(INTENT_LABELS)}
User input: "{{user_query}}"
Classification:"""
    intent_prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = intent_prompt | _intent_llm | StrOutputParser()
    try:
        classification = chain.invoke({"user_query": user_query}).strip().upper()
        if classification in INTENT_LABELS: return classification
        else: return "LUNG_CANCER_QUERY" # Default if classification is unexpected
    except Exception as e: st.warning(f"Intent classification error: {e}"); return "LUNG_CANCER_QUERY"


# --- Streamlit App UI and Logic ---
st.set_page_config(page_title="Lung Cancer AI Assistant", layout="wide", initial_sidebar_state="collapsed")
load_custom_css()

st.markdown("<div class='custom-title-container'><div class='custom-title'><h1>Lung Cancer AI Assistant</h1></div></div>", unsafe_allow_html=True)

# Verify NLTK resources (now primarily checks if packaged data is found)
nltk_ready = verify_nltk_resources() # Store the result of verification

# Load FAISS index and LLMs
# These are cached, so they only run once per session if inputs don't change.
# The API key is now expected to be in os.environ from the top of the script.
faiss_index = load_faiss_artifact_cached()
rag_llm = get_llm(RAG_LLM_MODEL_NAME, temperature=0.3)
intent_llm = get_llm(INTENT_LLM_MODEL_NAME, temperature=0.1)

# Initialize chat history and memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant for lung cancer information. How can I help you today?"}
    ]
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferWindowMemory(
        k=CONVERSATION_WINDOW_K, memory_key="chat_history_str",
        input_key="question", return_messages=False
    )

# Display chat messages
for message in st.session_state.messages:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Supporting Evidence (Sources)", expanded=False):
                unique_sources_to_display_keys = set()
                sources_to_render = []
                for src in message["sources"]:
                    src_key = (src['file'], src['page'])
                    if src_key not in unique_sources_to_display_keys:
                        sources_to_render.append(src)
                        unique_sources_to_display_keys.add(src_key)
                for i, src_info in enumerate(sources_to_render):
                     st.markdown(f"**Source {i+1}:** File: {src_info['file']}, Page: {src_info['page']}")

if st.button("Start New Session", key="start_new_session_main"):
    st.session_state.messages = [
         {"role": "assistant", "content": "Hello! I am your AI assistant for lung cancer information. How can I help you today?"}
    ]
    if "chat_memory" in st.session_state: st.session_state.chat_memory.clear()
    st.rerun()

if user_query := st.chat_input("Ask a question about lung cancer..."):
    # --- CORRECTED API KEY AND COMPONENT CHECK ---
    # Check if the API key was effectively set for LangChain to use
    effective_api_key = os.getenv("GOOGLE_API_KEY")
    if not effective_api_key or effective_api_key == "YOUR_FALLBACK_API_KEY_FOR_LOCAL_TESTING_ONLY":
        st.error("Google API Key is not properly configured. Please set it via Streamlit secrets or environment variables.")
    elif not faiss_index or not rag_llm or not intent_llm:
        st.error("A critical component (FAISS index or LLM) failed to load. Cannot proceed. Check logs during startup.")
    elif not nltk_ready: # Check if NLTK resources are okay
        st.error("Essential NLTK data is missing. Please ensure the 'nltk_data' folder is correctly deployed.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💻"): st.markdown(user_query)
        
        ai_response_text = ""; ai_response_sources = []
        intent = classify_intent_with_llm_st(user_query, intent_llm)

        if intent == "GREETING_SMALLTALK":
            ai_response_text = "Hello again! How can I assist you with lung cancer information?"
            st.session_state.chat_memory.save_context({"question": user_query}, {"answer": ai_response_text})
        elif intent == "EXIT":
            ai_response_text = "Thank you for using the Lung Cancer AI Assistant. Goodbye!"
        elif intent == "OUT_OF_SCOPE":
            ai_response_text = "Sorry, I can only provide information related to lung cancer."
            st.session_state.chat_memory.save_context({"question": user_query}, {"answer": ai_response_text})
        elif intent == "LUNG_CANCER_QUERY":
            with st.spinner("Searching and formulating response..."):
                retrieved_docs_for_llm = hybrid_search_with_reranking_st(user_query, faiss_index)
                if not retrieved_docs_for_llm:
                    ai_response_text = "I couldn't find specific information for your query."
                else:
                    context_for_llm = format_docs_with_sources_st(retrieved_docs_for_llm)
                    history_variables = st.session_state.chat_memory.load_memory_variables({})
                    current_chat_history_str = history_variables.get("chat_history_str", "")
                    rag_prompt_template_str = """You are a helpful AI assistant specializing in providing information about lung cancer.
Your primary goal is to answer the user's question clearly and concisely, based *only* on the provided "Context from Retrieved Documents" and relevant "Chat History".
Instructions for your answer:
1. Provide a direct answer to the question.
2. If the answer involves listing multiple distinct items (such as symptoms, causes, types of treatments, risk factors, or steps), please use bullet points for clarity and readability. Each bullet point should be concise.
3. If the answer is more explanatory, a definition, or a single point, a well-structured paragraph is appropriate.
4. If the information to answer the question is not in the "Context from Retrieved Documents", clearly state: "I cannot answer your question based on the provided information."
5. Do not make up information or answer from general knowledge outside the provided context.
Here are some examples of how to answer:
---
Example 1 (Paragraph Answer):
Chat History: User: What is SCLC?
Context from Retrieved Documents: Source 1 (File: example_doc.pdf, Page: 5): Small cell lung cancer (SCLC) is a fast-growing type of lung cancer...
Current Question: What is small cell lung cancer?
Answer: Based on the provided context, Small Cell Lung Cancer (SCLC) is described as a fast-growing type of lung cancer...
---
Example 2 (Bullet Point Answer):
Chat History: User: What are the risk factors?
Context from Retrieved Documents: Source 1 (File: another_doc.pdf, Page: 12): Several factors increase the risk of lung cancer. Smoking tobacco is the most significant...
Current Question: What are the main risk factors for lung cancer?
Answer: According to the provided information, the main risk factors for lung cancer include:
* Smoking tobacco (most significant).
* Exposure to secondhand smoke.
* Occupational exposures...
---
Now, answer the following based on the new context and question:
Chat History (for context if needed):
{chat_history_str}
Context from Retrieved Documents:
{context}
Current Question: {question}
Answer:"""
                    rag_chat_prompt = ChatPromptTemplate.from_template(rag_prompt_template_str)
                    try:
                        rag_chain = rag_chat_prompt | rag_llm | StrOutputParser()
                        ai_response_text = rag_chain.invoke({
                            "context": context_for_llm, "question": user_query,
                            "chat_history_str": current_chat_history_str
                        })
                        for doc in retrieved_docs_for_llm:
                            source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                            page_num = doc.metadata.get('page', 'N/A')
                            display_page = page_num + 1 if isinstance(page_num, int) else page_num if page_num is not None else 'N/A'
                            ai_response_sources.append({"file": source_file, "page": display_page})
                    except Exception as e:
                        st.error(f"Error during RAG: {e}"); ai_response_text = "Error generating response."
                st.session_state.chat_memory.save_context({"question": user_query}, {"answer": ai_response_text})
        else:
            ai_response_text = "Not sure how to handle that. Rephrase?"
            st.session_state.chat_memory.save_context({"question": user_query}, {"answer": ai_response_text})

        st.session_state.messages.append({"role": "assistant", "content": ai_response_text, "sources": ai_response_sources})
        if intent == "EXIT":
            with st.chat_message("assistant", avatar="🤖"): st.markdown(ai_response_text)
            time.sleep(0.5); st.stop()
        else:
            st.rerun()
