import streamlit as st
import os
import base64
import time
import nltk # Import NLTK early for path configuration

# --- NLTK Data Path Configuration (Add this block near the top) ---
APP_ROOT_FOLDER_FOR_NLTK = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = os.path.join(APP_ROOT_FOLDER_FOR_NLTK, 'nltk_data')
if os.path.exists(NLTK_DATA_PATH):
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)
        # print(f"DEBUG: Added custom NLTK data path: {NLTK_DATA_PATH}") # For debugging
else:
    # This print will show in Streamlit logs if the folder is missing
    print(f"WARNING: Packaged NLTK data path not found: {NLTK_DATA_PATH}")
# --- END NLTK Data Path Configuration ---

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# ... (rest of your LangChain imports)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory


# ==============================================================================
# CONFIGURATION (User needs to set these - REVERTED TO LUNG CANCER FOCUS)
# ==============================================================================
# (Your existing configuration section - API_KEY, FOLDER_PATH, EMB_MODEL, etc.)
# For Streamlit Cloud deployment, use st.secrets["GOOGLE_API_KEY"]
# For local, set as env var or use the placeholder carefully for testing.
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "YOUR_ACTUAL_GOOGLE_API_KEY_FOR_LOCAL_TESTING" # Placeholder
    if GOOGLE_API_KEY == "YOUR_ACTUAL_GOOGLE_API_KEY_FOR_LOCAL_TESTING":
        st.sidebar.warning("⚠️ API Key not found. Using placeholder. App may not function fully.")

if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_ACTUAL_GOOGLE_API_KEY_FOR_LOCAL_TESTING":
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
elif not os.getenv("GOOGLE_API_KEY") and GOOGLE_API_KEY == "YOUR_ACTUAL_GOOGLE_API_KEY_FOR_LOCAL_TESTING":
     st.sidebar.error("Google API Key is not properly configured.")
     # st.stop() # Consider stopping if essential

# FOLDER_PATH should point to where the FAISS index SUBFOLDER is located
# If FAISS index is in the root with app.py, FOLDER_PATH can be APP_ROOT_FOLDER_FOR_NLTK
# Or if FAISS index is in a specific data subfolder:
# FOLDER_PATH = os.path.join(APP_ROOT_FOLDER_FOR_NLTK, "data_containing_faiss_subfolder")
FOLDER_PATH = APP_ROOT_FOLDER_FOR_NLTK # Assuming FAISS subfolder is in the app root

EMB_MODEL = "models/embedding-001"
RAG_LLM_MODEL_NAME = "gemini-pro"
INTENT_LLM_MODEL_NAME = "gemini-pro"

FAISS_INDEX_NAME = "my_faiss_index_artifact" # Your FAISS subfolder name
FAISS_INDEX_PATH = os.path.join(FOLDER_PATH, FAISS_INDEX_NAME) # Path to the FAISS subfolder

EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
HYBRID_TOP_N_SEMANTIC = 25
HYBRID_TOP_N_FOR_LLM = 5
HYBRID_KEYWORD_BOOST_FACTOR = 0.05
HYBRID_NO_KEYWORD_MATCH_PENALTY = 0.1
CONVERSATION_WINDOW_K = 5
BACKGROUND_IMAGE_FILENAME = "hospital.png" # Your background image
# ==============================================================================


# --- Helper Functions ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    # ... (same as before)
    try:
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

def load_custom_css():
    # ... (same as before, using BACKGROUND_IMAGE_FILENAME)
    image_filename = BACKGROUND_IMAGE_FILENAME
    img_path = os.path.join(APP_ROOT_FOLDER_FOR_NLTK, image_filename) # Assuming image is in app root
    img_base64 = get_base64_of_bin_file(img_path)
    if img_base64:
        background_css = f"background-image: url('data:image/png;base64,{img_base64}');"
    else:
        st.sidebar.warning(f"BG image '{image_filename}' not found. Using default BG.")
        background_css = "background-color: #0a0b0c;"
    # (The rest of your custom_css string)
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


@st.cache_data # Cache this check
def download_nltk_resources_if_needed(): # MODIFIED - Now primarily a check
    """Checks for NLTK resources, assuming they are packaged."""
    resources_to_check = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }
    all_found = True
    for resource_path, resource_name in resources_to_check.items():
        try:
            nltk.data.find(resource_path) # Will use nltk.data.path which includes our custom path
        except LookupError:
            st.error(f"Packaged NLTK resource missing: '{resource_name}' (expected at '{resource_path}' within nltk_data). "
                     "Please ensure the 'nltk_data' folder is correctly deployed with the app.")
            all_found = False
    
    if not all_found:
        st.warning("Some essential NLTK data packages are missing. Keyword extraction and intent classification might not work correctly.")
    # No download attempt here; relies on packaged data.

# (pos_tag_keyword_extractor, load_faiss_artifact_cached, hybrid_search_with_reranking_st,
# format_docs_with_sources_st, get_llm, classify_intent_with_llm_st functions remain the same
# as in your last fully working version, just ensure they use the globally defined constants)

# --- Make sure all your other functions are defined here ---
# For example:
def pos_tag_keyword_extractor(text: str, num_keywords=0, target_pos_tags={'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}) -> set:
    # (Your working pos_tag_keyword_extractor function)
    if not text: return set()
    words_from_text = []
    try:
        for sentence in nltk.sent_tokenize(text): words_from_text.extend(nltk.word_tokenize(sentence))
    except LookupError: words_from_text.extend(nltk.word_tokenize(text))
    try:
        stop_words_set = set(stopwords.words('english'))
        meaningful_words = [w.lower() for w in words_from_text if w.isalnum() and w.lower() not in stop_words_set and len(w) > 2]
        if not meaningful_words: return set()
        tagged_meaningful_words = nltk.pos_tag(meaningful_words)
    except LookupError: return set(meaningful_words[:num_keywords]) if num_keywords > 0 and meaningful_words else set(meaningful_words)
    extracted_keywords_list = [word for word, tag in tagged_meaningful_words if tag in target_pos_tags]
    if num_keywords > 0 and extracted_keywords_list:
        return set(word for word, count in Counter(extracted_keywords_list).most_common(num_keywords))
    return set(extracted_keywords_list)

@st.cache_resource
def load_faiss_artifact_cached(allow_dangerous_deserialization=True) -> FAISS | None:
    # (Your working load_faiss_artifact function, adapted for Streamlit caching)
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS artifact not found at {FAISS_INDEX_PATH}.")
        return None
    try:
        if not os.getenv("GOOGLE_API_KEY"):
             st.error("GOOGLE_API_KEY not available for embedding model.")
             return None
        query_embeddings_model = GoogleGenerativeAIEmbeddings(model=EMB_MODEL, task_type=EMBEDDING_TASK_TYPE_QUERY)
        index = FAISS.load_local(FAISS_INDEX_PATH, query_embeddings_model, allow_dangerous_deserialization=allow_dangerous_deserialization)
        return index
    except Exception as e: st.error(f"Error loading FAISS artifact: {e}"); return None

def hybrid_search_with_reranking_st(
    query_text: str, faiss_index: FAISS,
    top_n_semantic: int = HYBRID_TOP_N_SEMANTIC, top_n_final_for_llm: int = HYBRID_TOP_N_FOR_LLM,
    keyword_boost_factor: float = HYBRID_KEYWORD_BOOST_FACTOR, no_match_penalty: float = HYBRID_NO_KEYWORD_MATCH_PENALTY
) -> list[Document]:
    # (Your working hybrid_search_with_reranking function)
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
    # (Your working format_docs_with_sources function)
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A'); page = doc.metadata.get('page', 'N/A')
        filename = os.path.basename(source) if source != 'N/A' else 'N/A'
        page_display = page + 1 if isinstance(page, int) else page if page is not None else 'N/A'
        formatted_docs.append(f"Source {i+1} (File: {filename}, Page: {page_display}):\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

INTENT_LABELS = ["GREETING_SMALLTALK", "LUNG_CANCER_QUERY", "OUT_OF_SCOPE", "EXIT"]
@st.cache_resource
def get_llm(_model_name, temperature, _api_key):
    # (Your working get_llm function)
    try:
        if not _api_key:
             st.error(f"GOOGLE_API_KEY not available for LLM ({_model_name}).")
             return None
        llm = ChatGoogleGenerativeAI(model=_model_name, temperature=temperature, convert_system_message_to_human=True, google_api_key=_api_key)
        return llm
    except Exception as e: st.error(f"Error initializing LLM ({_model_name}): {e}"); return None

def classify_intent_with_llm_st(user_query: str, _intent_llm: ChatGoogleGenerativeAI) -> str:
    # (Your working classify_intent_with_llm function)
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
        else: return "LUNG_CANCER_QUERY"
    except Exception as e: return "LUNG_CANCER_QUERY"


# --- Streamlit App UI and Logic ---
st.set_page_config(page_title="Lung Cancer AI Assistant", layout="wide", initial_sidebar_state="collapsed")
load_custom_css()

# Custom Title - REVERTED TO LUNG CANCER THEME
st.markdown("<div class='custom-title-container'><div class='custom-title'><h1>Lung Cancer AI Assistant</h1></div></div>", unsafe_allow_html=True)

# Initialize NLTK (cached)
download_nltk_resources_if_needed() # This will now primarily check

# Load FAISS index and LLMs
current_api_key = os.getenv("GOOGLE_API_KEY") # Get the key that was set at the top
faiss_index = load_faiss_artifact_cached()
rag_llm = get_llm(RAG_LLM_MODEL_NAME, temperature=0.3, _api_key=current_api_key)
intent_llm = get_llm(INTENT_LLM_MODEL_NAME, temperature=0.1, _api_key=current_api_key)

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

if st.button("Start New Session", key="start_new_session_main"): # Added a key
    st.session_state.messages = [
         {"role": "assistant", "content": "Hello! I am your AI assistant for lung cancer information. How can I help you today?"}
    ]
    if "chat_memory" in st.session_state: st.session_state.chat_memory.clear()
    st.rerun()

if user_query := st.chat_input("Ask a question about lung cancer..."):
    if not current_api_key or (API_KEY == "YOUR_ACTUAL_GOOGLE_API_KEY_FOR_LOCAL_TESTING" and not GOOGLE_API_KEY_FROM_SECRETS): # Check current_api_key
        st.error("Google API Key is not properly configured.")
    elif not faiss_index or not rag_llm or not intent_llm:
        st.error("A critical component (FAISS index or LLM) failed to load.")
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
