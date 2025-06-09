import streamlit as st
import os
import base64
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

# --- FIX 1: st.set_page_config() is now the first Streamlit command ---
st.set_page_config(
    page_title="Lung Cancer AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# For Streamlit Cloud deployment, use st.secrets["GOOGLE_API_KEY"]
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY

EMB_MODEL = "models/embedding-001"
RAG_LLM_MODEL_NAME    = "gemini-1.5-flash-latest" # Using a stable, recent model
INTENT_LLM_MODEL_NAME = "gemini-1.5-flash-latest"

APP_ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_NAME = "my_faiss_index_artifact"
FAISS_INDEX_PATH = os.path.join(APP_ROOT_FOLDER, FAISS_INDEX_NAME)
BACKGROUND_IMAGE_FILENAME = "hospital.png"

EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
HYBRID_TOP_N_SEMANTIC         = 25
HYBRID_TOP_N_FOR_LLM          = 5
HYBRID_KEYWORD_BOOST_FACTOR   = 0.05
HYBRID_NO_KEYWORD_MATCH_PENALTY = 0.1
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
    image_filename = BACKGROUND_IMAGE_FILENAME
    img_path = os.path.join(APP_ROOT_FOLDER, image_filename)
    img_base64 = get_base64_of_bin_file(img_path)

    if img_base64:
        background_css = f"background-image: url('data:image/png;base64,{img_base64}');"
    else:
        st.sidebar.warning(f"Background image '{image_filename}' not found. Using default background.")
        background_css = "background-color: #0a0b0c;"

    # --- FIX 2: Reformatted CSS string to prevent parsing errors ---
    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');

        .stApp {{
            {background_css}
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* --- TITLE STYLING (from Code #1) --- */
        .custom-title-container {{
            text-align: center !important;
            width: 100% !important;
            margin-top: 20px !important;
            margin-bottom: 20px !important;
        }}
        .custom-title-box {{
            display: inline-block !important;
            background-color: rgba(0, 0, 0, 0.8) !important;
            padding: 10px 20px !important;
            border-radius: 5px !important;
        }}
        .custom-title {{
            font-family: 'Agdasima', sans-serif !important;
            font-size: 50px !important;
            color: cyan !important;
            margin: 0 !important;
        }}

        /* --- Existing Chat Styling (from Code #2) --- */
        div[data-testid="stChatMessage"] > div {{
            background-color: rgba(40, 42, 54, 0.85);
            border-radius: 15px;
            padding: 12px 15px;
            margin-bottom: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px);
        }}
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] code {{
            color: #EAEAEA;
        }}
        div[data-testid="stChatInput"] > div {{
            background-color: rgba(10, 10, 12, 0.9);
            border-radius: 12px;
            border: 1px solid #00C4FF;
            padding: 5px 10px;
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            color: #EAEAEA !important;
            border: none !important;
        }}
        div[data-testid="stExpander"] details {{
            background-color: rgba(40, 42, 54, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }}
        div[data-testid="stExpander"] summary {{
            color: #B0B0B0;
        }}
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid #00C4FF;
            background-color: rgba(0, 196, 255, 0.1);
            color: #00C4FF;
            font-weight: bold;
            padding: 8px 16px;
        }}
        .stButton>button:hover {{
            background-color: rgba(0, 196, 255, 0.2);
            color: #00C4FF;
            border: 1px solid #00C4FF;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_data
def download_nltk_resources_if_needed():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources_if_needed()


def pos_tag_keyword_extractor(
    text: str,
    num_keywords: int = 0,
    target_pos_tags: set[str] = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
) -> set[str]:
    """Extracts keywords from text using POS tagging."""
    if not text:
        return set()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    keywords = {
        word.lower() for word, tag in tagged_words
        if tag in target_pos_tags and word.lower() not in stop_words and word.isalnum()
    }
    if num_keywords > 0:
        return set(list(keywords)[:num_keywords])
    return keywords


@st.cache_resource
def load_faiss_artifact_cached(allow_dangerous_deserialization: bool = True) -> FAISS | None:
    """Loads the FAISS index from the local path."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index folder not found at {FAISS_INDEX_PATH}.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMB_MODEL, task_type=EMBEDDING_TASK_TYPE_QUERY)
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None


def hybrid_search_with_reranking_st(
    query_text: str,
    faiss_index: FAISS,
) -> list[Document]:
    """Performs a hybrid search and reranks results."""
    if not faiss_index:
        return []
    try:
        semantic_results = faiss_index.similarity_search_with_score(query_text, k=HYBRID_TOP_N_SEMANTIC)
    except Exception as e:
        st.warning(f"Semantic search failed: {e}")
        return []

    query_keywords = pos_tag_keyword_extractor(query_text)
    reranked_results = []
    for doc, score in semantic_results:
        doc_keywords = set(doc.metadata.get("keywords", []))
        common_keywords = len(query_keywords.intersection(doc_keywords))
        
        # Lower score is better, so we subtract a bonus for keyword matches
        rerank_score = score - (common_keywords * HYBRID_KEYWORD_BOOST_FACTOR)
        reranked_results.append({"document": doc, "score": rerank_score})

    reranked_results.sort(key=lambda x: x["score"])
    return [item["document"] for item in reranked_results[:HYBRID_TOP_N_FOR_LLM]]


def format_docs_with_sources_st(docs: list[Document]) -> str:
    """Formats documents for display in the LLM context."""
    formatted_strings = []
    for i, doc in enumerate(docs):
        filename = os.path.basename(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        page_num = page + 1 if isinstance(page, int) else page
        formatted_strings.append(f"Source {i+1} (File: {filename}, Page: {page_num}):\n{doc.page_content}")
    return "\n\n".join(formatted_strings)


INTENT_LABELS = ["GREETING_SMALLTALK", "LUNG_CANCER_QUERY", "OUT_OF_SCOPE", "EXIT"]

@st.cache_resource
def get_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI | None:
    """Initializes and caches a ChatGoogleGenerativeAI instance."""
    try:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, convert_system_message_to_human=True)
    except Exception as e:
        st.error(f"Error initializing LLM ({model_name}): {e}")
        return None


def classify_intent_with_llm_st(user_query: str, _intent_llm: ChatGoogleGenerativeAI) -> str:
    """Classifies user intent using an LLM."""
    if not _intent_llm:
        return "LUNG_CANCER_QUERY"
    
    prompt = ChatPromptTemplate.from_template(
        "Your task is to classify the user's intent. Respond with ONLY one of the following labels: "
        f"{', '.join(INTENT_LABELS)}\nUser input: \"{{user_query}}\"\nClassification:"
    )
    chain = prompt | _intent_llm | StrOutputParser()
    try:
        classification = chain.invoke({"user_query": user_query}).strip().upper()
        return classification if classification in INTENT_LABELS else "LUNG_CANCER_QUERY"
    except Exception:
        return "LUNG_CANCER_QUERY"


# --- Main App Logic ---

load_custom_css()

st.markdown("""
    <div class="custom-title-container">
        <div class="custom-title-box">
            <p class="custom-title">
                Lung Cancer AI Assistant
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load resources
faiss_index = load_faiss_artifact_cached()
rag_llm = get_llm(model_name=RAG_LLM_MODEL_NAME, temperature=0.3)
intent_llm = get_llm(model_name=INTENT_LLM_MODEL_NAME, temperature=0.1)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant for lung cancer information. How can I help you today?"}]
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferWindowMemory(k=CONVERSATION_WINDOW_K, memory_key="chat_history", input_key="question", return_messages=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                st.markdown(message["sources"])

if st.button("Start New Session"):
    st.session_state.messages = [{"role": "assistant", "content": "New session started. How can I help you with lung cancer information?"}]
    st.session_state.chat_memory.clear()
    st.rerun()

if user_query := st.chat_input("Ask a question about lung cancer..."):
    if not all([faiss_index, rag_llm, intent_llm]):
        st.error("A critical component (FAISS index or LLM) failed to load. Please check the logs.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        intent = classify_intent_with_llm_st(user_query, intent_llm)
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
                    
                    # --- FIX 3: Reformatted RAG prompt string ---
                    rag_prompt_template = (
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

                    prompt = ChatPromptTemplate.from_template(rag_prompt_template)
                    chain = prompt | rag_llm | StrOutputParser()
                    
                    try:
                        ai_response_content = chain.invoke({
                            "context": context,
                            "question": user_query,
                            "chat_history": chat_history
                        })
                        ai_response_sources = context
                    except Exception as e:
                        st.error(f"Error during response generation: {e}")
                        ai_response_content = "Sorry, I encountered an error while generating the response."

        # Save context and display response
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
