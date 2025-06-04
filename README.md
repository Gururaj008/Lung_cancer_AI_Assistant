🌟 App Name: LungGuardian  

"Your Trusted Companion in Lung Cancer Information" 

   
📄 Overview 

LungGuardian  is an AI-powered assistant designed to provide accurate, evidence-based information about lung cancer. Built with Streamlit , LangChain , and Google Gemini , it leverages a knowledge base of medical documents to answer user queries with supporting sources.   
🔍 Key Features 

    Interactive Q&A : Ask questions about lung cancer symptoms, treatments, risk factors, and more.  
    Hybrid Search : Combines semantic similarity (FAISS) and keyword-based ranking for precise answers.  
    Evidence-Based Responses : Sources are cited for every answer derived from the knowledge base.  
    Dark Mode UI : A glowing cyan-themed interface for a modern, eye-friendly experience.
     

🧠 Under the Hood: How It Works 
1. Data Ingestion & Preparation  

    Document Processing :  
        Files (PDFs/text) are read from a folder (FOLDER_PATH).  
        Text is split into 1,000-character chunks  with 100-character overlap  to preserve context.  
        Each chunk is stored with metadata:  
            source: File name (e.g., lung_cancer_guide.pdf).  
            page: Page number (if applicable).  
            keywords: Extracted via NLTK POS tagging  (nouns, adjectives, etc.) for hybrid search 


2. Embedding Generation & FAISS Storage  

    Embeddings :  
        Chunks are embedded using Google Gemini's embedding model  (embedding-001).
         
    Vector Database :  
        Embeddings and metadata are stored in a FAISS index  (index.faiss + index.pkl) for fast semantic search 

3. Hybrid Search Pipeline  

When a user asks a question:   
a. Semantic Search  

    Query is embedded and compared to FAISS vectors using cosine similarity.  
    Top 25 results  (HYBRID_TOP_N_SEMANTIC) are retrieved.
     

b. Keyword Extraction  

    Query is tokenized with NLTK  to extract keywords (nouns, adjectives) 

    .  
    Keywords are matched against chunk metadata to boost relevant results.
     

c. Reranking  

    Results are reranked by adjusting semantic scores:  
        Boost  matches with query keywords (HYBRID_KEYWORD_BOOST_FACTOR = 0.05).  
        Penalize  non-matching results (HYBRID_NO_KEYWORD_MATCH_PENALTY = 0.1).
         
    Final Top 5  results (HYBRID_TOP_N_FOR_LLM) are passed to the LLM.
     

4. LLM Answer Generation  

    Prompt Engineering :  
        Context from reranked chunks is formatted into a structured prompt.  
        Gemini Flash (gemini-2.5-flash-preview-05-20) generates answers with:  
            Bullet points for lists (e.g., symptoms, treatments).  
            Paragraphs for definitions/explanations.
             
        If no context matches, the LLM explicitly states: "I cannot answer your question based on the provided information." 
         
     

5. Source Citation  

    Retrieved chunk metadata is displayed in an expander:  
        File name, page number, and content snippet for transparency.
         
     
