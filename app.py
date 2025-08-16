## RAG Conversation with PDF Including Chat History and Citations

# UI framework
import streamlit as st 

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# To manage the conversation history across multiple interactions (any runnable)
from langchain_core.runnables.history import RunnableWithMessageHistory

# Importing the LLM model and the Embedding model
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Importing os for system operations
import os

# Importing psutil to measure memory usage
import psutil 

# Importing custom modules from this repository
from rag.pdf_loader import load_and_split_pdfs # PDF loading and chunking
from rag.faiss_utils import get_dir_size_mb # disk/memory utilities
from rag.openai_utils import estimate_cost # Cost estimation, balance, and LLM helpers
from rag.chat_history import get_session_history # Chat history/session management
from rag.rag_chain import build_rag_chain # RAG chain and prompt setup


# Set up Steamlit
st.title("Conversational RAG with PDF Including Chat History and Citations")
st.write("Upload a PDF and have a conversation with its content")

# Input the OpenAI API key and the Langchain API key
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
langchain_api_key = st.text_input("Enter your Langchain API key", type="password")

# Ask user for current OpenAI balance (in USD)
if "openai_balance" not in st.session_state:
    st.session_state.openai_balance = None
st.session_state.openai_balance = st.number_input(
    "Enter your current OpenAI API balance (USD) for tracking costs:",
    min_value=0.0,
    format="%.2f",
    value=st.session_state.openai_balance or 0.0,
    step=0.01,
    help="Check your balance at https://platform.openai.com/account/billing"
)

# Helper: Measure memory usage
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return round(mem_bytes / (1024 * 1024), 2)

# Helper: Estimate memory needed for FAISS (very rough, depends on embedding size and chunk count)
def estimate_faiss_memory(num_chunks, embedding_dim=1536, dtype_bytes=4):
    # Each vector: embedding_dim * dtype_bytes bytes
    # Total: num_chunks * embedding_dim * dtype_bytes
    total_bytes = num_chunks * embedding_dim * dtype_bytes
    return round(total_bytes / (1024 * 1024), 2)  # in MB

# Check if the API keys are provided and initialize the LLM and embedding models:
if not openai_api_key:
    #st.error() displays msg in red
    st.error("Please provide your OpenAI API key and LangChain API Key to continue.")
    st.stop()
elif openai_api_key and not openai_api_key.startswith("sk-"):
    # st.warning() displays msg in yellow.
    st.warning("Invalid API key format. It should start with 'sk-'.")
    st.stop()
else:
    pass

if not langchain_api_key:
    st.error("Please provide your Langchain API key to continue.")
    st.stop()
else:
    # Initialize the LLM and embedding models with the provided API key
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Chat Interface
    session_id = st.text_input("Enter your session ID (Optional)", value="default_session")
    
    # Create a session store if it doesn't exist
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # PDF file(s) upload button
    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)

    # Process the uploaded PDF files
    if uploaded_files:
        splits = load_and_split_pdfs(uploaded_files)  # Function defined in rag/pdf_loader.py

        # Estimate memory needed for FAISS
        est_faiss_mem = estimate_faiss_memory(len(splits))
        st.info(f"Estimated memory needed for FAISS index: {est_faiss_mem} MB")
        #mem_before = get_memory_usage_mb()
        #st.info(f"Memory usage before FAISS: {mem_before} MB")

        # Confirm before creating FAISS
        #proceed_faiss = st.button("Proceed to create FAISS index")

        # Create a FAISS vector store from the split documents
        
        
        vector_store = FAISS.from_documents(splits, embeddings) #splits - line105 & embeddings - line70
        vector_store.save_local("faiss_index") # Saves it in project directory
        retriever = vector_store.as_retriever() #instantiates a retriever from the vector store & returns raw document chunks
        #mem_after = get_memory_usage_mb()
        #st.success(f"Memory usage after FAISS: {mem_after} MB")

        # Calculate and display FAISS index disk usage
        faiss_disk_usage = get_dir_size_mb("faiss_index")
        st.info(f"FAISS index disk usage: {faiss_disk_usage} MB")

        # Create the System Prompt for the Chat
        contexualize_q_system_prompt = (
            "Given a chat history and the latest user question" #Instructs the model that it will receive both the conversation so far and a new question.
            "which might have reference context in the chat history, " #Reminds the model that the question could rely on earlier messages.
            "formulate a standalone question which can be understood " #Tells the model to rewrite the user’s inquiry so it makes sense in isolation.
            "without the chat history. Do NOT answer the question, " #Last 2 lines of the sys prompt -> Emphasizes that this is purely a rewriting 
            "just reformulate it if needed and otherwise return it as is." #(contd.) - task, not an answering task.
        )    

        # Answer Question:
        system_prompt = (
            "You are an assistant for question-answering tasks. " # This opening instruction establishes the LLM’s role and the high-level objective—to act as a QA assistant.
            "Use the following pieces of retrieved context to answer " #tells the model to rely on provided context snippets rather than its own internal knowledge.
            "the question. If you don't know the answer, say that you " #sets the refusal or fallback policy: rather than hallucinating, the assistant must explicitly admit ignorance.
            "don't know. Use three sentences maximum and keep the " #enforces brevity and format constraints, steering the model toward short,
            "answer concise." #(contd.) to-the-point answers.                                    
            "\n\n" # Inserts two line breaks, visually separating instructions from the dynamic context that follows.
            "{context}" # A placeholder that LangChain will replace at runtime with the actual retrieved document snippets relevant to the user’s query.
        )

        # Import from module rag/rag_chain.py
        rag_chain = build_rag_chain(
            llm=llm,  # The LLM to use for generating answers
            retriever=retriever,  # The retriever that fetches relevant documents based on the user’s question and chat history
            system_prompt=system_prompt,  # System message with instructions
            contextualize_q_system_prompt=contexualize_q_system_prompt  # System message for rewriting the question
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,  # The RAG chain that combines retrieval and QA
            get_session_history,  # The chat message history for the current session
            input_messages_key = "input", # Defines the dictionary key under which incoming user messages will be provided to the runnable.
            history_messages_key = "chat_history",  # Specifies the key name where the chain should look for historical messages.
            output_messages_key = "answer",  # Determines the key under which the chain’s response will be stored.
        )

        # User input for the question
        user_input = st.text_input("Ask a question about the PDF content")
        if user_input:
            # Retrieve the chat history for the current session
            session_history = get_session_history(session_id)

            chat_history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in session_history.messages])
            prompt = (
                f"System: {system_prompt}\n"
                f"Chat History: {chat_history_text}\n"
                f"User: {user_input}"
            )
            input_tokens, output_tokens, est_cost = estimate_cost(prompt)
            st.sidebar.info(
                f"Estimated API Call Cost: ${est_cost} "
                f"(Input tokens: {input_tokens}, Output tokens: {output_tokens})"
            )
            balance_after = round(st.session_state.openai_balance - est_cost, 4)
            st.sidebar.info(f"Balance after call (est.): ${balance_after}")

            #proceed = st.button("Proceed with API Call")
            # Run the conversational RAG chain with the user input and chat history
            response = conversational_rag_chain.invoke(
                {"input": user_input}, #Supplies the most recent user query under the "input" key, matching user_input defined in line 130.
                
                # Injects a session_id into the wrapper’s internal config. This tells the 
                # message-history manager to namespace or persist history under that 
                # ID—e.g. storing and retrieving messages in our store using the key "abc123".
                
                config={
                    "configurable" : {"session_id": session_id},
                }) # constructs a key "abc123" in 'store'. 

            # Subtract cost from balance and update UI
            st.session_state.openai_balance = balance_after
            if "total_api_cost" not in st.session_state:
                st.session_state.total_api_cost = 0
            st.session_state.total_api_cost += est_cost
            #st.success(f"New OpenAI API Balance: ${st.session_state.openai_balance}")
            st.sidebar.info(f"Total API Cost This Session: ${round(st.session_state.total_api_cost, 4)}")

            #st.write(st.session_state.store) # Display the store info
            st.write("Assistant:", response["answer"])  # Display the assistant's answer

            # Show sources/citations
            sources = response.get("context", None) or response.get("source_documents", None)
            if sources:
                # Light blue for heading, light gray for text (good for dark backgrounds)
                st.markdown(
                    '<span style="font-size:14px; color:#90caf9;"><b>Sources / Citations:</b></span>',
                    unsafe_allow_html=True
                )
                for i, doc in enumerate(sources):
                    page = doc.metadata.get("page", "N/A")
                    snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    st.markdown(
                        f'<div style="font-size:12px; color:#e0e0e0;">'
                        f'<b>Source {i+1}:</b> Page {page}<br>{snippet}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No sources/citations returned.")

            # Adding a line space before displaying chat history
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("Chat History:", session_history.messages)  # Display the chat history