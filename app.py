## RAG Conversation with PDF Including Chat History and Citations

# UI framework
import streamlit as st 

# To build a chain that combines documents
from langchain.chains.combine_documents import create_stuff_documents_chain 

# To use chat history & to link FAISS (retriever) with create_stuff_documents_chain:
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# To store and manage messages exchanged in the conversation
from langchain_community.chat_message_histories import ChatMessageHistory

# Defines an interface for storing and retrieving chat messages
from langchain_core.chat_history import BaseChatMessageHistory

# To build struc./reusable prompts & to dynamically inj chat hist into a prompt temeplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# To manage the conversation history across multiple interactions (any runnable)
from langchain_core.runnables.history import RunnableWithMessageHistory

# Importing the LLM model and the Embedding model
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# For Breaking docs to smaller, semantically meaningful chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For loading PDF files
from langchain_community.document_loaders import PyPDFLoader

# Importing os for system operations
import os

# Importing psutil to measure memory usage
import psutil

# For token counting
import tiktoken 

#for estimating the size of a directory in MB
import os

def get_dir_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)  # MB


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

# Helper: Estimate token count and cost for gpt-3.5-turbo-16k
def estimate_cost(prompt, model="gpt-3.5-turbo-16k"):
    input_token_price = 0.005 / 1000
    output_token_price = 0.015 / 1000
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(prompt))
    output_tokens = 75  # 3 sentences, concise
    est_cost = input_tokens * input_token_price + output_tokens * output_token_price
    return input_tokens, output_tokens, round(est_cost, 4)

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
        documents = []
        for uploaded_file in uploaded_files:
            # Construct a temporary filename on disk by prefixing 
            # temp_ to the original filename (uploaded_file.name).
            temppdf=f"./temp_{uploaded_file.name}"
            # Use 'wb' mode to write non-text data such as PDFs, images, etc.
            with open(temppdf, "wb") as f:
                # Write raw PDF bytes to temp_df   
                f.write(uploaded_file.getvalue())
                # store original file name for later use
                file_name = uploaded_file.name

            # Load the PDF file using PyPDFLoader.
            loader = PyPDFLoader(temppdf)
            # Defining docs to hold a list of Document objects
            docs = loader.load()
            # Appending as list items to the 'documents' list that was initialised earlier.
            documents.extend(docs)

        # Split the documents into smaller chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

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

        # This snippet builds a reusable prompt template for a chat model by stitching together three args in order
        contexualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contexualize_q_system_prompt), # sys msg that carries rewrite-the-question instructions
                MessagesPlaceholder("chat_history"), # placeholder that will be filled with the prior conversation history at runtime
                ("human", "{input}"), # a {question} variable supplied during runtime of the chain

            ]
        )

        # Takes the raw user question & chat history, Uses the LLM plus the 3rd arg (prompt) to rewrite the question into a standalone form,
        # and feeds that rewritten question into the original retriever to fetch relevant documents.
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contexualize_q_prompt)


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

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),  # System message with instructions
                MessagesPlaceholder("chat_history"),  # Placeholder for chat history
                ("human", "{input}"),  # Variable for the user's question
            ]
        )

        #builds the “stuff” chain for document-based QA
        question_answering_chain = create_stuff_documents_chain(
            llm=llm,  # The LLM to use for generating answers
            prompt=qa_prompt)  # The prompt template that structures the input to the LLM

        # A composite pipeline that first retrieves context using our retriever and then 
        # invokes a QA chain to generate a grounded answer
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,  # The retriever that fetches relevant documents based on the user’s question and chat history
            combine_docs_chain=question_answering_chain,  # The chain that answers questions based on retrieved documents
        )

        # Function to return base chat message history based on session ID
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store: #Checks for id in store defined in line 75
                st.session_state.store[session_id] = ChatMessageHistory() #stores hist by mapping it to the corresponding ID 
            return st.session_state.store[session_id]

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