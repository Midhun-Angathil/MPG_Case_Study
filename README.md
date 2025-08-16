Setup steps:

- Prerequisites:
    - Needs a Langchain API key:
        - Just need to create a free account in Langchain, generate a API key and input that in the streamlit UI.
    - Needs an OpenAI API key:
        - This is a paid alternative (but the costs incurred during the runtime of this project are very very minimal). However, OpenAI expects a minimum of $5 to be added (~ Rs 437.54) to the balance.

- Environment Setup:
    - Install Anaconda
    - Install VSCode
    - Create a folder for this project in the desired file explorer path
    - Open Anaconda Prompt
    - Navigate to the project folder path (cd /path/to/the/folder)
    - type 'code .' (without quotes) and hit enter
    - Open a new terminal using shortcut (ctrl+shift+`) or by clicking the 3-dotted-icon on the file menu at the top-right-corner and clicking on 'terminal' -> 'New terminal'.
    - type 'conda create -p venv_mpg python=='3.10' -y' (without quotes) where venv_mpg is the name of your virtual environment
    - type 'conda activate venv_mpg/' (without quotes) where venv_mpg is the name of your virtual environment. 
    - Run the command - 'pip install -r requirements.txt' (without quotes). If the installation does not work, make sure you are using cmd prompt in the VSCode terminal by closing any powershell icons you see on the right-hand-side of the VSCode terminal window.
    - Boot the app by running "streamlit run app.py" in the VSCode terminal cmd prompt


Model used:
    - OpenAI: gpt-3.5-turbo-16k


Embeddings:
    - OpenAI


Vector Store:
    - FAISS


Chunking strategy:
    - Recursive Character Text Splitter from LangChain


Retrieval strategy:

The retrieval strategy used in your code is Conversational Retrieval Augmented Generation (Conversational RAG) with a History-Aware Retriever.

Details:
    Retriever:
        You use a FAISS vector store as the retriever, which retrieves relevant document chunks based on the user's question.

    History-Aware Retriever:
        You wrap the retriever with create_history_aware_retriever, which uses the conversation history and the latest user question to reformulate the query. This ensures that follow-up questions (which may depend on previous context) are properly understood and relevant context is retrieved.

    Prompting:
        The system prompt explicitly instructs the model to use retrieved context and to answer concisely, and the chain is set up to always include chat history.

Reason for this strategy:
    Handles Follow-up Questions:
        By using a history-aware retriever, the system can handle follow-up or context-dependent questions, not just standalone queries. This is essential for a conversational experience where users may refer to previous answers or context.

    Improves Retrieval Relevance:
        Reformulating the user’s question with chat history helps retrieve more relevant document chunks, especially when the user’s question is ambiguous or references earlier parts of the conversation.

    Grounded Answers:
        The RAG approach ensures that answers are grounded in the actual content of the uploaded PDF, with sources/citations provided for transparency.



How conversation history is maintained in the app:

    In the app, conversation history is maintained using LangChain’s ChatMessageHistory and Streamlit’s st.session_state:

    - Session-Based Storage:

        - Each conversation is associated with a session_id (which can be entered by the user).
        - The app uses a dictionary in st.session_state (e.g., st.session_state.store) to store chat histories for each session.

    - ChatMessageHistory:

        - For each session, a ChatMessageHistory object is created (if it doesn’t already exist) and stored in st.session_state.store[session_id].
        - This object keeps track of all messages exchanged in the conversation (both user and assistant messages).
    
    - Retrieval and Update:

        - When a user asks a question, the app retrieves the chat history for the current session_id and includes it in the prompt for the language model.
        - After each interaction, the new message (question and answer) is appended to the session’s chat history.

    - Multi-Turn Context:

        - By including the full chat history in each prompt, the app enables the model to understand context, handle follow-up questions, and provide more relevant answers.

    Summary:
        Conversation history is session-based, stored in st.session_state using ChatMessageHistory, and is included in each prompt to maintain context across multiple turns in the conversation.


Limitations:

    - Token and Cost Estimation:

        The output token count is estimated, not exact, so API cost estimates may differ from actual usage.
        Only the prompt is tokenized for estimation; actual model responses may vary in length.

    -  Manual Balance Tracking:

        The OpenAI API balance is entered manually by the user and not fetched in real time, so it may become outdated. This would not be a problem in an enterprise.

    - Single User/Session:

        The app is not designed for concurrent multi-user access or persistent user sessions.

    - PDF Parsing Limitations:

        The PDF loader may not handle complex layouts, images, or scanned documents well.

    - Chunking Granularity:

        Fixed chunk size and overlap may not be optimal for all documents, possibly splitting context awkwardly.

    - Citation Granularity:

        Citations are based on chunks, not precise sentences or paragraphs, so references may be broad.

    - No Advanced Error Handling:

        Limited handling of API errors, file upload issues, or memory/disk space problems.

    - Resource Usage:

        Large PDFs may consume significant memory and disk space for FAISS indexing.

    - Security:

        API keys are entered in the UI and could be exposed if the app is not secured.

    - Limited Customization:

        The app is tailored for a specific workflow and may require code changes for other use cases or models.