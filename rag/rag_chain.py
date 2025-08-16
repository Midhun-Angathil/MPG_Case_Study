# To build a chain that combines documents
from langchain.chains.combine_documents import create_stuff_documents_chain 
# To use chat history & to link FAISS (retriever) with create_stuff_documents_chain:
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
# To build struc./reusable prompts & to dynamically inj chat hist into a prompt temeplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_rag_chain(llm, retriever, system_prompt, contextualize_q_system_prompt):
    """Builds the RAG chain with history-aware retriever and prompt templates."""

    # This snippet builds a reusable prompt template for a chat model by stitching together three args in order
    contexualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt), # sys msg that carries rewrite-the-question instructions
            MessagesPlaceholder("chat_history"), # placeholder that will be filled with the prior conversation history at runtime
            ("human", "{input}"), # a {question} variable supplied during runtime of the chain

        ]
    )

    # Takes the raw user question & chat history, Uses the LLM plus the 3rd arg (prompt) to rewrite the question into a standalone form,
    # and feeds that rewritten question into the original retriever to fetch relevant documents.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contexualize_q_prompt)


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

    return rag_chain