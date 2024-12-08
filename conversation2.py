from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import chainlit as cl

DB_FAISS_PATH = r"C:\Masters\NLP Project-20241206T163105Z-001\NLP Project\Data\vectorstores\db_faiss"

# More conversational and flexible prompt template
custom_prompt_template = """You are a compassionate and empathetic mental health support assistant. 
Respond naturally to the user's message, drawing from your knowledge and the conversation context.

Conversation History:
{chat_history}

Context (if relevant):
{context}

User's Current Message: {question}

Your response should be:
- Warm and conversational
- Empathetic and understanding
- Tailored to the user's current message
- Avoid clinical jargon unless necessary
- Encourage open communication

Assistant's Response:"""

def set_custom_prompt():
    """
    Creates a custom prompt template for conversational QA retrieval.
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=['chat_history', 'context', 'question']
    )
    return prompt

def load_llm():
    """
    Loads the LLM using CTransformers.
    """
    try:
        llm = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Note: Ensure you have the correct GGML/GGUF version
            model_type="mistral",  # Specify the model type
            max_new_tokens=6192,  # Maximum tokens to generate
            temperature=0.7,     # Sampling temperature
            top_k=50,            # Top-k sampling parameter
            top_p=0.95,          # Top-p sampling parameter
            #repetition_penalty=1.1  # Penalty for repeated tokens
)
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise

def retrieval_qa_chain(llm, prompt, db):
    """
    Creates a RetrievalQA chain with the given LLM, prompt, and vectorstore.
    """
    try:
        # Create the QA chain
        qa_chain = load_qa_chain(
            llm=llm, 
            chain_type="stuff", 
            prompt=prompt,
            verbose=True
        )
        
        # Create a retriever
        retriever = db.as_retriever(search_kwargs={'k': 1})  # Reduced to minimize irrelevant context
        
        return (qa_chain, retriever)
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        raise

def is_greeting(message):
    """
    Check if the message is a greeting.
    """
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'hola']
    return any(greeting in message.lower() for greeting in greetings)

def handle_greeting():
    """
    Generate a friendly, conversational greeting response.
    """
    greetings = [
        "Hi there! I'm here to listen and support you. How are you feeling today?",
        "Hello! Welcome. I'm ready to provide a compassionate ear and helpful guidance. What's on your mind?",
        "Hey! It's great that you've reached out. I'm here to support you through whatever you're experiencing.",
        "Good to see you! Mental health is a journey, and I'm here to walk alongside you. What would you like to discuss?"
    ]
    import random
    return random.choice(greetings)

def qa_bot():
    """
    Initializes the QA bot with embeddings, vectorstore, LLM, and prompt.
    """
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2', 
            model_kwargs={'device': 'cpu'}
        )

        # Load vectorstore
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Load LLM
        llm = load_llm()
        
        # Set custom prompt
        qa_prompt = set_custom_prompt()
        
        # Create QA chain and retriever
        qa_chain, retriever = retrieval_qa_chain(llm, qa_prompt, db)
        
        return qa_chain, retriever, db
    except Exception as e:
        print(f"Error initializing QA bot: {e}")
        raise

## Chainlit Integration ###
@cl.on_chat_start
async def start():
    """
    Initializes the Chainlit session with the QA bot and conversation history.
    """
    msg = cl.Message(content="Initializing Mental Health Bot...")
    await msg.send()
    
    try:
        # Initialize QA components
        qa_chain, retriever, db = qa_bot()
        
        # Store components in user session
        cl.user_session.set("qa_chain", qa_chain)
        cl.user_session.set("retriever", retriever)
        cl.user_session.set("db", db)
        cl.user_session.set("chat_history", [])  # Initialize conversation history
        
        # Update welcome message
        msg.content = "Welcome to the Virtual Mental Health Support Bot. I'm here to listen and support you."
        await msg.update()
    
    except Exception as e:
        msg.content = f"Error initializing the bot: {e}"
        await msg.update()

@cl.on_message
async def main(message):
    """
    Handles user messages and provides responses using the conversational bot.
    """
    # Retrieve session components
    qa_chain = cl.user_session.get("qa_chain")
    retriever = cl.user_session.get("retriever")
    db = cl.user_session.get("db")
    
    if not (qa_chain and retriever and db):
        await cl.Message(content="Error: Bot is not properly initialized.").send()
        return
    
    # Retrieve and update conversation history
    chat_history = cl.user_session.get("chat_history", [])
    user_message = message.content
    
    try:
        # Handle greetings separately
        if is_greeting(user_message):
            res = handle_greeting()
            response_message = cl.Message(content=res)
            await response_message.send()
            
            # Update conversation history
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Assistant: {res}")
            cl.user_session.set("chat_history", chat_history)
            return

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(user_message)
        
        # Prepare context for the response
        context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
        
        # Prepare conversation input
        input_context = {
            'input_documents': relevant_docs,
            'question': user_message,
            'chat_history': "\n".join(chat_history),
            'context': context
        }
        
        # Generate response
        cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
        res = await cl.make_async(qa_chain.run)(**input_context, callbacks=[cb])
        
        # Update conversation history
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Assistant: {res}")
        cl.user_session.set("chat_history", chat_history)
        
        # Send response
        response_message = cl.Message(content=res)
        
        await response_message.send()
    
    except Exception as e:
        error_message = f"Error processing your message: {e}"
        await cl.Message(content=error_message).send()