from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from datetime import date
from langchain_openai import ChatOpenAI

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain.output_parsers.json import SimpleJsonOutputParser

import os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')



embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model = 'gpt-3.5-turbo-0125', temperature=0.0)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# Final prompt template
today = date.today()
d1 = today.strftime("%d/%m/%Y")

system_prompt = ("You're a brilliant data sorcerer."
                 "Your task is to generate responses based on examples."
                 " Your responses should adhere to the patterns observed in the examples provided. Note that today's date is " + str(d1) + "."+"Should an inquiry pertain to showcasing or producing data for the preceding 7 days, last week, last quarter, or akin durations, ensure adept substitution of accurate dates within the parameters of start_date and end_date."
                 "\n\n"
                 "{context}"

)
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#chain = final_prompt | ChatOpenAI(openai_api_key=OPENAI_KEY, model = 'gpt-3.5-turbo-0125', temperature=0.0)
#chain.invoke({"input": "  I want to refresh all our records from the electronic health records, what steps are involved? "}).content



store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@traceable
def run(query):
    question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )
    
    fans = conversational_rag_chain.invoke(
        {"input": query},
        config={    
            "configurable": {"session_id": "abc123"}
               },
    )["answer"]  
    """ 
    rag_chain =   final_prompt | llm
    #rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )
    
    fans = conversational_rag_chain.invoke(
        {"input": query},
        config={    
            "configurable": {"session_id": "abc123"}
               }  # constructs a key "abc123" in `store`.
    )
    print(fans, type(fans), "------------------")
    return fans
    """
    return fans
    



demo = gr.Interface(fn=run, inputs="text", outputs="text")

#demo.launch()
demo.launch(server_name="0.0.0.0", share=True)
