from langchain_core import messages
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate , load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    temperature=0.6,
)

model = ChatHuggingFace(llm=llm)

chat_history  = [
    SystemMessage(content = "You are a helpful AI assistant. Answer user questions clearly, accurately, and concisely. Use the provided context when available. If the answer is not present in the provided context, say that you do not know rather than generating incorrect information. Be polite and explain concepts in an easy-to-understand way.")
]

while True:
    user_input = input("Enter your message: ")
    chat_history.append( HumanMessage(content=user_input))
    if user_input.lower() == "exit" or user_input.upper() == "EXIT":
        print("Chat ended.")
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Ai:" , result.content)

print("CHAT HISTORY!!" , chat_history)