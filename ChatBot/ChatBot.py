from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate , load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    temperature=0.6,
    max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)
while True:
    user_input = input("Enter your message: ")

    if user_input.lower() == "exit" or user_input.upper() == "EXIT":
        print("Chat ended.")
        break

    result = model.invoke(user_input)
    print(result.content)
