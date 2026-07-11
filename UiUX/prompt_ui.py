from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import re
from langchain_core.prompts import PromptTemplate , load_prompt


load_dotenv()

import os

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


st.header("Research Paper Summarizer")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="conversational",
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template = load_prompt('template.json')

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})


if st.button("Submit"):
    result = model.invoke(prompt)
    st.write(result.content)

