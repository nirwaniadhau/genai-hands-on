from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import PromptTemplate, load_prompt


load_dotenv()
st.header("Research Tool")
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

paper_input=st.selectbox('Choose a research paper:', 
    ('Paper 1: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'Paper 2: ImageNet Classification with Deep Convolutional Neural Networks', 'Paper 3: Climate Change Impacts'))

style_input=st.selectbox('Choose a summary style:', 
    ('Beginner Freindly', 'Technical', 'Code-oriented','Mathematical'))

length_input=st.selectbox('Choose summary length:', 
    ('Short(1-2 paragraphs)', 'Medium(3-5 paragraphs )', 'Long(detailed summary)'))


# template

template=load_prompt('template.json')

#fill the placeholders



if st.button('Summarize'):
    chain= template | model
    result=chain.invoke({
    "length_input":length_input,
    "paper_input":paper_input,
    "style_input":style_input
   })
    st.write(result.content)


