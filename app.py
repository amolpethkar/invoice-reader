#!pip install streamlit google-generativeai python-dotenv langchain PyPDF2  langchain_google_genai langchain-community chromadb faiss-cpu

# Imports
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image


##initialize our streamlit app

st.set_page_config(page_title="MultiLanguage Invoice Extractor")
st.header("MultiLanguage Invoice Extractor")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image of the Invoice", type=["jpg", "jpeg", "png"])
image=""   
submit=st.button("Tell me about the invoice")



