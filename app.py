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



# Load ENV and KEYS
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Function to load Gemini Pro Vision
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
##initialize our streamlit app


st.set_page_config(page_title="MultiLanguage Invoice Extractor")
st.header("MultiLanguage Invoice Extractor")
input=st.sidebar.text_input("Input Prompt: ",key="input")
uploaded_file = st.sidebar.file_uploader("Choose an image of the Invoice", type=["jpg", "jpeg", "png"])
image=""   


#submit=st.sidebar.button("Ask Me")


input_prompt = """
You are an expert in understanding the invoices. We will upload a image as invoice and you
will have to answer any question based on the uploaded invoice image.
"""

if st.sidebar.button("Ask me") and uploaded_file is not None:      
            image = Image.open(uploaded_file)
            st.sidebar.image(image, caption="Uploaded Image.", use_column_width=True)
            image_data = input_image_setup(uploaded_file)
            response=get_gemini_response(input_prompt,image_data,input)
            st.subheader("The Response is")
            st.write(response)
else:
             st.write("No image uploaded.")

   




