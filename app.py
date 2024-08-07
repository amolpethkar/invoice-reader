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

def main():
    st.sidebar.title("Input Sidebar")

    # Text box
    text_input = st.sidebar.text_input("Enter some text:")

    # File uploader for images
    uploaded_file = st.sidebar.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    input_prompt = """
    You are an expert in understanding the invoices. We will upload a image as invoice and you
    will have to answer any question based on the uploaded invoice image.
    """
    
    # Submit button
    if st.sidebar.button("Ask me"):
        st.write("You entered:", text_input)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            image_data = input_image_setup(uploaded_file)
            response=get_gemini_response(input_prompt,image_data,input)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("No image uploaded.")

if __name__ == "__main__":
    main()



