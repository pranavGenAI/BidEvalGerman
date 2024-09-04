import streamlit as st
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from streamlit.components.v1 import html
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json
import hashlib
import time

st.set_page_config(page_title="KI zur Bewertung von Angebotserwiderungen ", layout="wide")

st.markdown(
    """
    <style>
    .stButton button {
        background: linear-gradient(120deg,#FF007F, #A020F0 100%) !important;
        color: white !important;
    }
    body {
        color: white;
        background-color: #1E1E1E;
    }
    .stTextInput, .stSelectbox, .stTextArea, .stFileUploader {
        color: white;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Adding the logo and other elements in the header
st.markdown(
    f"""
    <header tabindex="-1" data-testid="stHeader" class="st-emotion-cache-12fmjuu ezrtsby2">
        <div data-testid="stDecoration" id="stDecoration" class="st-emotion-cache-1dp5vir ezrtsby1"></div>
        <div class="header-content">
            <!-- Add the logo here -->
            <img src="https://www.vgen.it/wp-content/uploads/2021/04/logo-accenture-ludo.png" class="logo" alt="Logo">
        
    </header>

    """,
    unsafe_allow_html=True
)


# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Define users and hashed passwords for simplicity
users = {
    "tomas": hash_password("tomas123"),
    "admin": hash_password("admin")
}


TOKEN_FILE = "./data/token_counts_eval.json"

def read_token_counts():
    try:
        with open("./data/token_counts_eval.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def write_token_counts(token_counts):
    with open("./data/token_counts_eval.json", "w") as f:
        json.dump(token_counts, f)


def get_token_count(username):
    token_counts = read_token_counts()
    return token_counts.get(username, 1000)  # Default to 1000 tokens if not found

def update_token_count(username, count):
    token_counts = read_token_counts()
    token_counts[username] = count
    write_token_counts(token_counts)


def login():
    col1, col2, col3 = st.columns([1, 1, 1])  # Create three columns with equal width
    with col2:  # Center the input fields in the middle column
       
        username = st.text_input("Benutzername")
        password = st.text_input("Passwort", type="password")
        
        if st.button("Einloggen"):
            hashed_password = hash_password(password)
            if username in users and users[username] == hashed_password:
                token_counts = read_token_counts()
                tokens_remaining = token_counts.get(username, 500)  # Default to 500 tokens if not found
                
                if tokens_remaining > 0:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.tokens_remaining = tokens_remaining
                    st.session_state.tokens_consumed = 0
                    st.success("Erfolgreich eingeloggt!")
                    st.experimental_rerun()  # Refresh to show logged-in state
                else:
                    st.error("No tokens remaining. Please contact support.")
            else:
                st.error("Invalid username or password")


def logout():
    # Clear session state on logout
    st.session_state.logged_in = False
    del st.session_state.username
    del st.session_state.tokens_remaining
    del st.session_state.tokens_consumed
    st.success("Logged out successfully!")
    st.experimental_rerun()  # Refresh to show logged-out state

st.markdown("")
with st.expander("**Modelle und Parameter**"):
                    st.session_state.temperature = st.slider(
                            "temperature",
	                        min_value=0.1,
                            max_value=1.0,
                            value=0.3,
                            step=0.1,
                            )
                    st.session_state.top_p = st.slider(
                            "top_p",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.95,
                            step=0.05,
                            )
		
st.markdown("""
<style>
    iframe {
        position: fixed;
        left: 0;
        right: 0;
        top: 0;
        bottom: 0;
        border: none;
        height: 100%;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        @keyframes gradientAnimation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .animated-gradient-text {
            font-family: "Graphik Semibold";
            font-size: 42px;
            background: linear-gradient(45deg, #22ebe8 30%, #dc14b7 55%, #fe647b 20%);
            background-size: 300% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientAnimation 20s ease-in-out infinite;
        }
    </style>
    <h3>
        KI zur Bewertung von Angebotserwiderungen: Bewertet Angebotserwiderungen!
    </h3>
""", unsafe_allow_html=True)

# This is the first API key input; no need to repeat it in the main function.
api_key = 'AIzaSyCiPGxwD04JwxifewrYiqzufyd25VjKBkw'

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def ocr_image(image):
    text = pytesseract.image_to_string(image)
    return text
	
def get_pdf_text(pdf_docs):
	
	text = "Response 1: "
	
	for pdf_path in pdf_docs:
		
		pdf_document = fitz.open(pdf_path)
		for page_num in range(len(pdf_document)):
			page = pdf_document.load_page(page_num)
			text += page.get_text()
			images = page.get_images(full=True)
			for img_index, img in enumerate(images):
				xref = img[0]
				base_image = pdf_document.extract_image(xref)
				image_bytes = base_image["image"]
				image_ext = base_image["ext"] 
				image = Image.open(io.BytesIO(image_bytes))
				text += ocr_image(image)
		pdf_document.close()
		text += "\n\nResponse 2: "

	return text

def user_input(api_key):
    st.write('inside input function')

def main():


    st.header("Angebotserwiderungen bewerten")
    
    st.markdown("""
    <style>
    input {
      border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Example PDF documents for testing
    pdf_docs = ["XYZ Consulting_withimage.pdf", "ABC Consulting Response.pdf"]
    
    if st.button("Beginnen Sie die Bewertung"):
        with st.spinner("Processing Responses..."):
            with st.spinner("Reading response documents..."):
                time.sleep(8)  # Simulating reading time
                raw_text = get_pdf_text(pdf_docs)
                
            # Standardize responses (example spinner blocks)
            with st.spinner("Standardizing Responses...."):
                time.sleep(5)
            with st.spinner("Inconsistent Response Found. Standardizing Sections...."):
                time.sleep(5)
            with st.spinner("Scanning for images..."):
                time.sleep(4)
            with st.spinner("Section wise Chunking Responses...."):
                time.sleep(4)
        
        with st.spinner("Embedding Text into vectors...."):
            time.sleep(3)
        
        with st.spinner("Evaluating Responses based on the scoring criteria"):
            time.sleep(6)
            with st.spinner("Drafting Response..."):
                prompt = ''' Consider yourself as bid evaluator who will evaluate bids received from different vendors basis the context provided and will generate score with explaination. I will provide you some context but before we jump into evaluation let's understand the bid. Below are the bid details for which we will be evaluating the responses: 
	              LCBO Background
	              The Liquor Control Board of Ontario (LCBO) is a leading global retailer and wholesaler of beverage alcohol, offering over 28,000 products from more than 80 countries. Through its Spirit of Sustainability (SoS) platform, launched in 2018, the LCBO supports Ontario’s social and environmental needs. Last year, it contributed over $16 million to community well-being and returned $2.55 billion to the province.
	          
	              RFP Objective
	              LCBO seeks a consulting services provider to develop and implement a five-year ESG strategy that aligns with SoS and establishes LCBO as a sustainability leader. Requirements include:
	          
	              Minimum of five years in ESG strategy development and implementation.
	              Expertise in the alcohol beverage and retail consumer goods industry, plus knowledge of government and environmental regulations.
	          
	              Scope of Work
	              Phase 1: ESG Research and Analysis
	          
	              Conduct internal and external ESG research.
	              Perform a double materiality assessment.
	          
	              Phase 2: ESG Strategy Development
	          
	              Design a five-year ESG strategy, roadmap, and action plan.
	              Align strategy with LCBO’s purpose and government mandates.
	              Innovate in ESG practices and industry collaboration.
	              Establish environmental and social initiatives.
	              Develop an impact measurement and reporting framework.
	          
	              Phase 3: ESG Strategy Execution
	          
	              Implement the action plan within financial projections.
	              Ensure alignment with organizational resources.
	              Produce LCBO ESG Annual Reports.
	              Track progress and adapt to emerging frameworks.
	          
	              Phase 4: Continued Support
	          
	              Continue executing the ESG strategy for the remaining 36 months.
	              Identify and implement new initiatives.
	              Provide ad-hoc support as needed.
	          
	              Evaluation Criteria
	              Company Qualifications - 5 points
	              Case Studies/Examples - 10 points
	              Team and Experience - 10 points
	              Work Plan, Approach and Methodology - 30 points
	          
	              Now you will evaluate both responses and return the detailed scoring result with table of scores for both Responses and rationale behind the scoring in another column. Rationale should be as detailed as possible. Do not mention LCBO in your response. And keep the detailed response.
	              Table format: Column 1 header - Criteria; Column 2 header - Response 1 (Company name); Column 3 header -Response 2 (company name); Column 4 header- Scoring Rationale
	              Provide another table for total score below the above table.
	              Total score table format: Column 1 header- Company Name; Column 2 header- Total Score which should be out of 55 points
	              Then provide the final recommendation paragraph explaining your opinion on evaluation. Try to be as detailed as possible in your response and provide summary in the end outside table. That can be your opinion what do you think is best option.
	              Here are the responses: {raw_text}
	              '''
                
                prompt = PromptTemplate(template=prompt, input_variables=["raw_text"])
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=st.session_state.temperature, google_api_key=api_key)
                chat_llm_chain = LLMChain(
	                llm=model,
	                prompt=prompt,
	                verbose=True
	            )	
                response = chat_llm_chain.predict(raw_text=raw_text)
		    # Calculate number of words in the response
                num_words = len(response.split())
            
                # Deduct tokens based on number of words
                token_cost = num_words  # Each word in the response costs 1 token (adjust as needed)
                if st.session_state.tokens_remaining > 0:
                    st.write(response)
                    st.session_state.tokens_consumed += token_cost  # Deduct tokens based on response length
                    st.session_state.tokens_remaining -= token_cost
                
                        # Update token count in JSON file
                    token_counts = read_token_counts()
                    token_counts[st.session_state.username] = st.session_state.tokens_remaining
                    write_token_counts(token_counts)
                else:
                    st.warning("Sie haben nicht genügend Token. Bitte kontaktieren Sie Ihren Administrator.")
            
                # Display remaining tokens to the user
                st.sidebar.text(f"Tokens Remaining: {st.session_state.tokens_remaining}")

if __name__ == "__main__":
    # with open('https://github.com/pranavGenAI/bidbooster/blob/475ae18b3c1f5a05a45ff983e06b025943137576/wave.css') as f:
        # css = f.read()
    # Ensure session state variables are initialized
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "tokens_consumed" not in st.session_state:
            st.session_state.tokens_consumed = 0
        if "tokens_remaining" not in st.session_state:
            st.session_state.tokens_remaining = 0
        if st.session_state.logged_in:
            st.sidebar.write(f"Welcome, {st.session_state.username}")
            st.sidebar.write(f"Tokens remaining: {st.session_state.tokens_remaining}")
            if st.sidebar.button("Abmelden"):
                logout()
            main()
        else:
            login()
# Custom CSS for the header and logo
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Graphik:wght@400;700&display=swap');

    body {
        background-color: #f0f0f0;
        color: black;
        font-family: 'Graphik', sans-serif;
    }
    .main {
        background-color: #f0f0f0;
    }
    .stApp {
        background-color: #f0f0f0;
    }
    header {
        background-color: #660094 !important;
        padding: 10px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .logo {
        height: 30px;
        width: auto;
        margin-right: 20px;  /* Space between logo and next item */
    }
    .header-content {
        display: flex;
        align-items: center;
    }
    .header-right {
        display: flex;
        align-items: center;
    }

    h1 {
        color: black;
        margin: 0;
        padding: 0;
    }

    .generated-text-box {
        border: 3px solid #A020F0; /* Thick border */
        padding: 20px;  
        border-radius: 10px; /* Rounded corners */
        color: black; /* Text color */
        background-color: #FFFFFF; /* Background color matching theme */
    }
    </style>
    """,
    unsafe_allow_html=True
)

