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

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "tokens_remaining" not in st.session_state:
    st.session_state.tokens_remaining = 500  # Default value or load from file

if "tokens_consumed" not in st.session_state:
    st.session_state.tokens_consumed = 0

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3  # Default value

if "top_p" not in st.session_state:
    st.session_state.top_p = 0.95  # Default value

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
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Define users and hashed passwords for simplicity
users = {
    "user": hash_password("user"),
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
    <h2>
        Bid Response Evaluation AI: Evaluates Bid responses!
    </h2>
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
                prompt = ''' Consider yourself as bid evaluator who will evaluate bids received from different vendors basis the context provided and will generate score with explaination in German langauge. I will provide you some context but before we jump into evaluation let's understand the bid. Below are the bid details for which we will be evaluating the responses: 
	              Kaufmännische Krankenkasse - KKH RFP Scope of Work:
			Scope of the RFP: Kaufmännische Krankenkasse (KKH) Digital Transformation Consultancy Services 
   			Objective: KKH aims to enhance its competitive edge and continue its digital transformation journey by engaging external consultancy services. The focus is on implementing cutting-edge technologies and applications to ensure operational stability and expand market position. 				              
			1. Los 1: UX/UI Design und Research
			Anforderungen:
			Entwicklung von benutzerzentrierten Designs.
			Durchführung von Nutzerforschung und Usability-Tests.
			Erstellung von Wireframes, Prototypen und Design-Systemen.
			Berücksichtigung von Barrierefreiheit und Responsivität.
			2. Los 2: App-Entwicklung
			Android-Entwicklung:
			Kenntnisse in Kotlin und Java.
			Erfahrung mit Android SDK und App-Architekturen.
			Integration von APIs und Backend-Systemen.
			iOS-Entwicklung:
			Kenntnisse in Swift und Objective-C.
			Erfahrung mit iOS SDK und App-Architekturen.
			Integration von APIs und Backend-Systemen.
			Erfahrung:
			Nachweisbare Projekte im Bereich App-Entwicklung.
			Technische Fähigkeiten:
			Kenntnisse in agilen Entwicklungsmethoden.
			Kommunikations- und Schulungsfähigkeiten:
			Fähigkeit zur Schulung von Endbenutzern und Stakeholdern.
			3. Los 3: Softwareentwicklung
			Anforderungen:
			Entwicklung von Softwarelösungen gemäß den Anforderungen.
			Erfahrung in verschiedenen Programmiersprachen (z.B. Java, C#, Python).
			Kenntnisse in Software-Architektur und Design-Patterns.
			4. Los 4: Test- und Qualitätsmanagement
			Anforderungen:
			Durchführung von Tests (funktional, automatisiert, manuell).
			Erstellung von Testplänen und -strategien.
			Sicherstellung der Softwarequalität und -sicherheit.
			5. Los 5: Input- und Prozessautomationsentwicklung
			Anforderungen:
			Entwicklung von Automatisierungslösungen zur Effizienzsteigerung.
			Kenntnisse in RPA (Robotic Process Automation) Tools.
			Analyse und Optimierung bestehender Prozesse.
			6. Los 6: Softwareentwicklung KI
			Anforderungen:
			Entwicklung von KI-gestützten Anwendungen.
			Kenntnisse in Machine Learning und Data Science.
			Erfahrung mit relevanten Frameworks (z.B. TensorFlow, PyTorch).
			7. Los 7: Telematikinfrastruktur
			Anforderungen:
			Kenntnisse in der Telematikinfrastruktur im Gesundheitswesen.
			Erfahrung mit der Integration von Systemen und Daten.
			Sicherstellung der Datensicherheit und -integrität.
			8. Los 8: IT-Security & IT-Forensik
			Anforderungen:
			Durchführung von Sicherheitsanalysen und Penetrationstests.
			Kenntnisse in IT-Forensik und Incident Response.
			Entwicklung von Sicherheitskonzepten und -richtlinien.
			9. Los 9: IT-Projektmanagement und PMO-Dienstleistungen
			Anforderungen:
			Planung, Durchführung und Überwachung von IT-Projekten.
			Kenntnisse in Projektmanagement-Methoden (z.B. PRINCE2, Scrum).
			Erstellung von Projektberichten und -dokumentationen.
			10. Los 10: Datenmigration Spezifikateur und ETL-Entwickler
			Anforderungen:
			Planung und Durchführung von Datenmigrationen.
			Kenntnisse in ETL-Prozessen (Extract, Transform, Load).
			Erfahrung mit Datenbankmanagementsystemen.
			11. Los 11: SAP-Experte für FI, CO, FS-CM
			Anforderungen:
			Expertise in SAP FI (Financial Accounting) und CO (Controlling).
			Erfahrung mit FS-CM (Financial Services Collections Management).
			Durchführung von Implementierungs- und Optimierungsprojekten.
			12. Los 12: Cloud
			Anforderungen:
			Kenntnisse in Cloud-Architekturen und -Diensten (z.B. AWS, Azure).
			Erfahrung mit Cloud-Migration und -Management.
			Sicherstellung der Sicherheit und Compliance in der Cloud.
			Diese Anforderungen bieten einen umfassenden Überblick über die spezifischen Fähigkeiten und Erfahrungen, die für jedes Los erforderlich sind, um die Qualität und Effizienz der angebotenen Dienstleistungen sicherzustellen.

	      
	      		Evaluation Criteria:
	      		1. Price (Preisbewertung): The cost of services is a primary factor, with different weighting given to price depending on the complexity and standardization of the services provided. Some lots may assess price as 100% of the evaluation if the services are highly standardized, where qualitative differences are minimal. 
			2. Quality of Service (Leistungsqualität): 
			Technical and Professional Qualification (Technische und Fachliche Qualifikation): This includes the depth of knowledge and experience of the proposed team, specifically related to the requirements of the specific lot. 
			Project Approach and Methodology (Projektansatz und Methodik): The bidder's approach to managing the project, including their methodology, project management capabilities, and innovative solutions. 
			3. Quality of the Submitted Concept (Qualität des eingereichten Konzepts): Evaluation of how well the proposal meets the RFP requirements and the innovativeness of the solutions proposed. 
			4. Experience (Erfahrung): 
			5. Relevant Project Experience (Relevante Projekterfahrung): Past success in similar projects, particularly within the same industry or with similar scopes, plays a crucial role. 
			6. References (Referenzen): Quality and relevance of the references provided, showing the bidder’s capability to deliver on projects of a similar scale and complexity. 
			7. Certifications (Zertifizierungen): Where applicable, the presence of industry-recognized certifications can be a factor, indicating the firm’s commitment to quality and adherence to industry standards. 
			8. Submission of a Complete and Compliant Proposal (Vollständigkeit und Konformität des Angebots): Proposals must be complete and adhere strictly to the RFP’s submission guidelines. Compliance with all formal requirements, clarity, and comprehensiveness of the proposal documentation is essential. 
			9. Innovative Value Adds (Innovative Zusatzleistungen): Proposals that include additional value beyond the basic requirements can score higher. These could be innovative uses of technology, additional services at no extra cost, or sustainability measures. 
          
	              Now you will evaluate both responses and return the detailed scoring result with table of scores for both Responses and rationale behind the scoring in another column. Rationale should be as detailed as possible.  And keep the detailed response.
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
               
if __name__ == "__main__":
    # with open('https://github.com/pranavGenAI/bidbooster/blob/475ae18b3c1f5a05a45ff983e06b025943137576/wave.css') as f:
        # css = f.read()
    # Ensure session state variables are initialized
    if st.session_state.logged_in:
        col1,col2,col3 = st.columns([10,10,1.5])
        with col3:
            if st.button("Logout"):
                logout()
        main()
    else:
        login()

# Custom CSS for the header and logo
# Custom CSS for the header and logo
if st.session_state.logged_in:
	st.markdown("")
	# with st.expander("**Modelle und Parameter**"):
	#                     st.session_state.temperature = st.slider(
	#                             "temperature",
	# 	                        min_value=0.1,
	#                             max_value=1.0,
	#                             value=0.3,
	#                             step=0.1,
	#                             )
	#                     st.session_state.top_p = st.slider(
	#                             "top_p",
	#                             min_value=0.1,
	#                             max_value=1.0,
	#                             value=0.95,
	#                             step=0.05,
	#                             )
		
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


# Adding the logo and other elements in the header st-emotion-cache-18ni7ap ezrtsby2
st.markdown(
    f"""
    <header tabindex="-1" data-testid="stHeader" class="st-emotion-cache-18ni7ap ezrtsby2">
        <div data-testid="stDecoration" id="stDecoration" class="st-emotion-cache-1dp5vir ezrtsby1"></div>
        <div class="header-content">
            <!-- Add the logo here -->
            <img src="https://www.vgen.it/wp-content/uploads/2021/04/logo-accenture-ludo.png" class="logo" alt="Logo">
        
    </header>

    """,
    unsafe_allow_html=True
)
