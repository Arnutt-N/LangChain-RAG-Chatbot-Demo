import streamlit as st
import os
import fnmatch
import io
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tempfile

# Set page configuration
st.set_page_config(layout="wide", page_title="Gen AI : RAG Chatbot with Documents")

# Load environment variables
# load_dotenv()

# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Ensure the script stops execution if the API key is not set
if api_key is None:
    st.error("OPENAI_API_KEY is not set. Please set it in the Streamlit Cloud secrets.")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Translations
translations = {
    "en": {
        "title": "Gen AI : RAG Chatbot with Documents (Demo)",
        "upload_button": "Upload Documents",
        "browse_files": "Browse files",
        "ask_placeholder": "Ask a question in Thai or English...",
        "processing": "Processing...",
        "welcome": "Hello! I'm ready to chat about various topics based on the documents. How can I assist you today?",
        "upload_success": lambda count: f"{count} new document(s) uploaded and processed successfully!",
        "local_knowledge": "My Documents",
        "thinking": "Thinking...",
        "language": "Language / ภาษา",
        "clear_chat": "Clear Chat",
    },
    "th": {
        "title": "Gen AI : RAG Chatbot with Documents (Demo)",
        "upload_button": "อัปโหลดเอกสาร",
        "browse_files": "เลือกไฟล์",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "processing": "กำลังประมวลผล...",
        "welcome": "สวัสดี! ฉันพร้อมที่จะพูดคุยเกี่ยวกับหัวข้อต่างๆ ตามเอกสารที่มี วันนี้ฉันจะช่วยคุณอย่างไรดี?",
        "upload_success": lambda count: f"อัปโหลดและประมวลผลเอกสารใหม่ {count} ฉบับสำเร็จแล้ว!",
        "local_knowledge": "คลังข้อมูลของฉัน",
        "thinking": "กำลังคิด...",
        "language": "ภาษา / Language",
        "clear_chat": "ล้างการแชท",
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "local_files" not in st.session_state:
    st.session_state.local_files = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Function to load .gitignore patterns
def load_gitignore():
    patterns = []
    if os.path.exists('.gitignore'):
        encodings = ['utf-8', 'cp874', 'tis-620', 'windows-1252', 'latin-1']
        for encoding in encodings:
            try:
                with open('.gitignore', 'r', encoding=encoding) as file:
                    patterns = file.read().splitlines()
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                continue  # If unsuccessful, try the next encoding
    return patterns

# Function to check if a file should be ignored
def should_ignore(filename, patterns):
    if filename == 'requirements.txt':
        return True
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

# Load documents
def load_documents(file_paths, uploaded_files):
    documents = []
    for file_path in file_paths:
        if file_path == 'requirements.txt':
            continue
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            loader = UnstructuredExcelLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Use PyPDFLoader with the temporary file path
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())

            # Remove the temporary file
            os.unlink(temp_file_path)
        elif uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(io.StringIO(uploaded_file.getvalue().decode()))
            documents.extend(loader.load())
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(io.StringIO(uploaded_file.getvalue().decode()))
            documents.extend(loader.load())
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            loader = UnstructuredExcelLoader(io.BytesIO(uploaded_file.getvalue()))
            documents.extend(loader.load())
    
    return documents

# Process documents
def process_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Setup retrieval chain
def setup_retrieval_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    return qa_chain

# Function to clear all uploaded files and reset vectorstore
def clear_uploaded_files():
    st.session_state.uploaded_files = []
    st.session_state.vectorstore = None

# Function to refresh local files
def refresh_local_files():
    ignore_patterns = load_gitignore()
    st.session_state.local_files = [f for f in os.listdir('.') if f.endswith(('.pdf', '.csv', '.txt', '.xlsx', '.xls')) and not should_ignore(f, ignore_patterns) and f != 'requirements.txt']

def main():
    if "language" in st.query_params:
        st.session_state.language = st.query_params["language"]

    t = translations[st.session_state.language]

    # Custom CSS
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTitle {
            text-align: center;
        }
        .footer {
            position: fixed;
            left: 50%;
            bottom: 0;
            transform: translateX(-50%);
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: #545454;
        }
        .compact-container {
            margin-bottom: 1rem;
        }
        .spacer {
            margin-bottom: 1rem;
        }
        .sidebar-label {
            font-size: 14px;
            font-weight: normal;
            margin-bottom: 0.5rem;
        }
        .uploaded-docs-header {
            font-size: 16px;
            font-weight: bold;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stChatInputContainer {
            max-width: 800px;
            margin: 0 auto;
        }
        .stButton > button {
            margin-top: 1.0rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Language selection moved to the top
        st.markdown(f"<div class='sidebar-label'>{t['language']}</div>", unsafe_allow_html=True)
        selected_lang = st.selectbox(
            "", 
            options=["ไทย", "English"], 
            index=1 if st.session_state.language == "en" else 0, 
            key="language_selection",
            label_visibility="collapsed"
        )
        if selected_lang == "ไทย":
            new_language = "th"
        else:
            new_language = "en"
        
        if new_language != st.session_state.language:
            st.session_state.language = new_language
            st.query_params["language"] = new_language
            st.rerun()
        
        # Spacer between language selection and file uploader
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        # Add "Upload Documents" text above the file uploader with consistent styling
        st.markdown(f"<div class='sidebar-label'>{t['upload_button']}</div>", unsafe_allow_html=True)

        # File uploader with translated "Browse files" button
        uploaded_files = st.file_uploader(
            "", 
            accept_multiple_files=True, 
            type=['pdf', 'csv', 'txt', 'xlsx', 'xls'], 
            key="file_uploader",
            label_visibility="collapsed",
            help=t["upload_button"]
        )

        # Custom "Browse files" button with correct translation
        st.markdown(
            f"""
            <script>
                var uploadButton = window.parent.document.querySelector('.stFileUploader label.st-eb');
                if (uploadButton) {{
                    uploadButton.textContent = "{t['browse_files']}";
                }}
            </script>
            """,
            unsafe_allow_html=True
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(t["upload_success"](len(uploaded_files)))
            st.session_state.vectorstore = None  # Reset vectorstore to force reprocessing

    # Main content
    st.title(t["title"])

    # Display local knowledge base
    refresh_local_files()  # Always refresh local files when rendering the main content
    if st.session_state.local_files:
        st.markdown(f"<div class='uploaded-docs-header'>{t['local_knowledge']}</div>", unsafe_allow_html=True)
        for file in st.session_state.local_files:
            st.write(f"- {file}")

    # Chat interface
    if st.session_state.vectorstore is None:
        documents = load_documents(st.session_state.local_files, st.session_state.uploaded_files)
        st.session_state.vectorstore = process_documents(documents)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(t["ask_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retrieval_chain = setup_retrieval_chain(st.session_state.vectorstore)
            with st.spinner(t["thinking"]):
                response = retrieval_chain({"question": prompt})
            st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

    # Clear chat button
    if st.button(t["clear_chat"]):
        st.session_state.messages = []
        clear_uploaded_files()
        st.rerun()

    # Footer
    st.markdown(
        '<div class="footer">Created by Arnutt Noitumyae, 2024</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    if "language" in st.query_params:
        st.session_state.language = st.query_params["language"]
    main()
