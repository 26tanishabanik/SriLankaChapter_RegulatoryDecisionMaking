import streamlit as st
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from PIL import Image
import os
import time
from dotenv import load_dotenv
import requests
import base64
import json
import datetime

load_dotenv()


GITHUB_USER = os.getenv("GITHUB_USER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Repository details
GITHUB_REPO_SLUG = "26tanishabanik/SriLankaChapter_RegulatoryDecisionMaking"
GITHUB_BRANCH = "deploy/streamlit"

if not GITHUB_USER or not GITHUB_TOKEN:
    st.error("GitHub credentials (GITHUB_USER and GITHUB_TOKEN) are not set. Please configure them as environment variables.")


def push_to_repo_branch(gitHubFileName, fileName, repo_slug, branch):
    try:
        branch_url = f"https://api.github.com/repos/{repo_slug}/branches/{branch}"
        response = requests.get(branch_url, auth=(GITHUB_USER, GITHUB_TOKEN))

        if not response.ok:
            st.error(f"Failed to retrieve branch info: {response.text}")
            return

        # Get the tree URL for the branch's latest commit
        tree_url = response.json()["commit"]["commit"]["tree"]["url"]
        tree_response = requests.get(tree_url, auth=(GITHUB_USER, GITHUB_TOKEN))

        if not tree_response.ok:
            st.error(f"Failed to retrieve the tree: {tree_response.text}")
            return

        sha = None
        for file in tree_response.json()["tree"]:
            if file["path"] == gitHubFileName:
                sha = file["sha"]
                break

        # Read the local file and encode its content
        with open(fileName, "rb") as file:
            content = base64.b64encode(file.read()).decode("utf-8")

        # Prepare the data payload for the API request
        commit_message = f"Automated update {datetime.datetime.now()}"
        data = {
            "message": commit_message,
            "content": content,
            "branch": branch,
        }
        if sha:
            data["sha"] = sha

        # Make the PUT request to update or create the file in the repo
        file_url = f"https://api.github.com/repos/{repo_slug}/contents/{gitHubFileName}"
        put_response = requests.put(file_url, auth=(GITHUB_USER, GITHUB_TOKEN), json=data)

        if put_response.ok:
            st.success(f"File '{gitHubFileName}' successfully pushed to GitHub!")
        else:
            st.error(f"Failed to push the file: {put_response.text}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.set_page_config(
    page_title="Sri Lanka Regulatory Archives Q&A",
    page_icon="📜",
    layout="wide"
)

# Sidebar Navigation
with st.sidebar:
    st.image("assets/SriLankaChapterLogo.jpeg", use_container_width=True)
    st.title("Sri Lanka Chapter Project")
    st.markdown("""
    **Navigate:**
    - **Home**: Main Q&A Application
    - **Collaborators**: Learn about our contributors
    - **Chapter Details**: About this chapter
    """)

page = st.sidebar.radio("Select Page", ["Home", "Collaborators", "Chapter Details"])

if page == "Home":
    # Home Page
    st.title("📜 Sri Lanka Tea Estate Digital Regulatory Archive Q&A System")
    st.markdown("""
    This tool digitizes physical archives and provides an AI-powered Q&A system to retrieve relevant documents for decision-making in industries like tea.
    """)

    st.header("📂 Upload Documents")
    uploaded_file = st.file_uploader("Upload a Text file", type=["txt"])
    
    loading_placeholder = st.empty()
    
    if uploaded_file:
        local_dir = "uploaded_files"
        os.makedirs(local_dir, exist_ok=True)
        local_file_path = os.path.join(local_dir, uploaded_file.name)
    
        with open(local_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved locally at {local_file_path}")
    
        
        if st.button("Push to GitHub"):
            push_to_repo_branch(
                gitHubFileName=uploaded_file.name,
                fileName=local_file_path,
                repo_slug=GITHUB_REPO_SLUG,
                branch=GITHUB_BRANCH
            )
        file_name = uploaded_file.name
        loading_placeholder.info("📜 Processing the document...")
        with st.spinner("Reading text from the document..."):
            document_text = uploaded_file.read().decode("utf-8")
        with st.spinner("Generating embeddings..."):
            # Initialize the HuggingFace embeddings model
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            document_chunks = [Document(page_content=document_text)]
            vector_store = FAISS.from_documents(document_chunks, embeddings)
        with st.spinner("Setting up RAG system..."):
            retriever = vector_store.as_retriever()
            tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
            model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xl')
            hf_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2
            )
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
            )
        
        st.success("Document processed successfully!")
        loading_placeholder.empty()
        st.header("💬 Ask Questions")
        question = st.text_input("Ask a question about the uploaded document:")
        if question:
            with st.spinner("Searching for the best answer..."):
                answer = qa_chain.run(question)
            st.markdown(f"**Answer:** {answer}")
    
elif page == "Collaborators":
    st.title("👥 Collaborators")
    st.markdown("""
    ### Project Contributors:
    - **[Your Name]**: Lead Developer  
    - **[Contributor 2]**: Data Specialist  
    - **[Contributor 3]**: AI Researcher  
    - **[Contributor 4]**: Domain Expert  
    """)

elif page == "Chapter Details":
    st.title("🌍 About the Sri Lanka Chapter")
    st.markdown("""
    This initiative is part of the Sri Lanka Chapter's efforts to leverage AI for solving local challenges.  
    By digitizing and enhancing access to archival records, we aim to revolutionize regulatory decision-making processes in critical industries.
    """)
    st.image("assets/SriLankaChapterLogo.jpeg", use_container_width=True)
