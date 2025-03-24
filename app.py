import streamlit as st
import pandas as pd
import PyPDF2
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from resumes
def extract_text_from_resume(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function to extract key fields from resumes
def extract_resume_details(text):
    doc = nlp(text)
    skills = []
    education = []
    experience = []
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            education.append(ent.text)
        elif ent.label_ == "DATE":
            experience.append(ent.text)
        elif ent.label_ == "SKILL":
            skills.append(ent.text)
    
    return {
        "Skills": ", ".join(set(skills)),
        "Education": ", ".join(set(education)),
        "Experience": ", ".join(set(experience))
    }

# Streamlit UI
st.title("AI-powered Resume Screening and Ranking System")

# Upload Job Description
st.subheader("Upload Job Description")
job_description = st.text_area("Paste the job description here")

# Upload Resumes
st.subheader("Upload Resumes")
resume_files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Process and Rank Candidates") and job_description and resume_files:
    job_description = preprocess_text(job_description)
    resumes = []
    extracted_details = []
    
    for file in resume_files:
        text = extract_text_from_resume(file)
        text = preprocess_text(text)
        details = extract_resume_details(text)
        resumes.append((file.name, text, details))
    
    # Convert to DataFrame
    df = pd.DataFrame(resumes, columns=["Candidate Name", "Resume Text", "Details"])
    df[["Skills", "Education", "Experience"]] = df["Details"].apply(pd.Series)
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    all_texts = [job_description] + df["Resume Text"].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df["Match Score"] = similarity_scores * 100  # Convert to percentage
    df = df.sort_values(by="Match Score", ascending=False)
    
    # Display results
    st.subheader("Candidate Ranking")
    st.dataframe(df[["Candidate Name", "Match Score", "Skills", "Education", "Experience"]])