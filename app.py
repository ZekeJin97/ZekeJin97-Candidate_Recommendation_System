import streamlit as st
import openai
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import io
import PyPDF2
from docx import Document
import re

# Set page config
st.set_page_config(
    page_title="Candidate Recommendation Engine",
    page_icon="🎯",
    layout="wide"
)


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    # For demo purposes - in production, use environment variables
    api_key = st.secrets.get("OPENAI_API_KEY", "your-api-key-here")
    return openai.OpenAI(api_key=api_key)


client = get_openai_client()


def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except:
        return ""


def extract_text_from_docx(file):
    """Extract text from uploaded DOCX file"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except:
        return ""


def extract_text_from_file(file):
    """Extract text from uploaded file based on type"""
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        st.error(f"Unsupported file type: {file.type}")
        return ""


def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip()
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return []


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def generate_candidate_summary(job_description: str, resume_text: str, similarity_score: float) -> str:
    """Generate AI summary of why candidate is a good fit"""
    prompt = f"""
    Based on the job description and candidate's resume, explain in 2-3 sentences why this candidate is a good fit for this role.
    Focus on specific skills, experience, or qualifications that align.

    Job Description:
    {job_description[:1000]}...

    Candidate Resume:
    {resume_text[:1000]}...

    Similarity Score: {similarity_score:.3f}

    Summary:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful HR assistant that explains why candidates match job requirements."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate summary: {str(e)}"


def extract_candidate_name(resume_text: str) -> str:
    """Try to extract candidate name from resume text"""
    lines = resume_text.split('\n')[:5]  # Check first 5 lines

    for line in lines:
        line = line.strip()
        # Simple heuristic: look for lines that might be names
        if len(line.split()) == 2 and line.replace(' ', '').isalpha():
            return line

    return "Unknown Candidate"


# Streamlit App
def main():
    st.title("🎯 Candidate Recommendation Engine")
    st.markdown("*Find the best candidates for your job opening*")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    num_candidates = st.sidebar.slider("Number of top candidates to show", 3, 15, 10)
    show_summaries = st.sidebar.checkbox("Generate AI summaries", value=True)

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Job Description")
        job_description = st.text_area(
            "Enter the job description:",
            height=200,
            placeholder="e.g., We are looking for a Machine Learning Engineer with experience in Python, deep learning frameworks like PyTorch/TensorFlow..."
        )

    with col2:
        st.header("Upload Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )

        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} resume(s)")

    # Process button
    if st.button("🚀 Find Best Candidates", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description")
            return

        if not uploaded_files:
            st.error("Please upload at least one resume")
            return

        # Processing
        with st.spinner("Processing resumes and calculating similarities..."):
            # Get job description embedding
            job_embedding = get_embedding(job_description)

            if not job_embedding:
                st.error("Failed to process job description")
                return

            # Process each resume
            candidates = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                # Extract text from resume
                resume_text = extract_text_from_file(file)

                if not resume_text.strip():
                    st.warning(f"Could not extract text from {file.name}")
                    continue

                # Get resume embedding
                resume_embedding = get_embedding(resume_text)

                if not resume_embedding:
                    continue

                # Calculate similarity
                similarity = calculate_cosine_similarity(job_embedding, resume_embedding)

                # Extract candidate name
                candidate_name = extract_candidate_name(resume_text)

                candidates.append({
                    'name': candidate_name,
                    'filename': file.name,
                    'similarity': similarity,
                    'resume_text': resume_text
                })

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

            progress_bar.empty()

        # Sort by similarity and get top candidates
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        top_candidates = candidates[:num_candidates]

        if not top_candidates:
            st.error("Could not process any resumes successfully")
            return

        # Display results
        st.header("🏆 Top Candidate Matches")

        for i, candidate in enumerate(top_candidates, 1):
            with st.expander(f"#{i} - {candidate['name']} (Score: {candidate['similarity']:.3f})", expanded=i <= 3):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**File:** {candidate['filename']}")
                    st.markdown(f"**Similarity Score:** {candidate['similarity']:.3f}")

                    if show_summaries:
                        with st.spinner("Generating AI summary..."):
                            summary = generate_candidate_summary(
                                job_description,
                                candidate['resume_text'],
                                candidate['similarity']
                            )
                            st.markdown(f"**Why this candidate fits:** {summary}")

                with col2:
                    # Similarity score visualization
                    score_percentage = candidate['similarity'] * 100
                    st.metric("Match %", f"{score_percentage:.1f}%")

        # Summary statistics
        st.header("📊 Summary")
        avg_score = np.mean([c['similarity'] for c in top_candidates])
        st.metric("Average Similarity Score", f"{avg_score:.3f}")


if __name__ == "__main__":
    main()