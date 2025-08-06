import streamlit as st
import openai
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import io
import PyPDF2
from docx import Document
import re

# Try to import OCR dependencies
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("⚠️ OCR dependencies not available. Image-based PDFs cannot be processed.")

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


def extract_text_with_ocr(file_bytes):
    """Extract text from image-based PDF using OCR"""
    if not OCR_AVAILABLE:
        st.error("OCR not available in this deployment. Please use text-searchable PDFs.")
        return ""

    try:
        # Convert PDF pages to images
        images = convert_from_bytes(file_bytes)
        text = ""

        with st.spinner(f"Processing image-based PDF with OCR... ({len(images)} pages)"):
            for i, image in enumerate(images):
                # Use OCR to extract text from each page
                page_text = pytesseract.image_to_string(image, config='--psm 6')
                text += f"Page {i + 1}:\n{page_text}\n\n"

        return text.strip()
    except Exception as e:
        st.error(f"OCR processing failed: {str(e)}")
        return ""


def extract_text_from_pdf(file):
    """Extract text from PDF (handles both text-based and image-based PDFs)"""
    try:
        # Read file bytes
        file.seek(0)
        file_bytes = file.read()

        # First try regular text extraction
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n"

        # Check if we got meaningful text (not just whitespace/garbled)
        clean_text = text.strip()
        word_count = len(clean_text.split())

        if word_count >= 10:  # If we have at least 10 words, assume it's readable
            return clean_text

        # If text extraction failed, try OCR if available
        if OCR_AVAILABLE:
            st.info("📄 PDF appears to be image-based or scanned. Using OCR to extract text...")
            ocr_text = extract_text_with_ocr(file_bytes)

            if len(ocr_text.strip()) < 50:
                st.warning("⚠️ OCR extraction yielded minimal text. Please check if the PDF is readable.")

            return ocr_text
        else:
            st.warning(
                f"⚠️ {file.name} appears to be image-based or scanned. OCR not available - please use text-searchable PDFs.")
            return ""

    except Exception as e:
        st.error(f"Error processing PDF {file.name}: {str(e)}")
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


def normalize_similarity_score(similarity_scores: List[float]) -> List[float]:
    """Normalize similarity scores to make them more interpretable"""
    if not similarity_scores:
        return []

    min_score = min(similarity_scores)
    max_score = max(similarity_scores)

    # Avoid division by zero
    if max_score == min_score:
        return [0.5] * len(similarity_scores)

    # Normalize to 0-1 range, then scale to 0.3-1.0 for better presentation
    normalized = []
    for score in similarity_scores:
        norm_score = (score - min_score) / (max_score - min_score)
        scaled_score = 0.3 + (norm_score * 0.7)  # Scale to 0.3-1.0 range
        normalized.append(scaled_score)

    return normalized


def get_match_quality(raw_score: float, normalized_score: float) -> str:
    """Get match quality description"""
    if normalized_score >= 0.8:
        return "🟢 Excellent Match"
    elif normalized_score >= 0.6:
        return "🟡 Good Match"
    elif normalized_score >= 0.4:
        return "🟠 Moderate Match"
    else:
        return "🔴 Weak Match"


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
    You are a hiring manager reviewing a candidate. Analyze their experience objectively.

    INSTRUCTIONS:
    - For strong matches: Mention 1-2 specific projects/achievements that align with requirements
    - For weak matches: Acknowledge relevant skills BUT mention key gaps or limitations
    - Start with strengths, then mention "however" or "but lacks" for significant gaps
    - Don't assume they fit - be honest about mismatches
    - Keep between 60-80 words for detailed assessment

    JOB REQUIREMENTS:
    {job_description}

    CANDIDATE RESUME:
    {resume_text}

    CANDIDATE ASSESSMENT:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an honest hiring manager. For strong candidates, highlight relevant experience. For weaker candidates, acknowledge skills but clearly mention gaps using 'however' or 'but lacks'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Increased for longer responses
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate summary: {str(e)}"


def extract_candidate_name(resume_text: str, filename: str) -> str:
    """Try to extract candidate name from resume text or filename"""
    lines = resume_text.split('\n')[:10]  # Check first 10 lines

    # Look for name patterns in the text
    for line in lines:
        line = line.strip()
        # Look for lines that might be names (2-3 words, alphabetic)
        words = line.split()
        if 2 <= len(words) <= 3 and all(word.replace('-', '').replace("'", '').isalpha() for word in words):
            # Additional checks to avoid headers like "WORK EXPERIENCE"
            if not any(keyword in line.upper() for keyword in
                       ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'RESUME', 'CV', 'PROFILE']):
                return line

    # Fallback: try to extract from filename
    base_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
    # Clean up common patterns
    base_name = base_name.replace('Resume_', '').replace('_Resume', '').replace('CV_', '').replace('_CV', '')
    base_name = base_name.replace('_', ' ').replace('-', ' ')

    if base_name and not base_name.lower().startswith('untitled'):
        return base_name

    return f"Candidate {filename[:8]}"  # Use first 8 chars of filename as ID


# Streamlit App
def main():
    st.title("🎯 Candidate Recommendation Engine")
    st.markdown("*Find the best candidates for your job opening*")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    num_candidates = st.sidebar.slider("Number of top candidates to show", 5, 10, 10)
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
            help="Upload PDF (text or image-based), DOCX, or TXT files. OCR will handle scanned/image PDFs if available."
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
                candidate_name = extract_candidate_name(resume_text, file.name)

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

        # Calculate normalized scores for better presentation
        raw_scores = [c['similarity'] for c in top_candidates]
        normalized_scores = normalize_similarity_score(raw_scores)

        # Add normalized scores to candidates
        for i, candidate in enumerate(top_candidates):
            candidate['normalized_score'] = normalized_scores[i]

        # Display results
        st.header("🏆 Top Candidate Matches")

        for i, candidate in enumerate(top_candidates, 1):
            match_quality = get_match_quality(candidate['similarity'], candidate['normalized_score'])

            with st.expander(f"#{i} - {candidate['name']} - {match_quality}", expanded=i <= 5):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**File:** {candidate['filename']}")
                    st.markdown(f"**Raw Similarity:** {candidate['similarity']:.3f}")
                    st.markdown(f"**Normalized Score:** {candidate['normalized_score']:.3f}")

                    if show_summaries:
                        with st.spinner("Generating AI summary..."):
                            summary = generate_candidate_summary(
                                job_description,
                                candidate['resume_text'],
                                candidate['similarity']
                            )
                            st.markdown(f"**Assessment:** {summary}")

                with col2:
                    # Raw similarity score
                    raw_percentage = candidate['similarity'] * 100
                    st.metric("Raw Match %", f"{raw_percentage:.1f}%")

                with col3:
                    # Normalized score visualization
                    norm_percentage = candidate['normalized_score'] * 100
                    st.metric("Adjusted Match %", f"{norm_percentage:.1f}%")

        # Summary statistics
        st.header("📊 Summary")
        avg_score = np.mean([c['similarity'] for c in top_candidates])
        st.metric("Average Similarity Score", f"{avg_score:.3f}")


if __name__ == "__main__":
    main()