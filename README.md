# 🎯 Candidate Recommendation Engine

A Streamlit web application that matches candidates to job descriptions using AI embeddings and cosine similarity.

**Built for SproutsAI ML Engineer Internship Assignment**

## 🚀 Live Demo

**[Try the app here: https://zekejin97-candidaterecommendationsystem-sprouts.streamlit.app/](https://zekejin97-candidaterecommendationsystem-sprouts.streamlit.app/)**

## ✨ Features

- **Job Description Input**: Enter job requirements via text area
- **Resume Upload**: Support for PDF, DOCX, and TXT files
- **OCR Support**: Handles both text-based and scanned/image PDFs
- **AI-Powered Matching**: Uses OpenAI embeddings for semantic similarity
- **Smart Ranking**: Cosine similarity calculation with normalized scores
- **AI Summaries**: GPT-generated explanations for why candidates match
- **Export Results**: Download rankings as CSV
- **Session Persistence**: Results remain available during session

## 🛠️ Technical Approach

### 1. Text Extraction
- **PDF**: PyPDF2 for text-based PDFs, OCR fallback for scanned documents
- **DOCX**: python-docx for Word documents
- **TXT**: Direct text reading

### 2. Embedding Generation
- **Model**: OpenAI `text-embedding-3-small`
- **Choice Rationale**: Balanced performance/cost, optimized for semantic similarity tasks, 1536 dimensions
- **Process**: Convert both job descriptions and resumes to vector embeddings
- **Caching**: 1-hour cache to avoid redundant API calls

### 3. Similarity Calculation
- **Method**: Cosine similarity between job and resume vectors
- **Normalization**: Scores normalized to 0.3-1.0 range for better interpretation
- **Ranking**: Candidates sorted by similarity score

### 4. AI Assessment (Bonus Feature)
- **Model**: GPT-3.5-turbo for natural language summaries
- **Choice Rationale**: Cost-effective for text generation, good reasoning capabilities for candidate assessment
- **Focus**: Work experience, skills, and project relevance
- **Accuracy**: Only mentions skills explicitly found in resumes

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd candidate-recommendation-engine

# Install dependencies
pip install streamlit openai numpy pandas PyPDF2 python-docx

# Optional: Install OCR dependencies for scanned PDFs
pip install pytesseract pillow pdf2image

# Set up Streamlit secrets
# Create .streamlit/secrets.toml with:
OPENAI_API_KEY = "your-openai-api-key-here"

# Run the app
streamlit run app.py
```

## 💻 Usage

1. **Enter Job Description**: Paste the job requirements in the left column
2. **Upload Resumes**: Upload candidate files (PDF/DOCX/TXT) in the right column
3. **Process**: Click "Find Best Candidates" to run the analysis
4. **Review Results**: View ranked candidates with similarity scores and AI summaries
5. **Export**: Download results as CSV for further analysis

## 🔧 Configuration

- **Number of candidates**: Adjust via sidebar slider (5-10)
- **AI summaries**: Toggle on/off via sidebar checkbox
- **File types**: PDF, DOCX, TXT supported
- **OCR**: Automatically handles scanned PDFs when dependencies available

## 📊 Output

For each candidate, the app provides:
- **Name**: Extracted from resume or filename
- **Raw similarity score**: Direct cosine similarity (0-1)
- **Normalized score**: Scaled for better interpretation
- **Match quality**: Color-coded rating (Excellent/Good/Moderate/Weak)
- **AI assessment**: GPT-generated explanation of candidate fit

## 🎯 Key Assumptions

- **Resume format**: Assumes standard resume structure with name near the top
- **Text quality**: Works best with well-formatted, readable text
- **Language**: Optimized for English-language resumes
- **Skill matching**: Focuses on explicit skills mentioned, not inferred capabilities
- **File size**: Reasonable file sizes (< 10MB per resume recommended)

## 🚀 Deployment

The app is deployed on Streamlit Cloud for easy access. To deploy your own instance:

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add `OPENAI_API_KEY` to Streamlit secrets
4. Deploy

## 🛡️ Error Handling

- **File processing errors**: Graceful handling of corrupted/unreadable files
- **API failures**: Informative error messages for OpenAI API issues
- **OCR fallback**: Automatic fallback when OCR dependencies unavailable
- **Empty results**: Clear messaging when no candidates can be processed

## 📈 Performance

- **Caching**: Embeddings cached for 1 hour to reduce API calls
- **Progress tracking**: Real-time progress bars for long operations
- **Session state**: Results persist during user session
- **Efficient processing**: Batch operations where possible

## 🔮 Future Enhancements

- Support for additional file formats (RTF, HTML)
- Bulk job description processing
- Advanced filtering options
- Integration with ATS systems
- Multi-language support
- Custom embedding models

---

*Built with ❤️ using Streamlit, OpenAI, and modern ML techniques*