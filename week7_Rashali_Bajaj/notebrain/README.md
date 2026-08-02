# NoteBrain QA

NoteBrain QA is a local, privacy-focused Streamlit RAG application designed for querying personal and study notes (PDFs and TXT documents).

## Setup & Running Instructions

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy `.env.example` to `.env` and set your `GROQ_API_KEY`:
   ```bash
   copy .env.example .env
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
