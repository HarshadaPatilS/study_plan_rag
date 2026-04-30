# 📚 StudyPlan AI — Personalized Study Plan Generator using RAG

> Upload your syllabus or notes → Get a custom day-by-day study schedule powered by RAG + LLM

---

## 🏗️ Architecture (Full RAG Pipeline)

```
PDF/TXT Upload
     ↓
PyMuPDF (text extraction)
     ↓
LangChain RecursiveCharacterTextSplitter (chunking)
     ↓
HuggingFace all-MiniLM-L6-v2 (local embeddings — FREE)
     ↓
FAISS Vector Store (local vector DB — FREE)
     ↓
Similarity Search (retrieve top-10 relevant chunks)
     ↓
Groq API → Llama 3.1 70B (LLM generation — FREE)
     ↓
Streamlit UI (personalized study plan displayed)
```

---

## 🚀 Quickstart (5 minutes)

### 1. Clone or download this folder

```bash
cd study_plan_rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your FREE Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Copy it

### 5. Set up your API key

```bash
cp .env.example .env
# Edit .env and paste your Groq API key
```

Or just enter it directly in the Streamlit UI.

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Folder Structure

```
study_plan_rag/
├── app.py                          # Streamlit UI — main entry point
├── requirements.txt                # All dependencies
├── .env.example                    # API key template
├── README.md                       # This file
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py               # PDF/TXT parsing + chunking
│   ├── vectorstore.py              # FAISS embedding + retrieval
│   └── planner.py                  # RAG prompt + Groq LLM call
└── data/
    └── sample_syllabus/
        └── dsa_syllabus.txt        # Test with this sample file
```

---

## 🛠️ Key Technologies

| Tool | Purpose | Cost |
|------|---------|------|
| **PyMuPDF (fitz)** | PDF text extraction | Free |
| **LangChain** | Text splitting, orchestration | Free |
| **all-MiniLM-L6-v2** | Text embeddings (runs locally) | Free |
| **FAISS** | Vector similarity search | Free |
| **Groq API** | LLM inference (Llama 3.1 70B) | Free tier |
| **Streamlit** | Web UI | Free |

---

## 💡 How to Use

1. **Get API key** from [console.groq.com](https://console.groq.com)
2. **Upload** your syllabus (PDF or .txt)
3. **Set preferences**: exam days, hours/day, weak topics
4. Click **Generate My Study Plan**
5. **Download** the plan as a .txt file

---

## 🧪 Testing without your own syllabus

Use the sample file:
```
data/sample_syllabus/dsa_syllabus.txt
```
Upload it in the UI to test the full pipeline.

---

## 🚀 Deploy for Free (Share with Interviewers!)

### Streamlit Cloud (recommended)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → deploy
4. Add `GROQ_API_KEY` in Streamlit Secrets

### HuggingFace Spaces
1. Create a Space with Streamlit SDK
2. Upload all files
3. Add API key as a Secret

---

## 🔧 Customization Ideas

- Add **multi-file upload** (upload multiple subjects at once)
- Add **calendar export** (.ics format)
- Add **Pomodoro session breakdown** per day
- Add **quiz generation** for each topic using the LLM
- Replace Groq with **Ollama** (fully offline, no API needed)

---

## ⚡ Offline Mode (No API Key)

Replace Groq with Ollama for fully local operation:

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1
```

Then in `utils/planner.py`, replace:
```python
from langchain_groq import ChatGroq
llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
```
With:
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.1")
```
