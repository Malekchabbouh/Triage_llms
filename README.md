# Triage HIL — Streamlit App

## Project Structure
- `APP/` → Main application (final working system)
- `Embeddings/` → Reference materials used during development (not required to run)
- `Evaluation/` → Evaluation scripts and experiments (for analysis only)

Note:
The **final database and runnable system are already inside the `APP` folder**.  
You only need the `APP` folder to run the project.

---
## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app_final.py
```

## Usage
1. Open the app in your browser
2. **Patient Dashboard**: 
   - Submit New Case: Enter National ID and describe symptoms
   - System auto-generates ticket number (T001, T002, etc.)
   - Answer follow-up questions if prompted (max 3 rounds)
   - Check Results: Enter ticket number to view final decision and booking details
3. **Nurse Dashboard** (Login required: haya/123, malek/123, yomna/123, admin/admin2026):
   - Review Cases: View AI triage assessment with urgency, RAG mode, and agent confidence
   - See retrieved medical evidence, clinical reasoning, and nurse guidance
   - Approve or override AI decision with required justification
   - Review History: View all past cases with AI vs Nurse decisions
4. **Developer Dashboard**:
   - View system status (RAG, Scheduler, CrewAI, Groq)
   - Analyze AI vs Nurse agreement rates and override patterns
   - Run cross-case learning agent for bias detection
   - Inspect two-agent CrewAI traces and evaluator feedback

## API Key
Set `GROQ_API_KEY` as an environment variable (required for production mode):
```bash
# Linux/Mac
export GROQ_API_KEY=gsk_your-key-here
streamlit run app_final.py

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your-key-here"
streamlit run app_creawai.py
```

Alternative (Gemini - legacy support):
```bash
export GEMINI_API_KEY=your-key-here
# or
export GOOGLE_API_KEY=your-key-here
```

If no API key is set, the app will show an error on first use.

## Data Files Required
- `chunked_docs_phase2.json` — RAG knowledge base (1108 chunks)
- `agentic_triage_schedules.xlsx` — Doctor schedules (Emergency + Routine sheets)

## Database
- Auto-created SQLite database: `triage_hil.db`
- Auto-migration: adds new columns without data loss
- Export from Developer Dashboard or query directly
