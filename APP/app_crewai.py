import streamlit as st
import sqlite3, json, uuid, random, re, os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# REAL CREWAI IMPORT -- 2 separate agents are defined formally with CrewAI
# ============================================================================
CREWAI_OK = False
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_OK = True
except Exception:
    CREWAI_OK = False
    # CrewAI not installed - app will use direct agent functions instead

# ============================================================================
# DATABASE -- auto-migration, NEVER drops data
# ============================================================================
DB_PATH = "triage_hil.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY, ticket_number TEXT, patient_symptoms TEXT,
        national_id TEXT,
        enriched_symptoms TEXT,
        status TEXT DEFAULT 'processing',
        llm_urgency TEXT, llm_reasoning TEXT, llm_reasoning_original TEXT,
        llm_recommendation TEXT, llm_next_steps TEXT, llm_sources TEXT,
        llm_evidence TEXT, llm_patient_explanation TEXT, llm_confidence TEXT,
        rag_mode TEXT,
        prompt_chain_complete BOOLEAN, prompt_chain_questions TEXT, prompt_chain_answers TEXT,
        prompt_chain_rounds INTEGER, gate_decision TEXT,
        agent_action_trace TEXT, agent_confidence TEXT, agent_plan TEXT,
        evaluator_enhanced BOOLEAN,
        nurse_tier TEXT, nurse_action TEXT, nurse_notes TEXT,
        nurse_override_reason TEXT, nurse_timestamp TEXT, nurse_name TEXT,
        final_tier TEXT, booking_status TEXT, booking_details TEXT,
        booking_agent_decision TEXT,
        created_at TEXT, updated_at TEXT)""")
    cur = conn.execute("PRAGMA table_info(cases)")
    existing = {row[1] for row in cur.fetchall()}
    new_cols = [('enriched_symptoms','TEXT'),('llm_reasoning_original','TEXT'),
        ('national_id','TEXT'),
        ('prompt_chain_complete','BOOLEAN'),('prompt_chain_questions','TEXT'),
        ('prompt_chain_answers','TEXT'),('prompt_chain_rounds','INTEGER'),
        ('gate_decision','TEXT'),('agent_action_trace','TEXT'),
        ('agent_confidence','TEXT'),('agent_plan','TEXT'),
        ('evaluator_enhanced','BOOLEAN'),('booking_agent_decision','TEXT'),
        ('router_complete','BOOLEAN'),('router_questions','TEXT'),('router_answers','TEXT'),
        ('fhir_bundle','TEXT'),('crew_mode','TEXT'),('retrieval_debug','TEXT'),('evaluator_feedback','TEXT')]
    for col, typ in new_cols:
        if col not in existing:
            try:
                conn.execute(f'ALTER TABLE cases ADD COLUMN {col} {typ}')
            except Exception:
                pass
    conn.commit(); conn.close()

def db_insert(case):
    conn = sqlite3.connect(DB_PATH)
    cols = ', '.join(case.keys()); phs = ', '.join(['?']*len(case))
    conn.execute(f'INSERT OR REPLACE INTO cases ({cols}) VALUES ({phs})', list(case.values()))
    conn.commit(); conn.close()

def db_update(cid, upd):
    conn = sqlite3.connect(DB_PATH)
    upd['updated_at'] = datetime.utcnow().isoformat()
    s = ', '.join(f'{k}=?' for k in upd)
    conn.execute(f'UPDATE cases SET {s} WHERE case_id=?', list(upd.values())+[cid])
    conn.commit(); conn.close()

def db_get_all(status=None):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute('SELECT * FROM cases WHERE status=? ORDER BY created_at DESC', (status,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM cases ORDER BY created_at DESC').fetchall()
    conn.close(); return [dict(r) for r in rows]

def db_get_one(cid):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM cases WHERE case_id=?', (cid,)).fetchone()
    conn.close(); return dict(row) if row else None

def db_get_by_ticket(t):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM cases WHERE ticket_number=? ORDER BY created_at DESC', (t,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def db_stats():
    conn = sqlite3.connect(DB_PATH)
    t = conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
    p = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
    r = conn.execute("SELECT COUNT(*) FROM cases WHERE status='reviewed'").fetchone()[0]
    o = conn.execute("SELECT COUNT(*) FROM cases WHERE nurse_action LIKE 'override%'").fetchone()[0]
    conn.close(); return {'total':t,'pending':p,'reviewed':r,'overrides':o}

def get_next_ticket():
    """Generate next sequential ticket number (never repeats)"""
    conn = sqlite3.connect(DB_PATH)
    # Get the highest ticket number
    result = conn.execute("SELECT ticket_number FROM cases ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()
    
    if not result:
        return "T001"  # First ticket
    
    last_ticket = result[0]
    try:
        # Extract number from ticket (e.g., "T001" -> 1)
        num = int(last_ticket.replace('T', ''))
        next_num = num + 1
        return f"T{next_num:03d}"  # Format as T001, T002, etc.
    except:
        # If parsing fails, use total count + 1
        stats = db_stats()
        return f"T{stats['total'] + 1:03d}"


init_db()

# ============================================================================
# LOGIN
# ============================================================================
NURSES = {
    'haya':  {'password': '123', 'name': 'Haya'},
    'malek': {'password': '123', 'name': 'Malek'},
    'yomna': {'password': '123', 'name': 'Yomna'},
    'admin': {'password': 'admin2026', 'name': 'Administrator'},
}

def check_nurse_login():
    return st.session_state.get('nurse_name', None)

def nurse_login_form():
    st.title("Nurse Login")
    st.markdown("Please enter your credentials to access the triage review dashboard.")
    with st.form("nurse_login"):
        username = st.text_input("Username", placeholder="e.g. nurse_name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    if submitted:
        if username in NURSES and NURSES[username]['password'] == password:
            st.session_state.nurse_name = NURSES[username]['name']
            st.session_state.nurse_username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ============================================================================
# GROQ / LLAMA
# ============================================================================
GROQ_OK = False
try:
    from groq import Groq
    import os
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_OK = True
except Exception:
    pass

# def groq_call(prompt, system="", temperature=0.0, max_tokens=900):
#     from google import genai
#     from google.genai import types
#     client = genai.Client()
#     resp = client.models.generate_content(
#         model='gemini-2.5-flash',
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             system_instruction=system if system else None,
#             temperature=temperature,
#             max_output_tokens=max_tokens,
#             thinking_config=types.ThinkingConfig(thinking_budget=0)
#         )
#     )
#     return resp.text or ''

# def groq_call(prompt, system="", temperature=0.0, max_tokens=900):
#     from google import genai
#     from google.genai import types
    
#     import os
#     api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
#     if not api_key:
#         try:
#             api_key = st.secrets['GOOGLE_API_KEY']
#         except Exception:
#             pass
    
#     client = genai.Client(api_key=api_key)
#     resp = client.models.generate_content(
#         model='gemini-2.5-flash',
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             system_instruction=system if system else None,
#             temperature=temperature,
#             max_output_tokens=max_tokens,
#         )
#     )
#     return resp.text or ''    

def groq_call(prompt, system="", temperature=0.0, max_tokens=900):
    """Call Groq API with Llama 3.3 70B"""
    if not GROQ_OK:
        raise RuntimeError("Groq not available. Set GROQ_API_KEY environment variable.")
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)}")


#---------------------------------------------------------------------------
#----------------------------------------------------------------------

def safe_json_parse(raw, repair_label="JSON"):
    """
    Robustly parse JSON produced by the LLM.
    This does NOT hardcode questions or answers.
    If JSON is malformed, Groq is asked to repair only the JSON syntax.
    If repair fails, an explicit error is raised.
    """
    if raw is None:
        raise RuntimeError(f"{repair_label} returned empty response.")

    clean = re.sub(r'```json\s*|```\s*', '', str(raw)).strip()

    try:
        return json.loads(clean)
    except Exception:
        pass

    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        extracted = clean[start:end]
        try:
            return json.loads(extracted)
        except Exception:
            pass

    repair_prompt = f"""Repair the following malformed JSON.
Return ONLY valid JSON.
Do not add explanations.
Do not change the meaning.
Do not invent new questions or fields.

Malformed JSON:
{raw}
"""
    repaired = groq_call(repair_prompt, max_tokens=400)
    repaired_clean = re.sub(r'```json\s*|```\s*', '', repaired).strip()

    start = repaired_clean.find("{")
    end = repaired_clean.rfind("}") + 1
    if start != -1 and end > start:
        repaired_clean = repaired_clean[start:end]

    try:
        return json.loads(repaired_clean)
    except Exception as e:
        raise RuntimeError(f"{repair_label} returned unrecoverable invalid JSON: {raw}") from e



# ============================================================================
# TOOL 1: ASK_FOLLOWUP -- prompt chaining for vague inputs
# NOTE: Emergency bypass was removed intentionally.
# Extreme ambulance-level cases are outside this walk-in triage system.
# ============================================================================
MAX_CHAIN_ROUNDS = 3

def tool_ask_followup(symptoms, previous_qa=None):
    """Tool: Generate follow-up questions for vague patient inputs."""
    if not GROQ_OK:
        raise RuntimeError("Groq is not available. Check google-genai installation and GOOGLE_API_KEY.")

    chain_context = ""
    if previous_qa and len(previous_qa) > 0:
        chain_context = "\n\nPrevious follow-up answers:\n"
        chain_context += "\n".join(f"Q: {q}\nA: {a}" for q, a in previous_qa)

    prompt = f"""You are an ER intake nurse for WALK-IN patients only. Decide if the patient gave enough information for triage.

Patient said: "{symptoms}"{chain_context}

IMPORTANT:
- This is NOT a long questionnaire.
- You ONLY ask questions if triage information is missing.
- DO NOT assume ambulance-level emergencies.
- Do NOT use generic questions unless they are truly needed.
- If the patient already mentioned duration (e.g., "for 1 year", "for 3 days", "since yesterday"), DO NOT ask about duration again.

READY examples (enough info for triage):
- "fever 40 today"
- "cough for 3 days"
- "headache with vomiting since yesterday"
- "burning urination for 2 days"
- "acne for 1 year and I don't like how I look"
- "knee pain for 2 weeks"
- "rash on my arm for 5 days"
- "back pain for 1 month"

NOT READY examples (too vague, need details):
- "I feel sick"
- "I have a cold"
- "I have pain"
- "I have fever"
- "I have cough"
- "I have acne"
- "my knee hurts"

DECISION RULES:
- A symptom name alone is NOT enough.
- Words like cold, flu, fever, cough, pain, headache, acne WITHOUT details = NOT READY.
- Mark READY only if there is at least one symptom AND at least one useful triage detail:
  duration, severity, temperature, location, progression, associated symptom, or risk context.
- If the patient already provided duration, DO NOT ask for it again.
- If NOT READY, ask 1–2 short symptom-specific questions about MISSING information only.
- Do not repeat previous questions.

QUESTION FORMATTING RULES:
- Each question MUST be grammatically complete.
- Use full sentences (e.g., "How long have you had this symptom?" not "How long?")
- Include the symptom in the question for clarity.
- Examples of good questions:
  * "When did the headache start?"
  * "How severe is the pain on a scale of 1-10?"
  * "Have you had any fever with the cough?"
  * "Is the rash spreading or getting worse?"

Respond ONLY with valid compact JSON.
Do not include markdown or explanations.
Keep each question clear and under 15 words.
Ensure all strings and brackets are closed.

{{"ready": true, "questions": []}}
or
{{"ready": false, "questions": ["question 1", "question 2"]}}"""

    raw = groq_call(prompt, max_tokens=600)
    r = safe_json_parse(raw, repair_label="Groq follow-up")

    return {
        "ready": r.get("ready", True),
        "questions": r.get("questions", [])
    }

# ============================================================================
# TOOL 2: SEARCH_KB -- Hybrid retrieval from medical knowledge base
# BM25 = keyword retrieval
# Dense = sentence-transformer embeddings + cosine similarity
# Hybrid score = dense_weight*dense + bm25_weight*BM25
# ============================================================================
def bm25_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def minmax_normalize(scores):
    scores = np.array(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min())
    return np.zeros_like(scores)

@st.cache_resource
def load_knowledge_base():
    from rank_bm25 import BM25Okapi

    kb_path = Path(__file__).parent / "chunked_docs_phase2.json"
    if not kb_path.exists():
        return None, None, None, None, None, None

    with open(kb_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    word_re = re.compile(r'\w+')
    texts = [c['page_content'] for c in chunks]

    tokenized = [bm25_preprocess(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    embedding_model = None
    doc_embeddings = None
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        doc_embeddings = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    except Exception as e:
        st.warning(f"Dense retrieval unavailable. Using BM25 only. Details: {e}")

    return chunks, bm25, word_re, embedding_model, doc_embeddings, texts

KB_CHUNKS, BM25_INDEX, WORD_RE, EMBEDDING_MODEL, DOC_EMBEDDINGS, DOC_TEXTS = load_knowledge_base()
RAG_AVAILABLE = KB_CHUNKS is not None
DENSE_AVAILABLE = EMBEDDING_MODEL is not None and DOC_EMBEDDINGS is not None

def tool_search_kb(query, k=5, dense_weight=0.7, bm25_weight=0.3, min_hybrid_score=0.15):
    """
    Tool: Search medical knowledge base using hybrid retrieval.
    Returns retrieved chunks, evidence display, IEEE source lines, and debug scores.
    """
    if not RAG_AVAILABLE:
        return [], [], '', []

    # BM25 sparse scores
    tokens = bm25_preprocess(query)
    bm25_scores = np.array(BM25_INDEX.get_scores(tokens), dtype=np.float32)
    bm25_norm = minmax_normalize(bm25_scores)

    # Dense cosine scores
    if DENSE_AVAILABLE:
        query_emb = EMBEDDING_MODEL.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        dense_raw = np.array(np.dot(DOC_EMBEDDINGS, query_emb), dtype=np.float32)
        dense_norm = minmax_normalize(dense_raw)
        retrieval_mode = "hybrid_dense_bm25"
    else:
        dense_raw = np.zeros_like(bm25_scores)
        dense_norm = np.zeros_like(bm25_scores)
        retrieval_mode = "bm25_only_dense_unavailable"

    if DENSE_AVAILABLE:
        hybrid_scores = dense_weight * dense_norm + bm25_weight * bm25_norm
    else:
        hybrid_scores = bm25_norm

    ranked_indices = np.argsort(-hybrid_scores)

    evidence_blocks, source_lines, retrieved, debug_rows = [], [], [], []
    rank = 1

    for idx in ranked_indices:
        hybrid_score = float(hybrid_scores[idx])
        if hybrid_score < min_hybrid_score:
            continue

        chunk = KB_CHUNKS[idx]
        meta = chunk.get('metadata', {})
        topic = meta.get('topic','Unknown')
        source = meta.get('source','Unknown')
        url = meta.get('url','')
        section = meta.get('section_title','General')
        date = meta.get('document_date','n.d.')
        content = chunk['page_content']

        evidence_blocks.append(
            f"[{rank}] **{topic}** -- {section}\n"
            f"> {content[:300]}...\n\n"
            f"_Hybrid: {hybrid_score:.3f} | Dense: {float(dense_norm[idx]):.3f} | BM25: {float(bm25_norm[idx]):.3f}_"
        )

        source_lines.append(
            f'[{rank}] {source}, "{topic}," section: "{section}," {date}. [Online]. Available: {url}'
        )

        retrieved.append({
            'rank': rank,
            'topic': topic,
            'section': section,
            'score': hybrid_score,
            'hybrid_score': hybrid_score,
            'dense_score': float(dense_norm[idx]),
            'bm25_score': float(bm25_norm[idx]),
            'raw_dense_score': float(dense_raw[idx]),
            'raw_bm25_score': float(bm25_scores[idx]),
            'retrieval_mode': retrieval_mode,
            'content': content[:700]
        })

        debug_rows.append({
            'rank': rank,
            'topic': topic,
            'section': section,
            'source': source,
            'hybrid_score': hybrid_score,
            'dense_score': float(dense_norm[idx]),
            'bm25_score': float(bm25_norm[idx]),
            'retrieval_mode': retrieval_mode
        })

        rank += 1
        if len(retrieved) >= k:
            break

    return retrieved, evidence_blocks, '\n'.join(source_lines), debug_rows

# ============================================================================
# TOOL 3: ASSESS_URGENCY -- RAG-grounded LLM triage
# ============================================================================
TRIAGE_GROUNDED = """You are an Emergency Department triage assistant supporting an ER nurse.
## Scope
- This system is for walk-in emergency department triage cases.
- Do not assume ambulance-level cases are handled by this system.
## Rules
- Use the retrieved medical evidence to support your assessment.
- Reference evidence using [1], [2], etc.
- Do NOT provide a diagnosis.
## Urgency Categories (choose EXACTLY ONE)
- Urgent: Immediate or same-day evaluation for walk-in patients.
- Routine: Non-urgent evaluation within days to weeks.
- Self-care: Safe home management. No red or yellow flags.
## Output Format (STRICT)
Urgency: <Urgent / Routine / Self-care>
Confidence: <High / Medium / Low>
Reasoning:
<2-4 sentences referencing evidence with [1], [2], etc.>
Recommendation:
<Specific clinical recommendation>
Next steps:
- <Action 1>
- <Action 2>
- <Action 3>
Patient explanation:
<2-3 simple sentences for the patient. No jargon.>"""

TRIAGE_FALLBACK = """You are an Emergency Department triage assistant for walk-in patients.
The knowledge base did not have enough relevant information.
Provide a cautious best-effort triage based only on symptoms.
## Output Format (STRICT)
Urgency: <Urgent / Routine / Self-care>
Confidence: <Low>
Reasoning:
<Include: "Limited evidence was available in the knowledge base for this query.">
Recommendation: <Clinical recommendation>
Next steps:
- <Action 1>
- <Action 2>
Patient explanation: <2-3 simple sentences for the patient.>"""

def tool_assess_urgency(symptoms):
    """Tool: Classify urgency using RAG evidence + Groq."""
    retrieved, evidence_blocks, sources_text, retrieval_debug = tool_search_kb(symptoms, k=5)
    has_evidence = len(retrieved) > 0 and retrieved[0].get('hybrid_score', retrieved[0].get('score', 0)) > 0.15
    if has_evidence:
        context = '\n\n'.join(f"[{r['rank']}] ({r['topic']} - {r['section']})\n{r['content']}" for r in retrieved)
        user_msg = f'Patient symptoms:\n"""{symptoms}"""\n\nRetrieved medical evidence:\n{context}\n\nClassify urgency using the evidence above.'
        system = TRIAGE_GROUNDED; mode = 'grounded_rag'
    else:
        user_msg = f'Patient symptoms:\n"""{symptoms}"""'
        system = TRIAGE_FALLBACK; mode = 'fallback_judgment'
    try:
        raw = groq_call(user_msg, system=system, max_tokens=1200)
        result = parse_triage(raw)
    except Exception as e:
        raise RuntimeError(f"Groq triage failed: {e}")
    result['evidence'] = '\n\n'.join(evidence_blocks) if evidence_blocks else 'No evidence retrieved.'
    result['sources'] = sources_text if sources_text else 'None'
    result['rag_mode'] = mode
    result['retrieval_debug'] = retrieval_debug
    return result

def run_triage_demo(symptoms):
    s = symptoms.lower()
    if any(k in s for k in ['chest pain','stroke','breathing','unconscious','bleeding','suicidal','seizure']):
        return {'urgency':'Urgent','confidence':'High','reasoning':'Possible red-flag symptoms in a walk-in triage context.','recommendation':'Same-day clinical evaluation.','next_steps':'- Nurse review\n- Check vital signs','patient_explanation':'Your symptoms may need prompt medical attention.'}
    if any(k in s for k in ['cough','burning','back pain','rash','headache','depressed','knee']):
        return {'urgency':'Routine','confidence':'Medium','reasoning':'Symptoms suggest a non-emergency walk-in case requiring clinical follow-up.','recommendation':'Schedule appointment.','next_steps':'- Monitor symptoms','patient_explanation':'A clinician should review your symptoms.'}
    return {'urgency':'Self-care','confidence':'Medium','reasoning':'Symptoms appear mild with no clear urgent features in the provided text.','recommendation':'Home management with monitoring.','next_steps':'- Rest and monitor','patient_explanation':'You may be able to manage this at home, but seek care if symptoms worsen.'}

def parse_triage(text):
    result = {'urgency':'Unknown','confidence':'Medium','reasoning':'','recommendation':'','next_steps':'','patient_explanation':'','sources':'','evidence':''}
    if not text:
        return result
    m = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)', text, re.IGNORECASE)
    if m:
        u = m.group(1).strip()
        result['urgency'] = 'Self-care' if 'self' in u.lower() else u.capitalize()
    m = re.search(r'Confidence:\s*(High|Medium|Low)', text, re.IGNORECASE)
    if m:
        result['confidence'] = m.group(1).capitalize()
    for key, pat in [('reasoning',r'Reasoning:\s*\n?(.*?)(?=\n(?:Recommendation|Next|Patient|Sources):|\Z)'),
                     ('recommendation',r'Recommendation:\s*\n?(.*?)(?=\n(?:Next|Patient|Sources):|\Z)'),
                     ('next_steps',r'Next [Ss]teps:\s*\n?(.*?)(?=\n(?:Patient|Sources):|\Z)'),
                     ('patient_explanation',r'Patient [Ee]xplanation:\s*\n?(.*?)(?=\n(?:Sources):|\Z)')]:
        m = re.search(pat, text, re.DOTALL|re.IGNORECASE)
        if m:
            result[key] = m.group(1).strip()
    return result

# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================
def agent_confidence_check(symptoms, urgency, evidence):
    """Agent reports calibrated confidence + guidance for nurse."""
    if not GROQ_OK:
        raise RuntimeError("Groq is not available for confidence calibration. Check GOOGLE_API_KEY.")
    prompt = f'''You just classified a walk-in ER triage patient as {urgency}.

Patient: "{symptoms}"
Evidence: {(evidence or "None")[:300]}

Answer THREE questions honestly:
1. CONFIDENCE (0-100): How confident are you in this classification?
2. WOULD CHANGE IF: What ONE piece of information would make you change the urgency?
3. NURSE SHOULD CHECK: What should the nurse specifically verify?

Respond ONLY in JSON:
{{"confidence_pct": 85, "would_change_if": "patient reports chest tightness", "nurse_watch_for": "check vitals and symptom severity"}}'''
    try:
        raw = groq_call(prompt, max_tokens=500)
        return safe_json_parse(raw, repair_label="Groq confidence calibration")
    except Exception as e:
        raise RuntimeError(f"Groq confidence calibration failed: {e}")

# ============================================================================
# TWO REAL CREWAI AGENTS + CUSTOM EXECUTION LOGIC
# Agentic AI 1: Prompt-Chaining + Gate + FHIR-like formatting
# Agentic AI 2: RAG Evaluator-Optimizer
# ============================================================================
def is_vague(text):
    """
    Deprecated helper kept for compatibility.
    We do not use symptom keyword lists to decide sufficiency.
    Agentic AI 1 uses the Groq prompt-chaining gate.
    """
    return True

def build_triage_crew():
    """Formal CrewAI two-agent structure."""
    if not CREWAI_OK:
        return None

    prompt_chaining_agent = Agent(
        role="Prompt-Chaining and Gate Agent",
        goal="Check if walk-in patient input is sufficient and convert it into FHIR-like structured data.",
        backstory=(
            "You prepare patient symptom input before it reaches the RAG triage module. "
            "You ask follow-up questions only when the input is too vague, and you produce a structured handoff."
        ),
        verbose=False,
        allow_delegation=False,
    )

    rag_evaluator_agent = Agent(
        role="RAG Evaluator-Optimizer Agent",
        goal="Use structured patient input to retrieve evidence, classify urgency, evaluate the output, and optimize reasoning.",
        backstory=(
            "You support nurse review by producing evidence-grounded triage recommendations, evaluator feedback, "
            "and confidence guidance."
        ),
        verbose=False,
        allow_delegation=False,
    )

    task1 = Task(
        description=(
            "Analyze the patient input. If information is insufficient, request follow-up questions. "
            "If sufficient, generate a FHIR-like structured handoff."
        ),
        expected_output="A JSON handoff containing status, enriched symptoms, gate decision, and FHIR-like data.",
        agent=prompt_chaining_agent,
    )

    task2 = Task(
        description=(
            "Use the structured handoff from Agent 1 to perform RAG-based triage, evaluation, optimization, "
            "and confidence calibration."
        ),
        expected_output="A final triage output containing urgency, reasoning, recommendation, evidence, confidence, and nurse guidance.",
        agent=rag_evaluator_agent,
        context=[task1],
    )

    return Crew(
        agents=[prompt_chaining_agent, rag_evaluator_agent],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=False,
    )

@st.cache_resource
def get_triage_crew():
    return build_triage_crew()

def make_fhir_like(symptoms, previous_qa=None):
    """
    FHIR-like structured handoff.
    This is not full certified HL7 FHIR. It is a FHIR-inspired structure for this prototype.
    """
    qa_items = []
    if previous_qa:
        qa_items = [{"text": q, "answer": [{"valueString": a}]} for q, a in previous_qa]

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [{"system": "local-triage-ticket", "value": "provided_in_app"}]
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "status": "in-progress",
                    "class": {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                        "code": "EMER",
                        "display": "emergency"
                    },
                    "serviceType": {"text": "Walk-in emergency triage"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "status": "preliminary",
                    "code": {"text": "Patient-reported symptoms"},
                    "valueString": symptoms
                }
            },
            {
                "resource": {
                    "resourceType": "QuestionnaireResponse",
                    "status": "completed",
                    "item": qa_items
                }
            }
        ]
    }

def prompt_chaining_gate_agent(symptoms, previous_qa=None):
    """
    Agentic AI 1:
    - Prompt chaining
    - Gate: enough information?
    - Gate: FHIR-like handoff ready?
    - No emergency bypass.
    """
    action_trace = []
    enriched = symptoms
    if previous_qa:
        qa_text = '\n'.join(f"Q: {q}\nA: {a}" for q, a in previous_qa)
        enriched = f"{symptoms}\n\nAdditional details:\n{qa_text}"

    round_num = len(previous_qa) if previous_qa else 0

    action_trace.append({
        "step": 1,
        "agent": "Agentic AI 1",
        "tool": "PROMPT_CHAIN_GATE",
        "reason": "Check whether input is sufficient for RAG triage",
        "result": "Running sufficiency check"
    })

    input_is_vague = is_vague(enriched)

    # Agentic AI 1 uses the Groq prompt-chaining gate.
    # If Groq fails, the error is shown; no hardcoded questions are generated.
    if round_num < MAX_CHAIN_ROUNDS:
        try:
            chain_result = tool_ask_followup(enriched, previous_qa)
        except Exception as e:
            action_trace.append({
                "step": 2,
                "agent": "Agentic AI 1",
                "tool": "ASK_FOLLOWUP_ERROR",
                "reason": "Groq failed during prompt-chaining gate",
                "result": str(e)
            })
            return {
                "status": "error",
                "error": str(e),
                "questions": [],
                "enriched_symptoms": enriched,
                "fhir_bundle": None,
                "gate_decision": "agent1_error",
                "agent_plan": "agent1_failed",
                "action_trace": action_trace,
            }

        if not chain_result.get("ready", True) and chain_result.get("questions"):
            action_trace.append({
                "step": 2,
                "agent": "Agentic AI 1",
                "tool": "ASK_FOLLOWUP",
                "reason": "Prompt-chaining gate decided that triage details are missing",
                "result": f"Asking {len(chain_result['questions'])} question(s)"
            })
            return {
                "status": "need_more_info",
                "questions": chain_result["questions"],
                "enriched_symptoms": enriched,
                "fhir_bundle": None,
                "gate_decision": "need_more_info",
                "agent_plan": "agent1_prompt_chain_gather_info",
                "action_trace": action_trace,
            }

    fhir_bundle = make_fhir_like(enriched, previous_qa)
    gate_decision = "ready_fhir_valid" if round_num < MAX_CHAIN_ROUNDS else "max_rounds_reached_fhir_valid"

    action_trace.append({
        "step": len(action_trace) + 1,
        "agent": "Agentic AI 1",
        "tool": "FHIR_FORMATTER",
        "reason": "Convert sufficient input into FHIR-like structured handoff",
        "result": "FHIR-like handoff created"
    })

    action_trace.append({
        "step": len(action_trace) + 1,
        "agent": "Agentic AI 1",
        "tool": "GATE_DECISION",
        "reason": "Input is sufficient and structured",
        "result": gate_decision
    })

    return {
        "status": "ready_for_agent2",
        "questions": [],
        "enriched_symptoms": enriched,
        "fhir_bundle": fhir_bundle,
        "gate_decision": gate_decision,
        "agent_plan": "agent1_ready_for_agent2",
        "action_trace": action_trace,
    }

def rag_evaluator_optimizer_agent(agent1_output):
    """
    Agentic AI 2:
    - RAG retrieval
    - LLM triage
    - Evaluator / optimizer
    - Confidence calibration
    """
    enriched = agent1_output["enriched_symptoms"]
    action_trace = agent1_output.get("action_trace", [])

    action_trace.append({
        "step": len(action_trace) + 1,
        "agent": "Agentic AI 2",
        "tool": "RAG_TRIAGE",
        "reason": "Use Agent 1 structured handoff for evidence-grounded triage",
        "result": "Starting RAG-based triage"
    })

    try:
        tri = tool_assess_urgency(enriched)
    except Exception as e:
        action_trace.append({
            "step": len(action_trace) + 1,
            "agent": "Agentic AI 2",
            "tool": "RAG_TRIAGE_ERROR",
            "reason": "Groq failed during RAG triage",
            "result": str(e)
        })
        return {
            "status": "error",
            "error": str(e),
            "triage": {},
            "action_trace": action_trace,
            "agent_plan": "agent2_failed",
            "original_reasoning": "",
            "evaluator_enhanced": False,
            "gate_decision": agent1_output.get("gate_decision", ""),
            "agent_confidence": {},
            "fhir_bundle": agent1_output.get("fhir_bundle", {}),
        }

    original_reasoning = tri.get("reasoning", "")

    action_trace.append({
        "step": len(action_trace) + 1,
        "agent": "Agentic AI 2",
        "tool": "ASSESS_URGENCY",
        "reason": "RAG-grounded urgency classification",
        "result": f"Urgency: {tri.get('urgency','Unknown')} | Mode: {tri.get('rag_mode','')}"
    })

    evaluator_ran = False
    evaluator_feedback = {}

    if GROQ_OK:
        reflect_prompt = f'''You are a senior emergency physician reviewing a RAG-based triage output for a WALK-IN emergency patient.

Patient structured input:
{json.dumps(agent1_output.get("fhir_bundle", {}), indent=2)}

Patient text:
"{enriched}"

Initial triage urgency:
{tri.get("urgency", "Unknown")}

Initial reasoning:
"{original_reasoning}"

Retrieved evidence:
{tri.get("evidence", "No evidence retrieved.")}

Evaluate the output:
1. Is the urgency level reasonable for a walk-in triage case?
2. Is the reasoning clinically sound?
3. Are the retrieved passages relevant to the patient's symptoms?
4. Is the reasoning supported by the retrieved evidence?
5. Can the reasoning be improved?

Rules:
- If the retrieved evidence is unrelated to the patient symptoms, mark confident as false.
- If the reasoning uses irrelevant evidence, mark confident as false.
- If the urgency is reasonable but the explanation can improve, keep confident true and enhance the reasoning.

If confident:
{{"confident": true, "evidence_relevance": "relevant", "evidence_relevance_reason": "why the retrieved evidence matches", "enhanced_reasoning": "improved reasoning"}}

If not confident:
{{"confident": false, "evidence_relevance": "irrelevant or partially relevant", "evidence_relevance_reason": "why the retrieved evidence is weak or unrelated", "concern": "what is wrong", "suggested_urgency": "Urgent or Routine or Self-care"}}

Respond ONLY in JSON.'''
        try:
            raw = groq_call(reflect_prompt, max_tokens=700)
            reflection = safe_json_parse(raw, repair_label="Groq evaluator")
        except Exception as e:
            action_trace.append({
                "step": len(action_trace) + 1,
                "agent": "Agentic AI 2",
                "tool": "EVALUATOR_ERROR",
                "reason": "Groq evaluator returned invalid JSON",
                "result": str(e)
            })
            return {
                "status": "error",
                "error": str(e),
                "triage": tri,
                "action_trace": action_trace,
                "agent_plan": "agent2_evaluator_failed",
                "original_reasoning": original_reasoning,
                "evaluator_enhanced": False,
                "gate_decision": agent1_output.get("gate_decision", ""),
                "agent_confidence": {},
                "fhir_bundle": agent1_output.get("fhir_bundle", {}),
                "evaluator_feedback": evaluator_feedback if 'evaluator_feedback' in locals() else {},
            }

        evaluator_feedback = reflection

        if reflection.get("confident"):
            if reflection.get("enhanced_reasoning") and len(reflection["enhanced_reasoning"]) > 50:
                tri["reasoning"] = reflection["enhanced_reasoning"]
                evaluator_ran = True
            action_trace.append({
                "step": len(action_trace) + 1,
                "agent": "Agentic AI 2",
                "tool": "EVALUATOR",
                "reason": "Evaluator approved the triage output",
                "result": "Reasoning enhanced" if evaluator_ran else "No change needed"
            })
        else:
            concern = reflection.get("concern", "unclear assessment")
            suggested = reflection.get("suggested_urgency", "")
            action_trace.append({
                "step": len(action_trace) + 1,
                "agent": "Agentic AI 2",
                "tool": "EVALUATOR",
                "reason": f"Evaluator was not confident: {concern}",
                "result": f"Triggering optimizer re-triage. Suggested urgency: {suggested}"
            })
            correction_input = (
                f"{enriched}\n\n"
                f"[EVALUATOR FEEDBACK]: Previous RAG assessment may be wrong. "
                f"Concern: {concern}. Suggested urgency: {suggested}."
            )
            tri = tool_assess_urgency(correction_input)
            evaluator_ran = True
            action_trace.append({
                "step": len(action_trace) + 1,
                "agent": "Agentic AI 2",
                "tool": "OPTIMIZER_RE_TRIAGE",
                "reason": "Optimizer re-ran triage using evaluator feedback",
                "result": f"New urgency: {tri.get('urgency','Unknown')}"
            })

    try:
        conf = agent_confidence_check(enriched, tri.get("urgency", ""), tri.get("evidence", ""))
    except Exception as e:
        action_trace.append({
            "step": len(action_trace) + 1,
            "agent": "Agentic AI 2",
            "tool": "CONFIDENCE_CHECK_ERROR",
            "reason": "Groq failed during confidence calibration",
            "result": str(e)
        })
        return {
            "status": "error",
            "error": str(e),
            "triage": tri,
            "action_trace": action_trace,
            "agent_plan": "agent2_confidence_failed",
            "original_reasoning": original_reasoning,
            "evaluator_enhanced": evaluator_ran,
            "gate_decision": agent1_output.get("gate_decision", ""),
            "agent_confidence": {},
            "fhir_bundle": agent1_output.get("fhir_bundle", {}),
        }

    action_trace.append({
        "step": len(action_trace) + 1,
        "agent": "Agentic AI 2",
        "tool": "CONFIDENCE_CHECK",
        "reason": "Generate calibrated confidence and nurse guidance",
        "result": f"{conf.get('confidence_pct',50)}% | Watch: {conf.get('nurse_watch_for','N/A')}"
    })

    return {
        "status": "complete",
        "triage": tri,
        "action_trace": action_trace,
        "agent_plan": "crewai_two_agent_pipeline",
        "original_reasoning": original_reasoning,
        "evaluator_enhanced": evaluator_ran,
        "gate_decision": agent1_output.get("gate_decision", ""),
        "agent_confidence": conf,
        "fhir_bundle": agent1_output.get("fhir_bundle", {}),
        "evaluator_feedback": evaluator_feedback,
    }

def run_triage_agent(symptoms, previous_qa=None):
    """
    Main Streamlit entry point.
    Real architecture:
    Agentic AI 1: Prompt-Chaining + Gate + FHIR-like formatting
    Agentic AI 2: RAG Evaluator-Optimizer
    CrewAI objects are initialized formally; Streamlit controls execution because follow-up is interactive.
    """
    crew = get_triage_crew()
    agent1_output = prompt_chaining_gate_agent(symptoms, previous_qa)
    if agent1_output["status"] == "need_more_info":
        agent1_output["crew_mode"] = "real_crewai_initialized" if crew is not None else "crewai_not_available"
        return agent1_output
    agent2_output = rag_evaluator_optimizer_agent(agent1_output)
    agent2_output["crew_mode"] = "real_crewai_initialized" if crew is not None else "crewai_not_available"
    return agent2_output

# ============================================================================
# CROSS-CASE LEARNING AGENT
# ============================================================================
def agent_cross_case_analysis():
    reviewed = db_get_all(status='reviewed')
    if len(reviewed) < 3:
        return {"status": "insufficient_data", "message": "Need at least 3 reviewed cases"}
    case_summaries = []
    for c in reviewed:
        case_summaries.append(
            f"- AI: {c.get('llm_urgency','?')}, Nurse: {c.get('final_tier','?')}, "
            f"Action: {c.get('nurse_action','?')}, "
            f"Override reason: {c.get('nurse_override_reason','none')}, "
            f"Symptoms: {(c.get('patient_symptoms',''))[:80]}")
    prompt = f'''You are a clinical AI researcher analyzing triage decision patterns.

{len(reviewed)} reviewed cases (AI vs nurse decisions):
{chr(10).join(case_summaries)}

Analyze:
1. AGREEMENT_RATE: percentage AI and nurse agreed
2. BIAS_PATTERN: when they disagree, is there a consistent direction?
3. FAILURE_MODES: what case types does the AI get wrong?
4. IMPROVEMENT: what ONE change would reduce disagreements?
5. NOVEL_FINDING: any unexpected or counterintuitive pattern?

Respond ONLY in JSON:
{{"agreement_rate": "X%", "bias_pattern": "description", "failure_modes": ["mode1","mode2"], "improvement": "suggestion", "novel_finding": "finding or null"}}'''
    try:
        raw = groq_call(prompt, max_tokens=800)
        return safe_json_parse(raw, repair_label="Cross-case analysis")
    except Exception as e:
        return {"status": "error", "message": f"Analysis failed: {e}"}

# ============================================================================
# HARD CODED BOOKING -- reads real schedule, always future dates
# ============================================================================
@st.cache_resource
def load_schedules():
    import pandas as pd
    path = Path(__file__).parent / "agentic_triage_schedules.xlsx"
    if not path.exists():
        return None, None
    return pd.read_excel(path, sheet_name='Emergency Schedule'), pd.read_excel(path, sheet_name='Routine Schedule')

EMERGENCY_DF, ROUTINE_DF = load_schedules()
SCHEDULE_AVAILABLE = EMERGENCY_DF is not None
if 'booked_slots' not in st.session_state:
    st.session_state.booked_slots = set()

def future_date(slot_date, slot_day, slot_time):
    today = datetime.now()
    try:
        orig = datetime.strptime(str(slot_date)[:10], '%Y-%m-%d')
        if orig.date() < today.date():
            days_diff = (today.date() - orig.date()).days
            weeks_ahead = (days_diff // 7) + 1
            new_date = orig + timedelta(weeks=weeks_ahead)
            return new_date.strftime('%B %d, %Y') + f" ({slot_day}) at {slot_time}"
        return orig.strftime('%B %d, %Y') + f" ({slot_day}) at {slot_time}"
    except Exception:
        return f"{slot_date} ({slot_day}) at {slot_time}"

def find_available_slot(schedule_df):
    if schedule_df is None:
        return None
    available = schedule_df[(schedule_df['Available']=='Available')&(~schedule_df['Slot ID'].isin(st.session_state.booked_slots))]
    return available.iloc[0] if not available.empty else None

def book_action(case_id, tier, symptoms=''):
    if tier == 'Urgent':
        if SCHEDULE_AVAILABLE:
            slot = find_available_slot(EMERGENCY_DF)
            if slot is not None:
                st.session_state.booked_slots.add(slot['Slot ID'])
                return {'status':'urgent_referral','type':'Immediate ER Admission','doctor':f"{slot['Doctor']} -- On Duty",'time':'IMMEDIATELY -- No appointment needed','dept':'Emergency','room':slot['Room'],'booking_id':slot['Slot ID'],'agent_decision':'emergency_referral','instructions':f"Proceed to {slot['Room']} immediately. {slot['Doctor']} is on duty."}
        return {'status':'urgent_referral','type':'Immediate ER Admission','doctor':'On-duty ER physician','time':'IMMEDIATELY','dept':'Emergency','room':f'ER-{random.randint(1,3)}','booking_id':f'UR-{uuid.uuid4().hex[:6]}','agent_decision':'emergency_fallback','instructions':'Proceed to Emergency Room immediately.'}
    elif tier == 'Routine':
        if SCHEDULE_AVAILABLE:
            slot = find_available_slot(ROUTINE_DF)
            if slot is not None:
                st.session_state.booked_slots.add(slot['Slot ID'])
                ft = future_date(slot['Date'], slot['Day'], slot['Time'])
                return {'status':'booked','type':'Scheduled Appointment','doctor':slot['Doctor'],'time':ft,'dept':slot['Department'],'room':slot['Room'],'booking_id':slot['Slot ID'],'agent_decision':'routine_appointment','instructions':f"Appointment with {slot['Doctor']} on {ft} in {slot['Room']}. Arrive 15 minutes early."}
        d = random.randint(2,10)
        return {'status':'booked','type':'Scheduled Appointment','doctor':'Dr. Hall','dept':'General','time':(datetime.now()+timedelta(days=d)).strftime('%B %d, %Y at %I:%M %p'),'room':f'G-{random.randint(101,104)}','booking_id':f'BK-{uuid.uuid4().hex[:6]}','agent_decision':'routine_fallback','instructions':'Arrive 15 minutes early.'}
    else:
        return {'status':'self_care_issued','type':'Self-Care Guidance','doctor':None,'time':None,'dept':None,'room':None,'booking_id':f'SC-{uuid.uuid4().hex[:6]}','agent_decision':'self_care','instructions':'Manage at home. Return if worsening.','guidance':'- Rest and hydrate\n- Monitor symptoms\n- OTC medication as needed\n- Return if fever > 38.5C or worsening'}

# ============================================================================
# STREAMLIT APP
# ============================================================================
st.set_page_config(page_title="Triage Decision Support System", layout="wide")
st.markdown("""<style>
    /* Increase base font size */
    html, body, [class*="css"] {
        font-size: 18px;
    }
    
    /* Larger text in main content */
    .stMarkdown, .stText {
        font-size: 18px;
    }
    
    /* Larger headers */
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.75rem !important; }
    h4 { font-size: 1.5rem !important; }
    
    /* Larger input text */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-size: 18px !important;
    }
    
    /* Larger button text */
    .stButton > button {
        font-size: 18px !important;
        padding: 12px 24px !important;
    }
    
    /* Larger selectbox/radio */
    .stSelectbox > div > div,
    .stRadio > label {
        font-size: 18px !important;
    }
    
    /* Metric cards styling */
    div[data-testid="stMetric"]{
        background:#f8f9fa;
        padding:12px;
        border-radius:8px;
        font-size: 18px;
    }
</style>""", unsafe_allow_html=True)

if 'fu_stage' not in st.session_state: st.session_state.fu_stage = 'initial'
if 'fu_qa' not in st.session_state: st.session_state.fu_qa = []
if 'fu_round' not in st.session_state: st.session_state.fu_round = 0
if 'fu_ticket' not in st.session_state: st.session_state.fu_ticket = ''
if 'fu_symptoms' not in st.session_state: st.session_state.fu_symptoms = ''
if 'fu_questions' not in st.session_state: st.session_state.fu_questions = []

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard","Nurse Dashboard","Developer Dashboard"])
stats = db_stats()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cases:** {stats['total']} total | {stats['pending']} pending")
st.sidebar.markdown("---")
with st.sidebar.expander("Admin"):
    if st.button("Reset Database", type="secondary"):
        conn = sqlite3.connect(DB_PATH); conn.execute("DELETE FROM cases"); conn.commit(); conn.close()
        st.session_state.booked_slots = set(); st.session_state.fu_stage = 'initial'
        st.session_state.fu_qa = []; st.session_state.fu_round = 0; st.rerun()

# ======================== PATIENT DASHBOARD ==================================
if page == "Patient Dashboard":
    st.title("🏥 Patient Triage Dashboard")
    tab1, tab2 = st.tabs(["Submit New Case","Check Results"])
    with tab1:
        if st.session_state.fu_stage == 'initial':
            # Auto-generate next ticket number
            next_ticket = get_next_ticket()
            st.info(f"📋 Your ticket number will be: **{next_ticket}**")
            
            with st.form("pf"):
                national_id = st.text_input("National ID / Passport Number", placeholder="e.g. 12345678901", max_chars=20)
                symptoms = st.text_area("Describe your symptoms", placeholder="Please describe what you are feeling, when it started, and how severe it is...", height=150)
                submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
            if submitted:
                if not national_id or not symptoms:
                    st.error("Please fill in both fields.")
                elif len(symptoms.strip()) < 10:
                    st.warning("Please describe your symptoms in more detail.")
                else:
                    ticket = next_ticket  # Use auto-generated ticket
                    with st.spinner("Thank you. We are reviewing your symptoms. Please wait a moment..."):
                        result = run_triage_agent(symptoms)
                    if result['status'] == 'need_more_info' and result.get('questions'):
                        st.session_state.fu_stage = 'followup'; st.session_state.fu_ticket = ticket
                        st.session_state.fu_national_id = national_id
                        st.session_state.fu_symptoms = symptoms; st.session_state.fu_qa = []
                        st.session_state.fu_round = 1; st.session_state.fu_questions = result['questions']
                        st.rerun()
                    elif result['status'] == 'error':
                        st.error("⚠️ AI service is temporarily unavailable or returned invalid structured output. No fallback answer was generated.")
                        st.caption(result.get('error', 'Unknown error'))
                    elif result['status'] == 'complete':
                        cid = f"CASE-{uuid.uuid4().hex[:8]}"; tri = result['triage']
                        conf = result.get('agent_confidence', {})
                        db_insert({'case_id':cid,'ticket_number':ticket,'national_id':national_id,'patient_symptoms':symptoms,'enriched_symptoms':result.get('fhir_bundle') and result.get('fhir_bundle',{}).get('entry',[{}, {}, {'resource':{'valueString':symptoms}}])[2].get('resource',{}).get('valueString',symptoms) or symptoms,'status':'pending','llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),'llm_reasoning_original':result.get('original_reasoning',''),'llm_recommendation':tri.get('recommendation',''),'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),'llm_patient_explanation':tri.get('patient_explanation',''),'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),'prompt_chain_complete':True,'prompt_chain_rounds':0,'gate_decision':result.get('gate_decision',''),'agent_action_trace':json.dumps(result.get('action_trace',[])),'agent_confidence':json.dumps(conf),'agent_plan':result.get('agent_plan',''),'evaluator_enhanced':result.get('evaluator_enhanced',False),'fhir_bundle':json.dumps(result.get('fhir_bundle',{})),'crew_mode':result.get('crew_mode',''),'retrieval_debug':json.dumps(tri.get('retrieval_debug',[])),'evaluator_feedback':json.dumps(result.get('evaluator_feedback',{})),'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                        st.success("✅ Submitted!"); st.info(f"**Case ID:** `{cid}` — Your case has been submitted. A nurse will review it shortly.")

        elif st.session_state.fu_stage == 'followup':
            round_num = st.session_state.fu_round
            st.markdown(f"### We need a few more details (Round {round_num}/{MAX_CHAIN_ROUNDS})")
            st.markdown(f"**Your initial description:** _{st.session_state.fu_symptoms}_")
            if st.session_state.fu_qa:
                with st.expander(f"Previous answers ({len(st.session_state.fu_qa)} answered)", expanded=False):
                    for q, a in st.session_state.fu_qa:
                        st.markdown(f"**Q:** {q}"); st.markdown(f"**A:** {a}"); st.markdown("")
            st.markdown("---")
            questions = st.session_state.fu_questions
            # Calculate cumulative question number (previous answered + current)
            prev_count = len(st.session_state.fu_qa)
            with st.form(f"followup_{round_num}"):
                answers = []
                for i, q in enumerate(questions):
                    q_num = prev_count + i + 1  # Cumulative question number
                    st.markdown(f"**Question {q_num}:** {q}")
                    ans = st.text_input("Your answer:", key=f"fqa_{round_num}_{i}", placeholder="Type your answer here...")
                    answers.append(ans); st.markdown("")
                fc1, fc2 = st.columns(2)
                with fc1:
                    back_btn = st.form_submit_button("Start Over", use_container_width=True)
                with fc2:
                    submit_btn = st.form_submit_button("Submit Answers", type="primary", use_container_width=True)
            if back_btn:
                st.session_state.fu_stage = 'initial'; st.session_state.fu_qa = []; st.session_state.fu_round = 0; st.rerun()
            if submit_btn:
                if not all(a.strip() for a in answers):
                    st.error("Please answer all questions.")
                else:
                    new_pairs = list(zip(questions, answers))
                    all_qa = st.session_state.fu_qa + new_pairs
                    qa_text = '\n'.join(f"Q: {q}\nA: {a}" for q, a in all_qa)
                    enriched = f"{st.session_state.fu_symptoms}\n\nAdditional details:\n{qa_text}"
                    with st.spinner("Thank you for the extra details. Please wait while we review your case..."):
                        result = run_triage_agent(enriched, previous_qa=all_qa)
                    if result['status'] == 'need_more_info' and result.get('questions') and round_num < MAX_CHAIN_ROUNDS:
                        st.session_state.fu_qa = all_qa; st.session_state.fu_round = round_num + 1
                        st.session_state.fu_questions = result['questions']; st.rerun()
                    elif result['status'] == 'error':
                        st.error("⚠️ AI service is temporarily unavailable or returned invalid structured output. No fallback answer was generated.")
                        st.caption(result.get('error', 'Unknown error'))
                    else:
                        if result['status'] != 'complete':
                            agent1_output = prompt_chaining_gate_agent(enriched, all_qa)
                            agent1_output['gate_decision'] = 'max_rounds_forced_fhir_valid'
                            result = rag_evaluator_optimizer_agent(agent1_output)
                            result['crew_mode'] = 'real_crewai_initialized' if get_triage_crew() is not None else 'crewai_not_available'
                        cid = f"CASE-{uuid.uuid4().hex[:8]}"; tri = result['triage']
                        conf = result.get('agent_confidence', {})
                        db_insert({'case_id':cid,'ticket_number':st.session_state.fu_ticket,'national_id':st.session_state.get('fu_national_id',''),'patient_symptoms':st.session_state.fu_symptoms,'enriched_symptoms':enriched,'status':'pending','llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),'llm_reasoning_original':result.get('original_reasoning',''),'llm_recommendation':tri.get('recommendation',''),'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),'llm_patient_explanation':tri.get('patient_explanation',''),'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),'prompt_chain_complete':True,'prompt_chain_questions':json.dumps([q for q,_ in all_qa]),'prompt_chain_answers':json.dumps([a for _,a in all_qa]),'prompt_chain_rounds':round_num,'gate_decision':result.get('gate_decision',''),'agent_action_trace':json.dumps(result.get('action_trace',[])),'agent_confidence':json.dumps(conf),'agent_plan':result.get('agent_plan',''),'evaluator_enhanced':result.get('evaluator_enhanced',False),'fhir_bundle':json.dumps(result.get('fhir_bundle',{})),'crew_mode':result.get('crew_mode',''),'retrieval_debug':json.dumps(tri.get('retrieval_debug',[])),'evaluator_feedback':json.dumps(result.get('evaluator_feedback',{})),'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                        st.success("✅ Submitted!"); st.info(f"**Case ID:** `{cid}` — Your case has been submitted. A nurse will review it shortly.")
                        st.session_state.fu_stage = 'initial'; st.session_state.fu_qa = []; st.session_state.fu_round = 0

    with tab2:
        chk = st.text_input("Enter your ticket number:", key="chk", placeholder="e.g. T1234")
        if chk:
            cases = db_get_by_ticket(chk)
            if not cases:
                st.info("No cases found.")
            for c in cases:
                if c['status'] == 'reviewed':
                    tier = c.get('final_tier','Unknown'); colors = {'Urgent':'red','Routine':'orange','Self-care':'green'}; icons = {'Urgent':'🔴','Routine':'🟡','Self-care':'🟢'}
                    st.markdown(f"### {icons.get(tier,'⚪')} Case {c['case_id']}"); st.markdown(f"**Final Decision:** :{colors.get(tier,'gray')}[{tier}]")
                    bd = {}
                    try:
                        bd = json.loads(c.get('booking_details','{}') or '{}')
                    except Exception:
                        pass
                    if bd:
                        st.markdown("---"); st.markdown("#### 📅 Your Appointment")
                        for fld, lbl in [('type','Type'),('doctor','Doctor'),('time','When'),('dept','Department'),('room','Room')]:
                            if bd.get(fld): st.markdown(f"**{lbl}:** {bd[fld]}")
                        if bd.get('instructions'): st.info(bd['instructions'])
                        if bd.get('guidance'):
                            st.markdown("**Self-Care Guidance:**"); st.markdown(bd['guidance'])
                    if c.get('llm_patient_explanation'):
                        st.markdown("---"); st.markdown("#### What This Means For You"); st.markdown(c['llm_patient_explanation'])
                    st.markdown("---"); st.markdown("#### Case Status")
                    st.markdown("Your case was submitted and reviewed by the nurse.")
                    if c.get('nurse_notes') and c['nurse_notes'].strip():
                        st.markdown("---"); st.markdown("#### Nurse Notes"); st.markdown(f"> {c['nurse_notes']}")
                elif c['status'] == 'pending':
                    st.warning(f"**{c['case_id']}** -- Waiting for nurse review...")

# ======================== NURSE DASHBOARD ====================================
elif page == "Nurse Dashboard":
    nurse = check_nurse_login()
    if not nurse:
        nurse_login_form()
    else:
        st.title("👩‍⚕️ Nurse Triage Review Dashboard")
        st.markdown(f"Logged in as: **{nurse}**")
        if st.sidebar.button("Logout"):
            del st.session_state['nurse_name']; del st.session_state['nurse_username']; st.rerun()

        pending = db_get_all(status='pending'); reviewed = db_get_all(status='reviewed')
        c1,c2,c3 = st.columns(3)
        c1.metric("Pending",len(pending)); c2.metric("Reviewed",len(reviewed))
        ov = sum(1 for c in reviewed if (c.get('nurse_action') or '').startswith('override'))
        c3.metric("Overrides",ov)
        tab_r, tab_h = st.tabs(["Review Cases","Review History"])
        with tab_r:
            if not pending:
                st.info("No pending cases.")
                if st.button("Refresh", use_container_width=True): st.rerun()
            else:
                opts = {f"{c['case_id']} -- Ticket #{c['ticket_number']}":c['case_id'] for c in pending}
                sel = st.selectbox("Select case:", list(opts.keys())); case = db_get_one(opts[sel])
                if case:
                    st.markdown("---"); cl, cr = st.columns([3,2])
                    with cl:
                        st.subheader("Patient Symptoms")
                        st.text_area("",value=case['patient_symptoms'],height=100,disabled=True,label_visibility="collapsed")
                        pq = case.get('prompt_chain_questions') or case.get('router_questions')
                        pa = case.get('prompt_chain_answers') or case.get('router_answers')
                        if pq:
                            try:
                                qs = json.loads(pq); ans = json.loads(pa or '[]')
                                if qs:
                                    rounds = case.get('prompt_chain_rounds', len(qs))
                                    with st.expander(f"🔗 Prompt Chain Q&A ({len(qs)} questions, {rounds} round(s))", expanded=False):
                                        for i,(q,a) in enumerate(zip(qs,ans)):
                                            st.markdown(f"**Q{i+1}:** {q}"); st.markdown(f"**A{i+1}:** {a}"); st.markdown("")
                            except Exception:
                                pass
                        st.subheader("AI Triage Assessment")
                        tier = case.get('llm_urgency','Unknown')
                        tc = {'Urgent':'red','Routine':'orange','Self-care':'green'}
                        m1,m2 = st.columns(2)
                        m1.markdown(f"**Urgency:** :{tc.get(tier,'gray')}[{tier}]"); m2.markdown(f"**Mode:** `{case.get('rag_mode','')}`")
                        if case.get('agent_confidence'):
                            try:
                                ac = json.loads(case['agent_confidence'])
                                pct = ac.get('confidence_pct', 50)
                                st.progress(min(pct, 100) / 100, text=f"🤖 Agent confidence: {pct}%")
                                if ac.get('would_change_if') and ac['would_change_if'] != 'N/A':
                                    st.info(f"🔍 **Would change if:** {ac['would_change_if']}")
                                if ac.get('nurse_watch_for') and ac['nurse_watch_for'] != 'N/A':
                                    st.warning(f"👩‍⚕️ **Nurse should check:** {ac['nurse_watch_for']}")
                            except Exception:
                                pass
                        if case.get('llm_evidence') and case['llm_evidence'] not in ('','No evidence retrieved.'):
                            st.markdown("**📚 Retrieved Medical Evidence:**")
                            with st.expander("View retrieved passages",expanded=True): st.markdown(case['llm_evidence'])
                        if case.get('llm_reasoning'): st.markdown(f"**Clinical Reasoning:** {case['llm_reasoning']}")
                        if case.get('llm_recommendation'): st.markdown(f"**Recommendation:** {case['llm_recommendation']}")
                        if case.get('llm_next_steps'): st.markdown(f"**Next Steps:** {case['llm_next_steps']}")
                        if case.get('llm_patient_explanation'): st.markdown(f"**Patient Explanation:** _{case['llm_patient_explanation']}_")
                        if case.get('llm_sources') and case['llm_sources'] not in ('None',''):
                            with st.expander("View sources (IEEE citations)"): st.markdown(case['llm_sources'])
                    with cr:
                        st.subheader("Your Clinical Decision")
                        tiers = ['Urgent','Routine','Self-care']; idx = tiers.index(tier) if tier in tiers else 0
                        nurse_tier = st.selectbox("Final Urgency Tier:",tiers,index=idx)
                        is_ov = nurse_tier != tier
                        if is_ov:
                            d = "upgrading" if tiers.index(nurse_tier)<tiers.index(tier) else "downgrading"
                            st.warning(f"You are **{d}** from {tier} to {nurse_tier}.")
                            ov_reason = st.text_area("Override reason (required):",height=80,placeholder=f"Why are you {d}?")
                        else:
                            ov_reason = ''
                        notes = st.text_area("Clinical notes (optional):",height=80,placeholder="Additional observations...")
                        can = True
                        if is_ov and not (ov_reason or '').strip():
                            can = False; st.error("Override reason required.")
                        if st.button("Confirm Decision",type="primary",use_container_width=True,disabled=not can):
                            if nurse_tier == tier: act = 'approve'
                            elif tiers.index(nurse_tier)<tiers.index(tier): act = 'override_upgrade'
                            else: act = 'override_downgrade'
                            bk = book_action(case['case_id'],nurse_tier,case.get('patient_symptoms',''))
                            db_update(case['case_id'],{'nurse_tier':nurse_tier,'nurse_action':act,'nurse_notes':notes,'nurse_name':nurse,'nurse_override_reason':ov_reason,'nurse_timestamp':datetime.utcnow().isoformat(),'final_tier':nurse_tier,'booking_status':bk['status'],'booking_details':json.dumps(bk),'booking_agent_decision':bk.get('agent_decision',''),'status':'reviewed'})
                            st.success(f"✅ Recorded: **{nurse_tier}** ({act})"); st.balloons(); st.rerun()
        with tab_h:
            if not reviewed:
                st.info("No reviewed cases yet.")
            else:
                filt = st.selectbox("Filter:",['All','Approved','Overridden'])
                for c in reviewed:
                    a = c.get('nurse_action') or ''
                    if filt == 'Approved' and a != 'approve': continue
                    if filt == 'Overridden' and not a.startswith('override'): continue
                    em = {'approve':'✅','override_upgrade':'⬆️','override_downgrade':'⬇️'}.get(a,'❓')
                    lb = {'approve':'Approved','override_upgrade':'Upgraded','override_downgrade':'Downgraded'}.get(a,a)
                    with st.expander(f"{em} {c['case_id']} -- AI: {c.get('llm_urgency','?')} -> Final: {c.get('final_tier','?')} ({lb})"):
                        st.subheader("Patient Symptoms")
                        st.text_area("", value=c.get('patient_symptoms',''), height=100, disabled=True, label_visibility="collapsed", key=f"hist_symptoms_{c['case_id']}")

                        pq = c.get('prompt_chain_questions') or c.get('router_questions')
                        pa = c.get('prompt_chain_answers') or c.get('router_answers')
                        if pq:
                            try:
                                qs = json.loads(pq); ans = json.loads(pa or '[]')
                                if qs:
                                    with st.expander(f"Prompt Chain Q&A ({len(qs)} questions)", expanded=False):
                                        for i,(q,a2) in enumerate(zip(qs,ans)):
                                            st.markdown(f"**Q{i+1}:** {q}")
                                            st.markdown(f"**A{i+1}:** {a2}")
                            except Exception:
                                pass

                        st.subheader("AI Triage Assessment")
                        h1,h2 = st.columns(2)
                        h1.markdown(f"**Urgency:** {c.get('llm_urgency','Unknown')}")
                        h2.markdown(f"**Mode:** `{c.get('rag_mode','')}`")

                        if c.get('agent_confidence'):
                            try:
                                ac = json.loads(c['agent_confidence'])
                                pct = ac.get('confidence_pct', 50)
                                st.progress(min(pct, 100) / 100, text=f"Agent confidence: {pct}%")
                                if ac.get('would_change_if') and ac['would_change_if'] != 'N/A':
                                    st.info(f"Would change if: {ac['would_change_if']}")
                                if ac.get('nurse_watch_for') and ac['nurse_watch_for'] != 'N/A':
                                    st.warning(f"Nurse should check: {ac['nurse_watch_for']}")
                            except Exception:
                                pass

                        if c.get('llm_evidence') and c['llm_evidence'] not in ('','No evidence retrieved.'):
                            with st.expander("Retrieved Medical Evidence", expanded=False):
                                st.markdown(c['llm_evidence'])

                        if c.get('llm_reasoning'): st.markdown(f"**Clinical Reasoning:** {c['llm_reasoning']}")
                        if c.get('llm_recommendation'): st.markdown(f"**Recommendation:** {c['llm_recommendation']}")
                        if c.get('llm_next_steps'): st.markdown(f"**Next Steps:** {c['llm_next_steps']}")
                        if c.get('llm_patient_explanation'): st.markdown(f"**Patient Explanation:** _{c['llm_patient_explanation']}_")
                        if c.get('llm_sources') and c['llm_sources'] not in ('None',''):
                            with st.expander("Sources"):
                                st.markdown(c['llm_sources'])

                        st.subheader("Nurse Final Decision")
                        st.markdown(f"**Final tier:** {c.get('final_tier','')}")
                        st.markdown(f"**Action:** {c.get('nurse_action','')}")
                        if c.get('nurse_override_reason'): st.markdown(f"**Override reason:** {c['nurse_override_reason']}")
                        if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")

                        try:
                            bd = json.loads(c.get('booking_details','{}') or '{}')
                            if bd and bd.get('type'):
                                st.markdown(f"**Booking:** {bd['type']} — {bd.get('doctor','N/A')} — {bd.get('time','N/A')}")
                        except Exception:
                            pass

                        st.caption(f"Nurse: {c.get('nurse_name','')} | {(c.get('nurse_timestamp','') or '')[:19]}")

# ======================== DEVELOPER DASHBOARD ================================
elif page == "Developer Dashboard":
    st.title("🔧 Developer Dashboard")
    st.caption("Audit trail -- AI vs Nurse decisions, two-agent CrewAI traces, override analysis, cross-case learning")
    
    # System Status Section
    st.markdown("---")
    st.subheader("🔧 System Status")
    sys1, sys2, sys3, sys4 = st.columns(4)
    
    with sys1:
        st.markdown("**📚 Hybrid RAG**")
        if RAG_AVAILABLE:
            st.success(f"✅ Connected ({len(KB_CHUNKS)} chunks)")
        else:
            st.error("❌ Demo mode")
    
    with sys2:
        st.markdown("**📅 Scheduler**")
        if SCHEDULE_AVAILABLE:
            st.success("✅ Connected")
            if st.session_state.booked_slots:
                st.caption(f"Booked this session: {len(st.session_state.booked_slots)}")
        else:
            st.error("❌ Demo mode")
    
    with sys3:
        st.markdown("**🤖 CrewAI**")
        if CREWAI_OK:
            st.success("✅ Real CrewAI initialized")
        else:
            st.error("❌ CrewAI not installed")
    
    with sys4:
        st.markdown("**🧠 Groq**")
        if GROQ_OK:
            st.success("✅ Enabled")
        else:
            st.error("❌ Disabled")
    
    st.caption("⚙️ No hardcoded follow-up fallback: JSON is repaired by Groq, otherwise error is shown.")
    st.markdown("---")
    
    all_cases = db_get_all()
    if not all_cases:
        st.info("No cases yet.")
    else:
        reviewed_cases = [c for c in all_cases if c['status'] == 'reviewed']
        pending_cases = [c for c in all_cases if c['status'] == 'pending']
        override_cases = [c for c in reviewed_cases if (c.get('nurse_action') or '').startswith('override')]
        upgrade_cases = [c for c in override_cases if c.get('nurse_action') == 'override_upgrade']
        downgrade_cases = [c for c in override_cases if c.get('nurse_action') == 'override_downgrade']
        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Total",len(all_cases)); d2.metric("Reviewed",len(reviewed_cases)); d3.metric("Pending",len(pending_cases))
        d4.metric("Overrides",len(override_cases))
        d5.metric("Override Rate",f"{len(override_cases)/max(len(reviewed_cases),1)*100:.0f}%")

        if reviewed_cases:
            st.markdown("---"); st.markdown("### 📈 Triage Distribution")
            tier_counts = {}
            for c in reviewed_cases:
                t = c.get('final_tier', 'Unknown')
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tcols = st.columns(len(tier_counts))
            tier_icons = {'Urgent':'🔴','Routine':'🟡','Self-care':'🟢'}
            for i, (t, cnt) in enumerate(tier_counts.items()):
                with tcols[i]: st.metric(f"{tier_icons.get(t,'⚪')} {t}", cnt)

        if override_cases:
            st.markdown("---"); st.markdown("### ⚖️ Override Analysis")
            o1,o2 = st.columns(2); o1.metric("⬆️ Upgrades",len(upgrade_cases)); o2.metric("⬇️ Downgrades",len(downgrade_cases))
            if upgrade_cases:
                with st.expander("View upgrade justifications"):
                    for c in upgrade_cases:
                        st.markdown(f"**{c['case_id']}:** {c.get('llm_urgency','?')} -> {c.get('final_tier','?')}"); st.markdown(f"> {c.get('nurse_override_reason','No reason')}"); st.markdown("")
            if downgrade_cases:
                with st.expander("View downgrade justifications"):
                    for c in downgrade_cases:
                        st.markdown(f"**{c['case_id']}:** {c.get('llm_urgency','?')} -> {c.get('final_tier','?')}"); st.markdown(f"> {c.get('nurse_override_reason','No reason')}"); st.markdown("")

        st.markdown("---"); st.markdown("### 🧠 Cross-Case Learning Agent")
        st.caption("AI analyzes patterns across all reviewed cases to detect biases and generate new knowledge")
        if st.button("🔍 Run Cross-Case Analysis", use_container_width=True):
            if len(reviewed_cases) < 3:
                st.warning("Need at least 3 reviewed cases for meaningful analysis.")
            else:
                with st.spinner("🧠 Agent is analyzing decision patterns across all cases..."):
                    analysis = agent_cross_case_analysis()
                if analysis.get('status') == 'error' or analysis.get('status') == 'insufficient_data':
                    st.error(analysis.get('message', 'Analysis failed.'))
                else:
                    a1, a2 = st.columns(2)
                    with a1: st.metric("📊 Agreement Rate", analysis.get('agreement_rate', 'N/A'))
                    with a2: st.metric("📋 Cases Analyzed", len(reviewed_cases))
                    if analysis.get('bias_pattern'):
                        st.markdown(f"**Bias Pattern:** {analysis['bias_pattern']}")
                    if analysis.get('failure_modes'):
                        st.markdown("**AI Failure Modes:**")
                        for fm in analysis['failure_modes']:
                            st.markdown(f"  - {fm}")
                    if analysis.get('improvement'):
                        st.success(f"**💡 Suggested Improvement:** {analysis['improvement']}")
                    if analysis.get('novel_finding') and analysis['novel_finding'] != 'null':
                        st.info(f"**🔬 Novel Finding:** {analysis['novel_finding']}")

        st.markdown("---"); st.markdown("### 📊 Case-by-Case Comparison")
        vf = st.selectbox("Show:",['All','Reviewed Only','Pending Only','Overrides Only'])
        if vf == 'Reviewed Only': display = reviewed_cases
        elif vf == 'Pending Only': display = pending_cases
        elif vf == 'Overrides Only': display = override_cases
        else: display = all_cases
        for c in display:
            ai_t = c.get('llm_urgency','--'); nr_t = c.get('final_tier','--')
            act = c.get('nurse_action') or 'pending'; is_rev = c['status'] == 'reviewed'
            agree = ai_t == nr_t and is_rev
            emoji = '✅' if agree else ('⚠️' if act.startswith('override') else '⏳')
            with st.expander(f"{emoji} {c['case_id']} -- Ticket #{c.get('ticket_number','')} -- {c['status']}"):
                x1,x2,x3 = st.columns(3)
                with x1:
                    st.markdown("**🤖 AI Decision**")
                    st.markdown(f"Urgency: **{ai_t}**")
                    st.markdown(f"Mode: {c.get('rag_mode','--')}")
                    st.markdown(f"Plan: `{c.get('agent_plan','--')}`")
                    st.markdown(f"Gate: {c.get('gate_decision','--')}")
                    st.markdown(f"Evaluator: {'✅' if c.get('evaluator_enhanced') else '—'}")
                    if c.get('agent_confidence'):
                        try:
                            ac = json.loads(c['agent_confidence'])
                            st.markdown(f"Agent conf: {ac.get('confidence_pct','?')}%")
                        except Exception:
                            pass
                with x2:
                    st.markdown("**👩‍⚕️ Nurse Decision**")
                    st.markdown(f"Tier: **{nr_t}**" if is_rev else "_Pending_")
                    st.markdown(f"Action: {act}" if is_rev else "")
                    st.markdown(f"Nurse: {c.get('nurse_name','--')}" if is_rev else "")
                with x3:
                    st.markdown("**Result**")
                    st.markdown("✅ Agreement" if agree else (f"⚠️ {ai_t} -> {nr_t}" if is_rev else "⏳ Awaiting"))
                st.markdown("---")
                st.markdown("**Developer Debug Details**")
                if c.get('fhir_bundle'):
                    try:
                        fb = json.loads(c.get('fhir_bundle') or '{}')
                        if fb:
                            with st.expander("📄 FHIR-like handoff", expanded=False):
                                st.json(fb)
                    except Exception:
                        st.warning("Could not parse FHIR-like handoff.")
                if c.get('retrieval_debug'):
                    try:
                        rd = json.loads(c.get('retrieval_debug') or '[]')
                        if rd:
                            with st.expander("🔎 Hybrid Retrieval Scores", expanded=False):
                                st.dataframe(rd, use_container_width=True)
                    except Exception:
                        st.warning("Could not parse retrieval debug.")
                if c.get('evaluator_feedback'):
                    try:
                        ef = json.loads(c.get('evaluator_feedback') or '{}')
                        if ef:
                            with st.expander("🧪 Evaluator Feedback", expanded=False):
                                st.json(ef)
                    except Exception:
                        st.warning("Could not parse evaluator feedback.")
                if c.get('agent_action_trace'):
                    try:
                        trace = json.loads(c.get('agent_action_trace') or '[]')
                        if trace:
                            with st.expander(f"🤖 Two-Agent CrewAI Trace ({len(trace)} steps)", expanded=False):
                                tool_icons = {
                                    'PROMPT_CHAIN_GATE':'🧠',
                                    'ASK_FOLLOWUP':'💬',
                                    'FHIR_FORMATTER':'📄',
                                    'GATE_DECISION':'🚦',
                                    'RAG_TRIAGE':'📚',
                                    'ASSESS_URGENCY':'🏥',
                                    'EVALUATOR':'🔬',
                                    'OPTIMIZER_RE_TRIAGE':'🔄',
                                    'CONFIDENCE_CHECK':'📊',
                                    'ASK_FOLLOWUP_ERROR':'❌',
                                    'RAG_TRIAGE_ERROR':'❌',
                                    'EVALUATOR_ERROR':'❌',
                                    'CONFIDENCE_CHECK_ERROR':'❌'
                                }
                                for step in trace:
                                    tool = step.get('tool','')
                                    icon = tool_icons.get(tool, '▶️')
                                    agent_label = step.get('agent','')
                                    st.markdown(f"**{icon} Step {step.get('step','')}: {tool}** {f'({agent_label})' if agent_label else ''}")
                                    if step.get('reason'):
                                        st.markdown(f"  _Reason: {step['reason']}_")
                                    if step.get('result'):
                                        st.markdown(f"  Result: {step['result']}")
                                    if step.get('plan'):
                                        st.markdown(f"  Plan: `{step['plan']}`")
                    except Exception:
                        st.warning("Could not parse agent trace.")
                if c.get('llm_reasoning_original'):
                    with st.expander("🔬 Evaluator Changes (Before/After)", expanded=False):
                        col_b, col_a = st.columns(2)
                        with col_b:
                            st.markdown("**Original RAG Output:**")
                            st.text_area("", value=(c.get('llm_reasoning_original',''))[:1000], height=220, disabled=True, label_visibility="collapsed", key=f"dev_before_{c['case_id']}")
                        with col_a:
                            st.markdown("**After Evaluator:**")
                            st.text_area("", value=(c.get('llm_reasoning',''))[:1000], height=220, disabled=True, label_visibility="collapsed", key=f"dev_after_{c['case_id']}")
                if c.get('nurse_override_reason'): st.markdown(f"**Override:** {c['nurse_override_reason']}")
                if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")
                try:
                    bd = json.loads(c.get('booking_details','{}') or '{}')
                    if bd and bd.get('type'): st.markdown(f"**📅 Booking:** {bd['type']} -- {bd.get('doctor','N/A')} -- {bd.get('time','N/A')}")
                except Exception:
                    pass
                st.caption(f"Symptoms: {(c.get('patient_symptoms',''))[:120]}...")
        st.markdown("---")
        export_data = json.dumps(all_cases, indent=2, default=str)
        st.download_button("📥 Export All Cases as JSON", data=export_data, file_name=f"triage_export_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json", use_container_width=True)