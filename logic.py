import os
import json
import re
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

# Speech & TTS
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

# Reporting/PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# LangChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------- ENV -------------------- #
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_DB_DIR = os.path.abspath("vectorstore/faiss")
DEFAULT_MODEL = "gemini-1.5-flash"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment (.env)")


# -------------------- RETRIEVER -------------------- #
def create_retriever(embeddings, faiss_path="vectorstore/faiss", bm25_path="vectorstore/bm25.pkl", k=5):
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    with open(bm25_path, "rb") as f:
        import pickle
        bm25_retriever = pickle.load(f)
        bm25_retriever.k = k
    return EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.7, 0.3])


# -------------------- UTILITIES -------------------- #
def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*|`", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def speak_autoplay(text: str, lang_code: str = "en-IN"):
    """Convert text to speech and auto-play."""
    if not text:
        return
    try:
        base_lang = "hi" if any("\u0900" <= c <= "\u097F" for c in text) else "en"
        tts = gTTS(text=text, lang=base_lang, tld="co.in")
        filename = "response.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("⚠️ TTS Error:", e)


def save_bytes_to_file(data: bytes, path: str):
    with open(path, "wb") as f:
        f.write(data)
    return path


def mic_input_sr(language_code="en-IN", timeout=None, phrase_time_limit=None):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    text = r.recognize_google(audio, language=language_code)
    return text


# -------------------- PROMPTS -------------------- #
reasoning_prompt_template = """You are a certified medical assistant acting as a doctor.
Your role is to provide safe, empathetic, and medically accurate responses.

STRICT INSTRUCTIONS:
- Do NOT invent or hallucinate facts.  
- Only use the provided patient details and retrieved medical context.  
- If uncertain or if symptoms are severe, advise the patient to seek immediate medical consultation.  
- Be empathetic, supportive, and clear.  
- Always keep patient safety as the top priority.  
- Respond strictly in the same language the user is using.  

Tasks:
1. Interpret the patient’s symptoms.  
2. Reference relevant medical knowledge or studies (from the provided context).  
3. Offer a reasoned response as a medical expert.  
4. Ask if the patient wants to provide more details.  

Symptoms/Details:
  {joined_facts}

Medical Context:
  {retrieved_context}
"""

action_plan_prompt_template = """You are a medical expert. Based on the patient’s symptoms and context,
generate a clear action plan divided into sections:

**Do:** Helpful steps, precautions, treatments, consultations, self-care.  
**Don’t:** Harmful actions, risky behaviors, wrong practices.  
**Diet:** Foods to eat, foods to avoid.  
**Exercise/Yoga:** Helpful stretches, light exercise, or yoga if appropriate.  

Guidelines:
- Do NOT hallucinate. Only use safe, general medical knowledge.  
- Keep language empathetic, clear, and simple.  
- If unsure, recommend consulting a doctor.  
- Respond strictly in the same language as the patient.  

Symptoms/Details:
  {joined_facts}

Medical Context:
  {retrieved_context}
"""

medical_note_prompt_template = """You are a healthcare advisor. Based on the patient’s symptoms and
retrieved medical context, draft a professional **medical note or advice summary**.

It should include:
- Summary of patient’s condition.  
- References to medical guidance (from context).  
- Recommended next actions (tests, consultation, precautions).  
- Professional and respectful tone.  

STRICT RULES:
- Do NOT fabricate facts.  
- If unsure, recommend doctor consultation.  
- Respond in the same language as the patient.  

Symptoms/Details:
  {joined_facts}

Medical Context:
  {retrieved_context}
"""

social_prompt_template = """You are a healthcare assistant bot. Draft a supportive,
awareness-based social media post (tweet/reddit/LinkedIn) based on the patient’s health journey.

Guidelines:
- Tone must be empathetic and supportive.  
- Do NOT add medical claims beyond context.  
- Ask if the patient wants to add more details.  
- Use the same language as the patient.  

Symptoms/Details:
  {joined_facts}

Medical Context:
  {retrieved_context}
"""

links_prompt_template = """You are a medical assistant chatbot. Suggest 4–7 **trusted external resources**
for healthcare support.

Rules:
- Use only official portals (.gov, .gov.in, .org, .who.int, .nih.gov, etc.).  
- Use markdown clickable links with a short description.  
- Do NOT include unverified sources.  
- Ask if they want to add more details.  

Symptoms/Details:
  {joined_facts}

Medical Context:
  {retrieved_context}
"""

followup_prompt_template = """Based on the patient’s symptoms and medical context,
generate some useful clarifying questions to better understand their condition.

Rules:
- Keep tone empathetic and supportive.  
- Do NOT hallucinate.  
- Use the same language as the patient.  

Symptoms/Details:
  {joined_facts}

Medical Context:
  {retrieved_context}
"""

# -------------------- SECTIONS -------------------- #
SECTION_ORDER = [
    "medical_response",
    "action_plan",
    "medical_note",
    "social_post",
    "helpful_links",
    "follow_ups",
]

PROMPT_MAP = {
    "medical_response": reasoning_prompt_template,
    "action_plan": action_plan_prompt_template,
    "medical_note": medical_note_prompt_template,
    "social_post": social_prompt_template,
    "helpful_links": links_prompt_template,
    "follow_ups": followup_prompt_template,
}


# -------------------- BOT CORE -------------------- #
def init_bot(db_dir: str = DEFAULT_DB_DIR, llm_model: str = DEFAULT_MODEL, max_questions: int = 6):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # ✅ HuggingFace Embeddings instead of Gemini embeddings
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    retriever = create_retriever(embeddings, faiss_path=db_dir)

    # ✅ Keep Gemini LLM
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3, google_api_key=GEMINI_API_KEY)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    state = {
        "retriever": retriever,
        "llm": llm,
        "qa_chain": qa_chain,
        "initial_query": "",
        "qa_history": [],
        "asked_questions": [],
        "user_answers": [],
        "questions_asked": 0,
        "max_questions": max_questions,
        "final_output": {},
        "sections_done": {s: False for s in SECTION_ORDER},
        "lang_code": "en-IN",
        "_last_generated_text": None,
    }
    return state


def generate_guided_questions(state, symptom_summary: str, num: int = 6):
    default_questions = [
        "When did the symptoms first start?",
        "Can you describe the main symptoms you are experiencing?",
        "Have the symptoms been getting better, worse, or staying the same?",
        "Do you have any known medical conditions or allergies?",
        "Are you currently taking any medications or supplements?",
        "Have you recently traveled, had any surgeries, or been exposed to illness?",
    ][:num]

    prompt = f"""
A patient described their health concern as:
\"\"\"{symptom_summary}\"\"\"  

Generate {num} clarifying follow-up questions (one per line).  

Rules:
- Be empathetic, simple, and medically relevant.  
- Cover duration, severity, triggers, lifestyle, and history.  
- Use the same language as the patient.  
"""

    try:
        resp = state["llm"].invoke(prompt).content
        lines = [re.sub(r'^\d+\.\s*', '', ln).strip() for ln in resp.splitlines() if ln.strip()]
        return lines[:num] if lines else default_questions
    except Exception as e:
        print("⚠️ Error generating guided questions:", e)
        return default_questions


def generate_clarifying_question(state):
    asked_text = "\n".join([f"- {q}" for q in state["asked_questions"]])
    qa_context = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(state["qa_history"])])

    prompt = f"""
First interaction: Greet warmly, with empathy, and assure the patient you are here to help.  

Initial user query: "{state["initial_query"]}"  
Previous Q&A:
{qa_context}

Already asked:
{asked_text}

Generate ONE safe clarifying question (avoid duplication).  
"""

    resp = state["llm"].invoke(prompt).content
    for line in resp.splitlines():
        line = line.strip()
        if line:
            q = re.sub(r'^\d+\.\s*', '', line)
            if not q.endswith("?"):
                q = q.rstrip(".") + "?"
            state["asked_questions"].append(q)
            state["questions_asked"] += 1
            return q
    fallback = "Can you describe exactly what happened and when it first occurred?"
    state["asked_questions"].append(fallback)
    state["questions_asked"] += 1
    return fallback


def record_answer(state, answer: str):
    last_q = state["asked_questions"][-1]
    state["qa_history"].append((last_q, answer))
    state["user_answers"].append(answer)
    return True


def _get_joined_facts(state):
    return "\n".join([f"{i+1}. {a}" for i, a in enumerate(state["user_answers"]) if a.strip()])


def _get_retrieved_context(state, joined_facts: str, top_k: int = 5):
    docs = state["retriever"].get_relevant_documents(joined_facts)
    return "\n\n".join(d.page_content for d in docs[:top_k]) if docs else ""


def generate_section(state, section_name: str):
    joined_facts = _get_joined_facts(state)
    retrieved_context = _get_retrieved_context(state, joined_facts)
    prompt_template = PROMPT_MAP.get(section_name)
    prompt = (
        f"The user is speaking in language code: {state['lang_code']}.\n"
        f"Respond in the same language.\n\n"
        + prompt_template.format(joined_facts=joined_facts, retrieved_context=retrieved_context)
    )
    resp = state["llm"].invoke(prompt).content.strip()
    if section_name in ("helpful_links", "follow_ups"):
        lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
        state["final_output"][section_name] = lines
    else:
        state["final_output"][section_name] = resp
    state["sections_done"][section_name] = True
    state["_last_generated_text"] = resp
    return resp


def generate_all_sections(state):
    for sec in SECTION_ORDER:
        if not state["sections_done"].get(sec):
            generate_section(state, sec)
    return state["final_output"]


def export_to_pdf_bytes(state, output_dict: dict = None):
    if output_dict is None:
        output_dict = state["final_output"]

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    pdf_sections = [
        ("medical_response", "Medical Response"),
        ("action_plan", "Action Plan"),
        ("medical_note", "Medical Note"),
        ("social_post", "Social Post"),
        ("helpful_links", "Helpful Links"),
        ("follow_ups", "Follow-up Questions"),
    ]

    for key, title in pdf_sections:
        content = output_dict.get(key, "")
        if isinstance(content, list):
            content_text = "\n".join(content)
        else:
            content_text = content or ""

        if content_text:
            elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            elements.append(Spacer(1, 0.15 * inch))
            elements.append(Paragraph(str(content_text).replace("\n", "<br/>"), styles["BodyText"]))
            elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def save_state(state, path: str):
    payload = {
        "initial_query": state["initial_query"],
        "qa_history": state["qa_history"],
        "asked_questions": state["asked_questions"],
        "user_answers": state["user_answers"],
        "questions_asked": state["questions_asked"],
        "final_output": state["final_output"],
        "sections_done": state["sections_done"],
        "lang_code": state["lang_code"],
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_state(state, path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    state["initial_query"] = payload.get("initial_query", "")
    state["qa_history"] = payload.get("qa_history", [])
    state["asked_questions"] = payload.get("asked_questions", [])
    state["user_answers"] = payload.get("user_answers", [])
    state["questions_asked"] = payload.get("questions_asked", 0)
    state["final_output"] = payload.get("final_output", {})
    state["sections_done"] = payload.get("sections_done", {s: False for s in SECTION_ORDER})
    state["lang_code"] = payload.get("lang_code", "en-IN")
    return state


# ----------------- CLI Demo ----------------- #
def bot_run():
    state = init_bot()

    initial = input("Enter the user's initial complaint (single line):\n> ").strip()
    if not initial:
        print("No input provided; exiting.")
        return
    state["initial_query"] = initial

    for _ in range(state["max_questions"]):
        q = generate_clarifying_question(state)
        print("\nBot asks:", q)
        ans = input("Your answer (or 'skip'/'done'):\n> ").strip()
        if ans.lower() == "done":
            break
        if ans.lower() == "skip":
            record_answer(state, "")
            continue
        record_answer(state, ans)

    print("\nCollected facts:")
    for i, a in enumerate(state["user_answers"], 1):
        print(f"{i}. {a}")

    final = generate_all_sections(state)
    print("\n--- Generated Sections ---")
    for sec in SECTION_ORDER:
        val = final.get(sec)
        if isinstance(val, list):
            print(f"\n[{sec}]")
            for line in val:
                print("-", line)
        else:
            print(f"\n[{sec}]\n{val[:800]}\n{'...' if val and len(val) > 800 else ''}")

    pdf_bytes = export_to_pdf_bytes(state)
    out_pdf_path = f"patient_case_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pdf"
    save_bytes_to_file(pdf_bytes, out_pdf_path)
    print(f"\nSaved combined PDF to: {out_pdf_path}")

    state_path = f"patient_case_state_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    save_state(state, state_path)
    print(f"Saved case state to: {state_path}")


if __name__ == "__main__":
    bot_run()