import re
from io import BytesIO
from datetime import datetime, time
from typing import Tuple

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from langchain.tools import tool


# ---------- INTENT DETECTION ---------- #
def detect_intent(state, user_message: str) -> str:
    """Classify user intent into BOOK_APPOINTMENT, DECLINE_APPOINTMENT, or FOLLOW_UP."""
    prompt = f"""
You are an intent classifier for a medical assistant.
The patient message is:
"{user_message}"

Classify it into one of these intents:
- BOOK_APPOINTMENT (if they clearly want to book/schedule/yes)
- DECLINE_APPOINTMENT (if they clearly refuse/no/not now)
- FOLLOW_UP (if they ask another medical question or provide new symptoms)

Return only the intent label.
"""
    resp = state["llm"].invoke(prompt).content.strip().upper()
    if "BOOK" in resp:
        return "BOOK_APPOINTMENT"
    elif "DECLINE" in resp or "NO" in resp:
        return "DECLINE_APPOINTMENT"
    else:
        return "FOLLOW_UP"


# ---------- DOCTOR SELECTION ---------- #
def choose_specialist(symptom_note: str) -> str:
    symptom_note = symptom_note.lower()
    if any(word in symptom_note for word in ["chest", "heart", "bp", "pressure", "palpitation"]):
        return "Cardiologist"
    elif any(word in symptom_note for word in ["headache", "brain", "seizure", "stroke", "memory", "migraine"]):
        return "Neurologist"
    elif any(word in symptom_note for word in ["lung", "breath", "asthma", "cough", "respiration"]):
        return "Pulmonologist"
    elif any(word in symptom_note for word in ["stomach", "liver", "digest", "ulcer", "gastric"]):
        return "Gastroenterologist"
    else:
        return "Physician"


# ---------- SLOT VALIDATION ---------- #
def validate_slot(appointment_time: str) -> Tuple[bool, str]:
    """
    Validate appointment time.
    Allowed: Mon–Sat, 10:00–16:00.
    """
    try:
        dt = datetime.strptime(appointment_time, "%Y-%m-%d %H:%M")
    except ValueError:
        return False, "Invalid format. Use YYYY-MM-DD HH:MM (24-hour)."

    if dt.weekday() > 5:  # Sunday = 6
        return False, "Appointments are available only Monday to Saturday."

    start, end = time(10, 0), time(16, 0)
    if not (start <= dt.time() <= end):
        return False, "Appointments are available only between 10:00 and 16:00."

    return True, "Valid slot."


# ---------- BOOKING TOOL ---------- #
@tool("book_appointment", return_direct=True)
def book_appointment(inputs: dict) -> dict:
    """
    Book a hospital appointment. Requires:
    - patient_name (str)
    - patient_age (str/int)
    - phone (str)
    - appointment_time (str, format 'YYYY-MM-DD HH:MM')
    - symptom_note (str)

    Returns appointment details if valid, else error.
    """
    patient_name = str(inputs.get("patient_name", "Unknown"))
    patient_age = inputs.get("patient_age", "Unknown")
    try:
        patient_age = int(patient_age)
    except (ValueError, TypeError):
        patient_age = "Unknown"

    phone = str(inputs.get("phone", "Unknown"))
    appointment_time = str(inputs.get("appointment_time", ""))
    symptom_note = str(inputs.get("symptom_note", "No history provided"))

    # validate slot
    ok, msg = validate_slot(appointment_time)
    if not ok:
        return {"status": "failed", "message": msg}

    doctor = choose_specialist(symptom_note)
    confirmation_id = f"APT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    return {
        "status": "confirmed",
        "patient_name": patient_name,
        "age": patient_age,
        "phone": phone,
        "doctor": doctor,
        "time": appointment_time,
        "symptom_note": symptom_note,
        "hospital": "City Care Hospital",
        "reference": confirmation_id,
    }


# ---------- RECEIPT TOOL ---------- #
@tool("generate_receipt", return_direct=True)
def generate_receipt(appointment_details: dict) -> bytes:
    """
    Generate a PDF receipt from appointment details.
    Input: appointment_details (dict)
    Output: PDF file in bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Hospital Appointment Receipt</b>", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))

    for k, v in appointment_details.items():
        elements.append(Paragraph(f"{k.replace('_',' ').title()}: {v}", styles["BodyText"]))
        elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
