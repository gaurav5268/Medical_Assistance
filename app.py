import streamlit as st
from datetime import datetime
import json
from book import detect_intent, book_appointment, generate_receipt, validate_slot

from logic import (
    init_bot,
    generate_clarifying_question,
    mic_input_sr,
    record_answer,
    speak_autoplay,
    generate_all_sections,
    export_to_pdf_bytes,
    DEFAULT_DB_DIR,
    SECTION_ORDER,
)

from PIL import Image
import pytesseract

# Configure pytesseract (update if different install path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Streamlit setup ---
st.set_page_config(page_title="Medical Bot", layout="wide")

if "bot" not in st.session_state:
    st.session_state.bot = init_bot(db_dir=DEFAULT_DB_DIR)

if "messages" not in st.session_state:
    st.session_state.messages = []

bot = st.session_state.bot

st.title("🏥 Medical Assistance")

# --- Chat history ---
chat_box = st.container()
with chat_box:
    for role, content in st.session_state.messages:
        if role == "user":
            st.markdown(
                f"<div style='color: white; background-color: #0d6efd; padding: 8px; border-radius: 12px; margin: 4px 0 4px auto; display: inline-block; max-width: 70%; text-align: right;'>{content}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color: black; background-color: #e9ecef; padding: 8px; border-radius: 12px; margin: 4px auto 4px 0; display: inline-block; max-width: 70%; text-align: left;'>{content}</div>",
                unsafe_allow_html=True,
            )

# --- Input options ---
col1, col2, col3 = st.columns([3, 7, 1])

with col1:
    if st.button("Upload", key="pause_left"):
        pass

    ocr_file = st.file_uploader("📷 OCR", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="ocr_upload")
    if ocr_file is not None:
        image = Image.open(ocr_file)
        with st.spinner("Extracting text..."):
            extracted_text = pytesseract.image_to_string(image, lang="eng")
        if extracted_text.strip():
            st.success("Text extracted!")
            user_input = extracted_text
        else:
            st.warning("No readable text found in the image.")
            user_input = None
    else:
        user_input = None

with col2:
    chat_input_val = st.chat_input("Type your message here...", key="user_input")
    if chat_input_val:
        user_input = chat_input_val

with col3:
    if st.button("🎙 Speak", key="pause_right"):
        st.info("🎙 Listening...")
        spoken = mic_input_sr()
        if spoken:
            user_input = spoken
        else:
            st.warning("Could not recognize speech, try again.")
            user_input = None

# --- Handle new user input ---
if user_input:
    st.session_state.messages.append(("user", f"{user_input}"))
    st.rerun()

# --- Bot response ---
if len(st.session_state.messages) > 0 and st.session_state.messages[-1][0] == "user":
    user_message = st.session_state.messages[-1][1]

    thinking_placeholder = st.empty()
    with thinking_placeholder:
        st.markdown(
            "<div style='text-align: left; color: gray; background-color: #f8f9fa; padding: 8px; border-radius: 12px; margin: 4px 0; display: inline-block; max-width: 70%;'>🤔 Thinking...</div>",
            unsafe_allow_html=True,
        )

    # --- Intent Detection ---
    intent = detect_intent(bot, user_message)

    if intent == "BOOK_APPOINTMENT":
        if "booking" not in st.session_state:
            st.session_state.booking = {}

        booking = st.session_state.booking

        # Collect name
        if "patient_name" not in booking:
            booking["patient_name"] = user_message.strip()
            response = "Got it. Please provide your age."
        # Collect age
        elif "patient_age" not in booking:
            booking["patient_age"] = user_message.strip()
            response = "Thanks! Can you also share your phone number?"
        # Collect phone
        elif "phone" not in booking:
            booking["phone"] = user_message.strip()
            response = "When would you like to book your appointment? (Format: YYYY-MM-DD HH:MM, 24-hour)"
        # Collect appointment time
        elif "appointment_time" not in booking:
            ok, msg = validate_slot(user_message.strip())
            if not ok:
                response = f"❌ {msg}\n\nPlease provide a valid time (YYYY-MM-DD HH:MM)."
            else:
                booking["appointment_time"] = user_message.strip()

                # Prepare symptom note
                if bot["user_answers"]:
                    symptom_note = "\n".join(bot["user_answers"])
                else:
                    symptom_note = "General checkup"
                    response = "Before I confirm, can you briefly describe your health concern?"

                booking["symptom_note"] = symptom_note

                # Book appointment directly
                appointment = book_appointment(booking)

                if appointment.get("status") == "confirmed":
                    receipt_pdf = generate_receipt(appointment)
                    thinking_placeholder.empty()
                    st.success(
                        f"✅ Appointment booked with {appointment['doctor']} at {appointment['hospital']} on {appointment['time']}"
                    )
                    st.download_button(
                        "Download Appointment Receipt",
                        receipt_pdf,
                        file_name="appointment_receipt.pdf",
                        mime="application/pdf",
                    )

                    st.session_state.booking = {}
                    response = "Your appointment has been successfully scheduled!"
                else:
                    response = appointment.get("message", "Something went wrong while booking.")

        st.session_state.messages.append(("assistant", response))

    elif intent == "DECLINE_APPOINTMENT":
        thinking_placeholder.empty()
        response = "Okay, I won’t book an appointment right now. You can ask me anytime later if you change your mind."
        st.session_state.messages.append(("assistant", response))

    else:  # FOLLOW_UP (symptom Q&A flow)
        if not bot["initial_query"]:
            bot["initial_query"] = user_message.strip()
            response = "Complaint recorded!.\n\n" + generate_clarifying_question(bot)
        elif bot["questions_asked"] < bot["max_questions"]:
            record_answer(bot, user_message.strip())
            response = generate_clarifying_question(bot)
        else:
            with st.spinner("Generating drafts..."):
                generate_all_sections(bot)
            response = "Drafts generated! Scroll down to view/download."

        thinking_placeholder.empty()
        st.session_state.messages.append(("assistant", response))
        st.session_state.to_speak = response
        st.rerun()

# --- TTS output ---
if "to_speak" in st.session_state:
    speak_autoplay(st.session_state.to_speak, bot["lang_code"])
    del st.session_state["to_speak"]

# --- Show generated sections + downloads ---
if bot["final_output"]:
    st.subheader("Generated Sections")
    for sec in SECTION_ORDER:
        content = bot["final_output"].get(sec)
        if not content:
            continue
        st.markdown(f"### {sec.replace('_',' ').title()}")
        if isinstance(content, list):
            for line in content:
                st.markdown(f"- {line}")
        else:
            st.markdown(content)

    pdf_bytes = export_to_pdf_bytes(bot)
    st.download_button(
        label="Download Full Report (PDF)",
        data=pdf_bytes,
        file_name=f"Patient_case_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pdf",
        mime="application/pdf",
    )

    state_data = {
        "initial_query": bot["initial_query"],
        "qa_history": bot["qa_history"],
        "asked_questions": bot["asked_questions"],
        "user_answers": bot["user_answers"],
        "final_output": bot["final_output"],
    }
    json_bytes = json.dumps(state_data, indent=2, ensure_ascii=False).encode("utf-8")

    st.download_button(
        label="Download Case State (JSON)",
        data=json_bytes,
        file_name=f"consumer_case_state_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json",
        mime="application/json",
    )
