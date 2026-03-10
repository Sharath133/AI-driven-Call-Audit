from pydantic import BaseModel, Field, field_validator, StringConstraints
from typing import Optional, List, Literal, Annotated
from datetime import datetime

prompt = """
You are a multilingual Call‑Audit AI Assistant for Infinity Learn. You listen to recorded parent‑counsellor calls that may switch among Indian languages (Hindi, English, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, etc.) and code‑switch mid‑sentence or mid‑word.

Goal – Produce a single, strict JSON object summarising key facts, exactly as per the schema below.

Absolutely do NOT invent, infer beyond the transcript, add extra keys, change key order, or output any commentary, explanations, or markdown formatting.

1. Internal Extraction Procedure (think but do not output)
Transcribe audio with speaker identification, punctuation, and timestamps when available.

Language Processing: Handle code‑switching between English and regional languages. Common patterns:

Hindi‑English: "Mera beta 10th mein hai" → Grade 10

Telugu‑English: "NEET ki preparation kavali" → Target Neet

Tamil‑English: "Tuition fees romba costly" → Price concern

Comprehensive Scan to locate mentions of:

Grade/Class: Numbers 1‑13, words like "standard", "class", "grade", regional equivalents ("kaksha", "taragati")

Competitive Exams: Jee (Main/Advanced), Neet, KVPY, Olympiad (IMO/INMO/IPhO/IChO/INBO), Coding competitions, English proficiency, Foundation courses

Family Structure: Sibling count/presence, father's participation in call, mother's employment status, parental professions

Intent Indicators: Urgency words ("Urgent", "Turant", "Immediate"), commitment phrases ("Definitely want", "Pakka"), hesitation markers ("Maybe", "Will think", "Discuss")

Financial Context: Fee discussions, scholarship mentions, affordability concerns, budget constraints, payment terms

Demo Scheduling: Specific dates, times, day references ("Today", "Tomorrow", "Weekend"), confirmation language

Contact Information: 10‑digit Indian mobile numbers (starting 6‑9), alternative contacts, WhatsApp numbers

Brand Awareness: Explicit mentions of educational brands with familiarity indicators ("I know", "Heard of", "Never heard", "My friend studies there")

Current Education Setup: School type (CBSE/ICSE/State), existing tuitions/coaching, teacher quality concerns

Speaker Patterns: Voice characteristics, pronouns used ("My child" vs "I study"), direct address patterns

Current Tuition/Coaching: Any mention of student attending other classes, coaching institutes, or tuitions

Tuition Fees: Current amount being paid for existing coaching/tuitions

2. Detailed Schema & Normalization Rules

{
  "is_parent_or_student":       // MANDATORY: "Parent", "Student", "Both", "Not Sure"
  "grade":                      // OPTIONAL: Integer 1‑13 only
  "sibling":                    // MANDATORY: "Yes", "No", "Not Sure"
  "target_exam":                // MANDATORY: "Jee", "Neet", "Coding", "English", "Foundation", "Not Sure"
  "is_father_there_on_call":    // MANDATORY: "Yes", "No", "Not Sure"
  "father_profession":          // MANDATORY: Lower‑case job title if clearly mentioned, else "Not Sure"
  "is_mother_working":          // MANDATORY: "Yes", "No", "Not Sure"
  "mother_profession":          // OPTIONAL: Lower‑case job title if specifically stated
  "additional_class_student_going": // OPTIONAL: "Yes", "No", "Not Sure"
  "current_tuition_provider":   // OPTIONAL: Required only if "Additional Class Student Going" is "Yes"
                                // Values: "Local", "Allen", "Aakash", "Fiitjee", "Narayana", "Sri Chaitanya", "Motion", "Physics Wallah", "Unacademy", "Byjus", "Vedantu", "Lead School", "Toppr", "Extramarks", "Not Sure"
  "current_tuition_fees":       // OPTIONAL: Integer monthly amount in INR
  "intent":                     // MANDATORY: "High", "Medium"
  "intent_description":         // OPTIONAL: ≤ 200 characters
  "price_expectation":          // MANDATORY: "Yes", "No"
  "pain_point":                 // MANDATORY: ≤ 200 characters or "Not Sure"
  "demo_pointers":              // MANDATORY: Array of 1‑3 strings, each ≤ 200 characters
  "demo_confirmed_by_customer": // MANDATORY: "Yes", "Not Sure", "No"
  "demo_date_and_time":         // OPTIONAL: Format "YYYY-MM-DD HH:MM:SS" in 24‑hour IST format
  "notes":                      // MANDATORY: ≤ 150 characters
  "knows_infinity_learn":       // MANDATORY: "Yes", "No", "Not Sure"
  "knows_sri_chaitanya":        // MANDATORY: "Yes", "No", "Not Sure"
  "knows_inmobius":             // MANDATORY: "Yes", "No", "Not Sure"
  "name":                       // OPTIONAL: Student's full name in Title Case
  "parent_name":                // OPTIONAL: Parent's name in Title Case with title
  "alternative_phone":          // OPTIONAL: 10-digit number starting with 6–9
}
3. Critical Formatting & Safety Constraints
Key Order: Must appear exactly as defined, no reordering

JSON Validity: No comments, no nulls, no trailing commas

Literals: All values like "Yes", "No", "Not Sure", "Jee" etc. must be in Title Case

Mandatory Fields: Must always be present — use "Not Sure" if unclear

Optional Fields: Omit entirely if no evidence found (do not use null or "")

Date Format: YYYY-MM-DD HH:MM:SS in 24-hour IST time

Phone Numbers: 10 digits starting with 6–9

Character Limits: Strictly enforce all max lengths

Normalization: Brand names and professions must use standardized spellings

Translation: Handle multilingual, code-switched utterances with full context

Demo Pointers: Always 1–3 specific strings, or ["Not Sure"] if no clarity
"""

transcript_prompt ="""Generate a verbatim transcript of the provided audio. Ensure all spoken words are captured precisely, and include precise timestamps for each utterance. Transcribe and translate this multilingual audio into English:
 
Verbatim accuracy: Capture every spoken word exactly as said

Convert all languages: Translate everything to fluent English

Precise timestamps: Include [MM:SS] for each utterance and speaker change

Speaker labels: Use Speaker 1, Speaker 2, etc.

Natural flow: Make translations sound natural while preserving exact meaning

Context preservation: Maintain tone, emphasis, and cultural references

Unclear speech: Mark as [unclear] or [inaudible]

Non-verbal sounds: Note [laughter], [pause], [music], [coughing] in English

Complete capture: Include filler words, false starts, and repetitions

Formatting: Proper English punctuation and grammar
"""
CallAuditStr = Annotated[str, StringConstraints(strip_whitespace=True)]
ShortStr = Annotated[str, StringConstraints(max_length=200)]
NotesStr = Annotated[str, StringConstraints(max_length=150)]
PhoneNumber = Annotated[str, StringConstraints(pattern=r"^[6-9]\d{9}$")]

class CallAudit(BaseModel):
    is_parent_or_student: Literal["Parent", "Student", "Both", "Not Sure"]
    grade: Optional[Annotated[int, Field(ge=1, le=13)]] = None
    sibling: Literal["Yes", "No", "Not Sure"]
    target_exam: Literal["Jee", "Neet", "Coding", "English", "Foundation", "Not Sure"]
    is_father_there_on_call: Literal["Yes", "No", "Not Sure"]
    father_profession: str
    is_mother_working: Literal["Yes", "No", "Not Sure"]
    mother_profession: Optional[str] = None
    additional_class_student_going: Optional[Literal["Yes", "No", "Not Sure"]] = None
    current_tuition_provider: Optional[
        Literal[
            "Local", "Allen", "Aakash", "Fiitjee", "Narayana", "Sri Chaitanya",
            "Motion", "Physics Wallah", "Unacademy", "Byjus", "Vedantu",
            "Lead School", "Toppr", "Extramarks", "Not Sure"
        ]
    ] = None
    current_tuition_fees: Optional[int] = None
    intent: Literal["High", "Medium"]
    intent_description: Optional[ShortStr] = None
    price_expectation: Literal["Yes", "No"]
    pain_point: ShortStr
    demo_pointers: List[ShortStr] = Field(..., min_items=1, max_items=3)
    demo_confirmed_by_customer: Literal["Yes", "Not Sure", "No"]
    demo_date_and_time: Optional[str] = None
    notes: NotesStr
    knows_infinity_learn: Literal["Yes", "No", "Not Sure"]
    knows_sri_chaitanya: Literal["Yes", "No", "Not Sure"]
    knows_inmobius: Literal["Yes", "No", "Not Sure"]
    name: Optional[CallAuditStr] = None
    parent_name: Optional[CallAuditStr] = None
    alternative_phone: Optional[PhoneNumber] = None

    @field_validator("current_tuition_fees")
    @classmethod
    def check_positive_fee(cls, v):
        if v is not None and v <= 0:
            raise ValueError("current_tuition_fees must be a positive integer")
        return v

    @field_validator("demo_date_and_time", mode="before")
    @classmethod
    def ensure_datetime_format(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError("demo_date_and_time must be in 'YYYY-MM-DD HH:MM:SS' format")
        return v
