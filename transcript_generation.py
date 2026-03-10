# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp",
#     "google-generativeai",
#     "psycopg2-binary",
#     "pydantic-settings",
#     "google-auth",
#     "google-api-python-client",
#     "python-dotenv",
#     "requests",
#     "pymongo",
# ]
# ///

import asyncio
import logging
import tempfile
import os,gc
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
from datetime import datetime, timedelta, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from google.oauth2 import service_account
from googleapiclient.discovery import build
import time
import random
import uuid
from prompt import transcript_prompt
from app.external.leadsquared_api import LeadSquared
from pymongo import MongoClient, ASCENDING

import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings and create unverified context
urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Alternative: Create a more permissive SSL context
def create_ssl_context():
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

load_dotenv(dotenv_path="gen-ai/demo-booking-analysis/.env")
# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# --- Environment Variable Check ---
required_vars = [
    "GOOGLE_GEMINI_API_KEY",
    "GOOGLE_CREDENTIALS_PATH",
    "ORGANIZER_EMAILS",
    "TRANSCRIPT_POSTGRES_HOST",
    "TRANSCRIPT_POSTGRES_PORT",
    "TRANSCRIPT_POSTGRES_DB",
    "TRANSCRIPT_POSTGRES_USER",
    "TRANSCRIPT_POSTGRES_PASSWORD",
    "TRANSCRIPT_SOURCE_TABLE",        # Table name for source data
    "MONGO_URI",
    "DB_NAME",
    "JOB_LOG_COLLECTION_NAME",
    "TRANSCRIPT_COLLECTION_NAME"
]
for var in required_vars:
    if var not in os.environ:
        raise ValueError(f"Missing required environment variable: {var}")

genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
print("Available Gemini Models:")
print("=" * 60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Description: {model.description}")
        print(f"Supported Methods: {model.supported_generation_methods}")
        print("-" * 60)

# --- Setup LeadSquared client ---
leadsquared_client = LeadSquared()

# --- Setup MongoDB ---
mongo_client = MongoClient(os.environ["MONGO_URI"])
db = mongo_client[os.environ["DB_NAME"]]
transcript_collection = db[os.environ["TRANSCRIPT_COLLECTION_NAME"]]
job_log_collection = db[os.environ["JOB_LOG_COLLECTION_NAME"]]

# Create unique index on recording_url
transcript_collection.create_index([("recording_url", ASCENDING)], unique=True)
logging.info("Ensured index exists on 'recording_url' in call-transcripts collection.")

job_id = str(uuid.uuid4())
job_name = "call_transcript_generation"
started_at = datetime.now(timezone.utc)
logs = []
status = "success"
failure_reason = None

total_calls_found = 0
already_processed_count = 0
new_calls_to_process = 0
successfully_processed = 0

# --- PostgreSQL Connection ---
def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.environ["TRANSCRIPT_POSTGRES_HOST"],
        port=os.environ["TRANSCRIPT_POSTGRES_PORT"],
        database=os.environ["TRANSCRIPT_POSTGRES_DB"],
        user=os.environ["TRANSCRIPT_POSTGRES_USER"],
        password=os.environ["TRANSCRIPT_POSTGRES_PASSWORD"],
        connect_timeout=10
    )

class GoogleDocsService:
    """Service class to handle Google Docs operations with folder organization."""

    def __init__(self, organizer_emails: List[str]):
        """Initialize Google Docs service with delegated authentication."""
        self.credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
        if not self.credentials_path:
            raise ValueError("Google service account credentials path is required")

        self.services = self._initialize_services(organizer_emails)

    def _initialize_services(self, organizer_emails) -> List[Tuple[str, Any, Any]]:
        """Initialize Google Docs and Drive services."""
        SCOPES = [
            "https://www.googleapis.com/auth/documents",    # Create/edit docs
            "https://www.googleapis.com/auth/drive.file",   # Create files & manage folders
        ]

        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES
        )

        services = []
        for email in organizer_emails:
            try:
                delegated_creds = credentials.with_subject(email)

                docs_service = build("docs", "v1", credentials=delegated_creds)
                drive_service = build("drive", "v3", credentials=delegated_creds)
                services.append((email, docs_service, drive_service))
                logging.info(f"Successfully initialized Google Docs and Drive services for {email}")
            except Exception as e:
                logging.error(f"Failed to initialize services for {email}: {e}")

        return services

    def find_folder(self, drive_service, folder_name, parent_folder_id='root'):
        try:
            # Search for existing folder
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_folder_id}' in parents and trashed=false"
            results = drive_service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()

            folders = results.get('files', [])

            if folders:
                logging.info(f"Found existing folder '{folder_name}': {folders[0]['id']}")
                return folders[0]['id']
            else:
                # If not found in root, search in all accessible folders
                query_all = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
                results_all = drive_service.files().list(
                    q=query_all,
                    fields="files(id, name, parents)"
                ).execute()

                folders_all = results_all.get('files', [])
                if folders_all:
                    logging.info(f"Found existing folder '{folder_name}' in Drive: {folders_all[0]['id']}")
                    return folders_all[0]['id']
                else:
                    logging.error(f"Folder '{folder_name}' not found in Google Drive")
                    return None

        except Exception as e:
            logging.error(f"Failed to find folder '{folder_name}': {e}")
            return None


    def create_transcript_document(self, activity_id: str, prospect_id: str, start_time: str,
                        end_time: str, duration: str, transcript_content: str) -> Optional[str]:
        """Create a new Google Doc with transcript content in the Call Transcripts folder."""
    #     if not self.services:
    #         logging.error("No Google Docs services available")
    #         return None

    #     # Use the first available service
    #     email, docs_service, drive_service = self.services[0]

    #     max_retries = 3
    #     base_delay = 2

    #     for attempt in range(max_retries):
    #         try:
    #             # Use the known folder ID directly
    #             folder_id = "1S50d5MMDG2Ix-V9gnkZ1vNE3ll0xYdOf"  # Call Transcripts folder

    #             # Document title using Activity ID
    #             document_title = f"Activity_{activity_id}"

    #             # Create document
    #             doc = docs_service.documents().create(body={'title': document_title}).execute()
    #             doc_id = doc.get('documentId')

    #             # Move document to folder
    #             try:
    #                 # drive_service.files().update(
    #                 #     fileId=doc_id,
    #                 #     addParents=folder_id,
    #                 #     removeParents='root',
    #                 #     fields='id, parents'
    #                 # ).execute()
    #             except Exception as move_error:
    #                 logging.warning(f"Failed to move document to folder (continuing anyway): {move_error}")

    #             # Format the document content
    #             formatted_content = f"""Activity ID: {activity_id}
    # Prospect ID: {prospect_id}
    # Start Time: {start_time}
    # End Time: {end_time}
    # Duration: {duration}

    # TRANSCRIPT:
    # =====================================

    # {transcript_content}"""

    #             # Add content to document
    #             requests = [{
    #                 'insertText': {
    #                     'location': {'index': 1},
    #                     'text': formatted_content
    #                 }
    #             }]

    #             docs_service.documents().batchUpdate(
    #                 documentId=doc_id,
    #                 body={'requests': requests}
    #             ).execute()

    #             # Return document URL
    #             doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
    #             return doc_url

    #         except Exception as e:
    #             if any(keyword in str(e).lower() for keyword in ['ssl', 'timeout']) and attempt < max_retries - 1:
    #                 delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
    #                 logging.warning(f"Retrying in {delay:.2f}s: {e}")
    #                 time.sleep(delay)
    #                 continue
    #             else:
    #                 logging.error(f"Failed to create document: {e}")
    #                 break

        return None

def generate_transcript_with_gemini(filepath: str) -> Optional[str]:
    """Generate transcript from audio file using Gemini."""
    try:
        uploaded_file = genai.upload_file(filepath)

        # Generate transcript using Gemini Flash (faster and cheaper for transcription)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content([transcript_prompt, uploaded_file])

        transcript_text = response.text.strip()

        return transcript_text

    except Exception as e:
        logging.error(f"Failed to generate transcript for {filepath}: {e}")
        return None
    finally:
        # Clean up uploaded file to free native memory
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception as cleanup_error:
                logging.warning(f"Failed to cleanup uploaded file: {cleanup_error}")

def calculate_call_duration(duration_seconds) -> str:
    """Calculate duration string from seconds."""
    try:
        if not duration_seconds or duration_seconds == 0:
            return "N/A"

        total_minutes = int(duration_seconds) // 60
        seconds = int(duration_seconds) % 60

        hours = total_minutes // 60
        minutes = total_minutes % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"

    except Exception as e:
        logging.error(f"Error calculating duration: {e}")
        return "N/A"

def calculate_end_time(start_time_str: str, duration_seconds: int) -> str:
    """Calculate end time by adding duration to start time."""
    try:
        # Parse the start time
        start_dt = datetime.strptime(start_time_str.split('.')[0], "%Y-%m-%d %H:%M:%S")

        # Add duration to get end time
        end_dt = start_dt + timedelta(seconds=duration_seconds)

        # Return in same format as start time
        return end_dt.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        logging.error(f"Error calculating end time: {e}")
        return "N/A"

def fetch_calls_for_transcript() -> List[Dict[str, str]]:
    """
    Fetches call records from PostgreSQL database that need transcript generation.
    Returns a list of call dictionaries with all required information.
    """
    calls = []
    conn = None

    try:
        # Calculate 24 hours ago in UTC
        twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
        logging.info(f"Fetching calls from last 24 hours (since {twenty_four_hours_ago})")

        # Get database connection
        conn = get_db_connection()
        logging.info("Successfully connected to PostgreSQL database")

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Query to get call records with specific criteria
            cursor.execute(f"""
                SELECT
                    "RelatedProspectActivityId" as activity_id,
                    "RelatedProspectId" as prospect_id,
                    "ActivityEvent" as activity_event,
                    "mx_Custom_4" as recording_url,
                    "mx_Custom_2" as start_time,
                    "mx_Custom_3" as duration_seconds,
                    "mx_Custom_7" as call_status,
                    "ModifiedOn" as activity_datetime,
                    "CreatedOn" as created_on
                FROM {os.environ["TRANSCRIPT_SOURCE_TABLE"]}
                WHERE "ActivityEvent" IN ('21', '22')  -- Both inbound (21) and outbound (22) calls
                    AND "mx_Custom_7" = 'Answered'  -- Status Answered
                    AND "mx_Custom_4" IS NOT NULL  -- Recording Link exists
                    AND "mx_Custom_4" != ''
                    AND "mx_Custom_3" IS NOT NULL  -- Duration exists
                    AND "mx_Custom_3" != ''
                    AND "CreatedOn" >= %s -- Created in last 24 hours
            """, (twenty_four_hours_ago,))

            rows = cursor.fetchall()
            logging.info(f"Retrieved {len(rows)} call records from database")

            for row in rows:
                try:
                    # Convert row to dictionary
                    call_data = dict(row)

                    # Validate and convert duration
                    try:
                        duration_seconds = int(call_data.get('duration_seconds', 0))
                        if duration_seconds <= 240:
                            continue
                    except (ValueError, TypeError):
                        logging.warning(f"Skipping call {call_data.get('activity_id')} with invalid duration: {call_data.get('duration_seconds')}")
                        continue

                    # Simple string extraction for start_time (format: 2025-04-19 14:02:18)
                    start_time = str(call_data.get('start_time', 'N/A'))

                    # Calculate end time
                    end_time = calculate_end_time(start_time, duration_seconds)

                    # Validate required fields
                    if not call_data.get('recording_url') or not call_data.get('activity_id'):
                        logging.warning(f"Missing recording_url or activity_id for call, skipping")
                        continue

                    # Add to calls list with all required information
                    calls.append({
                        "activity_id": call_data['activity_id'],
                        "prospect_id": call_data['prospect_id'],
                        "activity_event": call_data['activity_event'],
                        "start_time": start_time,
                        "duration_seconds": duration_seconds,
                        "duration_display": calculate_call_duration(duration_seconds),
                        "end_time": end_time,
                        "recording_url": call_data['recording_url'],
                        "activity_datetime": call_data['activity_datetime'],
                        "call_status": call_data.get('call_status', 'Answered'),
                    })

                except Exception as row_error:
                    logging.error(f"Error processing row {call_data.get('activity_id', 'unknown')}: {row_error}")
                    continue

            logging.info(f"Successfully processed {len(calls)} valid calls for transcript generation")

    except psycopg2.Error as db_error:
        logging.error(f"Database error while fetching calls: {db_error}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while fetching calls from PostgreSQL: {e}")
        return []
    finally:
        if conn:
            try:
                conn.close()
                logging.info("Database connection closed successfully")
            except Exception as close_error:
                logging.error(f"Error closing database connection: {close_error}")

    return calls

def filter_unprocessed_calls(call_activities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filters out calls that have already been successfully processed in the last 24 hours."""
    if not call_activities:
        return []

    urls_to_check = [call["recording_url"] for call in call_activities]
    twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=96)

    # Check for successfully processed transcripts
    processed_docs = transcript_collection.find(
        {
            "recording_url": {"$in": urls_to_check},
            "status": "success",
            "processed_at": {"$gte": twenty_four_hours_ago},
        },
        {"recording_url": 1}
    )

    processed_urls = {doc["recording_url"] for doc in processed_docs}

    unprocessed_calls = [
        call for call in call_activities if call["recording_url"] not in processed_urls
    ]
    return unprocessed_calls

async def download_audio(
    session: aiohttp.ClientSession, call_info: Dict[str, str]
) -> Optional[Tuple[str, Dict[str, str]]]:
    """Downloads an audio file from a URL to a temporary local file."""
    try:
        async with session.get(call_info["recording_url"]) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                temp.write(await resp.read())
                return temp.name, call_info
    except Exception as e:
        logging.error(f"Download error for {call_info['recording_url']}: {e}")
        return None


def process_transcript_file(filepath: str, call_info: Dict[str, str]):
    """
    Processes a single audio file for transcript generation: generates transcript, creates Google Doc, and sends to LeadSquared.
    """
    breakpoint()
    try:
        activity_event = call_info.get('activity_event', '22')
        call_type = "Inbound" if activity_event == '21' else "Outbound"

        # Generate transcript
        transcript_text = generate_transcript_with_gemini(filepath)

        if not transcript_text:
            raise Exception("Failed to generate transcript")

        prospect_id = f"https://in21.leadsquared.com/LeadManagement/LeadDetails?LeadID={call_info['prospect_id']}"

        # Try to create Google Doc with comprehensive error handling
        transcript_doc_url = None
        max_doc_attempts = 2

        for doc_attempt in range(max_doc_attempts):
            try:
                transcript_doc_url = docs_service.create_transcript_document(
                    activity_id=call_info['activity_id'],
                    prospect_id=prospect_id,
                    start_time=call_info.get('start_time', 'N/A'),
                    end_time=call_info.get('end_time', 'N/A'),
                    duration=call_info.get('duration_display', 'N/A'),
                    transcript_content=transcript_text
                )
                if transcript_doc_url:
                    break

            except Exception as doc_error:
                logging.error(f"Google Doc creation attempt {doc_attempt + 1} failed: {doc_error}")
                if doc_attempt < max_doc_attempts - 1:
                    time.sleep(3)  # Wait before retry

        # Create fallback URL if all Google Doc attempts fail
        if not transcript_doc_url:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_doc_url = f"TRANSCRIPT_SSL_ERROR_{call_info['activity_id']}_{timestamp}_LEN_{len(transcript_text)}"
            logging.warning(f"Using fallback URL due to SSL errors: {transcript_doc_url}")

        # Update LeadSquared activity (this should work even if Google Doc fails)
        activity_id = call_info["activity_id"]
        call_duration = str(call_info.get('duration_seconds', 0))  # Required field
        mx_Custom_11= transcript_doc_url

        try:
            lsq_response = leadsquared_client.update_activity(
                activity_id=activity_id,
                call_duration= call_duration,
                mx_Custom_11= mx_Custom_11
            )
            lsq_response.raise_for_status()
            # --- Store successful transcript record ---
            transcript_record = {
                "recording_url": call_info["recording_url"],
                "activity_datetime": call_info.get('activity_datetime'),
                "prospect_id": call_info["prospect_id"],
                "status": "success",
                "transcript_doc_url": transcript_doc_url,
                "processed_at": datetime.now(timezone.utc),
                "error_message": None
            }
            transcript_collection.update_one(
                {"recording_url": call_info["recording_url"]},
                {"$set": transcript_record},
                upsert=True
            )
            logging.info(f"SUCCESS: Complete transcript processing finished for Activity: {activity_id} - Prospect: {call_info['prospect_id']}")
        except Exception as lsq_error:
            logging.error(f"Failed to update LeadSquared activity: {lsq_error}")
            raise lsq_error

    except Exception as e:
        logging.error(f"Transcript processing error for {filepath} (Activity: {call_info['activity_id']}): {e}")

        error_record = {
            "recording_url": call_info["recording_url"],
            "activity_datetime": call_info.get('activity_datetime'),
            "prospect_id": call_info.get("prospect_id"),
            "status": "error",
            "transcript_doc_url": None,
            "processed_at": datetime.now(timezone.utc),
            "error_message": str(e)
        }
        transcript_collection.update_one(
            {"recording_url": call_info["recording_url"]},
            {"$set": error_record},
            upsert=True
        )
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.debug(f"Deleted temp file: {filepath}")
    gc.collect()

async def process_all_transcript_files(
    call_activities: List[Dict[str, str]], concurrency: int = 2
):
    """Downloads and processes a list of audio files for transcript generation."""
    async with aiohttp.ClientSession() as session:
        download_tasks = [
            download_audio(session, call_info) for call_info in call_activities
        ]
        downloaded = await asyncio.gather(*download_tasks)
        # Filter out any failed downloads
        downloaded_files = [d for d in downloaded if d is not None]

    logging.info(
        f"Starting transcript processing of {len(downloaded_files)} downloaded audio files..."
    )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(process_transcript_file, filepath, call_info)
            for filepath, call_info in downloaded_files
        ]
        for future in futures:
            future.result()

    logging.info("All transcript files processed.")

# Initialize GoogleDocsService
organizer_emails = os.environ.get("ORGANIZER_EMAILS", "").split(",")
organizer_emails = [email.strip() for email in organizer_emails if email.strip()]

if not organizer_emails:
    raise ValueError("ORGANIZER_EMAILS environment variable is required")

docs_service = GoogleDocsService(organizer_emails)

if __name__ == "__main__":
    try:
        all_calls_for_transcript = fetch_calls_for_transcript()
        total_calls_found = len(all_calls_for_transcript)
        logging.info(f"COMPLETED: Found {total_calls_found} total calls")

        if all_calls_for_transcript:
            unprocessed_calls = filter_unprocessed_calls(all_calls_for_transcript)
            already_processed_count = total_calls_found - len(unprocessed_calls)
            new_calls_to_process = len(unprocessed_calls)
            logging.info(f"COMPLETED: Found {new_calls_to_process} unprocessed calls")

            if unprocessed_calls:
                batch_size = 200
                for i in range(0, len(unprocessed_calls), batch_size):
                    batch = unprocessed_calls[i:i + batch_size]
                    logging.info(f"Processing batch {i//batch_size + 1}: {len(batch)} calls")
                    asyncio.run(process_all_transcript_files(batch, concurrency=2))

                    # Force cleanup between batches
                    gc.collect()
                    time.sleep(1)

                successfully_processed = new_calls_to_process
            else:
                logging.info("No new calls to process for transcripts.")
        else:
            logging.warning("No answered calls with recordings found.")

    except Exception as e:
        failure_reason = str(e)
        status = "failed"
        logging.critical(
            f"An unexpected error occurred in the transcript generation pipeline: {e}", exc_info=True
        )
    finally:
        completed_at = datetime.now(timezone.utc)
        job_log_doc = {
            "_id": job_id,
            "job_name": job_name,
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": completed_at.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "logs": "; ".join(logs) if logs else "Job completed.",
            "total_calls_found": total_calls_found,
            "already_processed_count": already_processed_count,
            "new_calls_to_process": new_calls_to_process,
            "successfully_processed": successfully_processed,
            "failure_reason": failure_reason,
        }

        try:
            job_log_collection.insert_one(job_log_doc)
            logging.info(f"Job log saved with ID: {job_id}")
        except Exception as log_error:
            logging.error(f"Failed to save job log: {log_error}")