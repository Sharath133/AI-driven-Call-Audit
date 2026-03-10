# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp",
#     "google-genai",
#     "pydantic",
#     "pymongo",
#     "requests",
#     "pydantic-settings",
# ]
# ///


import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
import tempfile
import os
import uuid
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient, ASCENDING
from google import genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from prompt import prompt
from app.external.leadsquared_api import LeadSquared


load_dotenv()

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# --- Environment Variable Check ---
# Ensure all required environment variables are set
required_vars = [
    "GOOGLE_GEMINI_API_KEY",
    "MONGO_URI",
    "DB_NAME",
    "COLLECTION_NAME",
    "AUDIT_COLLECTION_NAME",
    "FETCH_SINCE_HOURS",
    "JOB_LOG_COLLECTION_NAME",
]
for var in required_vars:
    if var not in os.environ:
        raise ValueError(f"Missing required environment variable: {var}")


# --- Setup clients ---
genai_client = genai.Client(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
leadsquared_client = LeadSquared()
mongo_client = MongoClient(os.environ["MONGO_URI"])
db = mongo_client[os.environ["DB_NAME"]]
source_collection = db[os.environ["COLLECTION_NAME"]]
audit_collection = db[os.environ["AUDIT_COLLECTION_NAME"]]  # New audit collection
job_log_collection = db[os.environ["JOB_LOG_COLLECTION_NAME"]]

# --- Create Index ---
audit_collection.create_index([("recording_url", ASCENDING)], unique=True)
logging.info(
    f"Ensured index exists on 'recording_url' in '{os.environ['AUDIT_COLLECTION_NAME']}' collection."
)

job_id = str(uuid.uuid4())
job_name = "demo_booking_analysis"
started_at = datetime.now(timezone.utc)
logs = []
status = "success"
failure_reason = None

total_calls_found = 0
already_processed_count = 0
new_calls_to_process = 0
successfully_processed = 0


def fetch_recent_audio_urls() -> List[Dict[str, str]]:
    """Fetches call activity documents from the source collection."""
    since = datetime.now(timezone.utc) - timedelta(
        hours=int(os.environ["FETCH_SINCE_HOURS"])
    )
    logging.info(
        f"Fetching source documents from the last {os.environ['FETCH_SINCE_HOURS']} hours..."
    )
    docs = source_collection.find(
        {"CreatedOnDate": {"$gte": since}},
        {
            "ProspectActivityId": 1,
            "RelatedProspectId": 1,
            "related_calls.mx_Custom_4": 1,
            "related_calls.CreatedOn": 1,
        },
    )
    urls = []
    for doc in docs:
        lead_id = doc.get("RelatedProspectId")
        related_calls = doc.get("related_calls", [])
        for call in related_calls:
            url = call.get("mx_Custom_4")
            activity_datetime = call.get("CreatedOn")
            if url:
                urls.append(
                    {
                        "lead_id": lead_id,
                        "recording_url": url,
                        "activity_datetime": activity_datetime,
                    }
                )
    return urls


def filter_unprocessed_urls(
    call_activities: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Filters out URLs that have already been successfully processed in the last 24 hours."""
    if not call_activities:
        return []

    urls_to_check = [call["recording_url"] for call in call_activities]

    # Define the time threshold — 24 hours ago from now
    twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(
        hours=int(os.environ["FETCH_SINCE_HOURS"])
    )

    # Fetch only documents with status=success and processed_at >= 24 hours ago
    processed_docs = audit_collection.find(
        {
            "recording_url": {"$in": urls_to_check},
            "status": "success",
            "processed_at": {"$gte": twenty_four_hours_ago},
        },
        {"recording_url": 1},  # Only need the URL
    )

    processed_urls = {doc["recording_url"] for doc in processed_docs}
    logging.info(
        f"Found {len(processed_urls)} successfully processed URLs in the last 24 hours."
    )

    unprocessed_calls = [
        call for call in call_activities if call["recording_url"] not in processed_urls
    ]
    logging.info(f"Found {len(unprocessed_calls)} new URLs to process.")
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
                logging.info(f"Downloaded audio from {call_info['recording_url']}")
                return temp.name, call_info
    except Exception as e:
        logging.error(f"Download error for {call_info['recording_url']}: {e}")
        return None


def process_audio_file(filepath: str, call_info: Dict[str, str]):
    """
    Processes a single audio file: gets AI analysis, posts to CRM, and logs the result.
    """
    try:
        logging.info(
            f"Processing audio file: {filepath} for lead: {call_info['lead_id']}"
        )
        file = genai_client.files.upload(file=filepath)
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt, file],
            config={
                "response_mime_type": "application/json",
            },
        )
        logging.info(f"Processing complete for: {filepath}")

        call_audit = json.loads(response.text)

        lead_id = call_info["lead_id"]
        activity_time = call_info["activity_datetime"]

        attributes = {
            "mx_Custom_2": call_audit.get("parent_name"),
            "mx_Custom_3": call_audit.get("alternative_phone"),
            "mx_Custom_4": call_audit.get("grade"),
            "mx_Custom_5": call_audit.get("target_exam"),
            "mx_Custom_6": call_audit.get("sibling"),
            "mx_Custom_7": call_audit.get("mother_profession"),
            "mx_Custom_8": call_audit.get("father_profession"),
            "mx_Custom_9": call_audit.get("is_father_there_on_call"),
            "mx_Custom_11": call_audit.get("demo_confirmed_by_customer"),
            "mx_Custom_12": call_audit.get("knows_sri_chaitanya"),
            "mx_Custom_13": call_audit.get("knows_infinity_learn"),
            "mx_Custom_14": call_audit.get("knows_inmobius"),
            "mx_Custom_15": call_audit.get("intent"),
            "mx_Custom_16": call_audit.get("intent_description"),
            "mx_Custom_17": call_audit.get("pain_point"),
            "mx_Custom_18": ",".join(call_audit.get("demo_pointers", []))[:200],
            "mx_Custom_19": call_audit.get("name"),
            "mx_Custom_20": call_audit.get("price_expectation"),
            "mx_Custom_21": call_audit.get("is_mother_working"),
        }

        lsq_response = leadsquared_client.create_activity_for_lead(
            id=lead_id,
            activity_event_code=348,
            activity_time=activity_time,
            attr=attributes,
        )
        lsq_response.raise_for_status()  # Raise an exception for non-2xx responses
        logging.info(f"Successfully created LeadSquared activity for lead {lead_id}.")
        logging.debug(
            {"lsq_response": lsq_response.json(), "activity_attributes": attributes}
        )

        # --- Log successful processing to audit collection ---
        audit_log = {
            "lead_id": lead_id,
            "activity_datetime": activity_time,
            "ai_audit": call_audit,
            "status": "success",
            "error_message":None,
            "processed_at": datetime.now(timezone.utc),
        }
        audit_collection.update_one(
            {"recording_url": call_info["recording_url"]},
            {"$set": audit_log},
            upsert=True,
        )
        logging.info(f"Successfully logged audit for {call_info['recording_url']}")

    except Exception as e:
        logging.error(
            f"Processing error for {filepath} (Lead ID: {call_info['lead_id']}): {e}"
        )
        # --- Log failed processing to audit collection to prevent retries ---
        error_log = {
            "recording_url": call_info["recording_url"],
            "lead_id": call_info.get("lead_id"),
            "activity_datetime": call_info.get("activity_datetime"),
            "status": "error",
            "error_message": str(e),
            "processed_at": datetime.now(timezone.utc),
        }
        audit_collection.update_one(
            {"recording_url": call_info["recording_url"]},
            {"$set": error_log},
            upsert=True,
        )
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.debug(f"Deleted temp file: {filepath}")


async def process_all_audio_files(
    call_activities: List[Dict[str, str]], concurrency: int = 3
):
    """Downloads and processes a list of audio files concurrently."""
    async with aiohttp.ClientSession() as session:
        download_tasks = [
            download_audio(session, call_info) for call_info in call_activities
        ]
        downloaded = await asyncio.gather(*download_tasks)
        # Filter out any failed downloads
        downloaded_files = [d for d in downloaded if d is not None]

    logging.info(
        f"Starting processing of {len(downloaded_files)} downloaded audio files..."
    )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(process_audio_file, filepath, call_info)
            for filepath, call_info in downloaded_files
        ]
        for future in futures:
            future.result()

    logging.info("All files processed.")


if __name__ == "__main__":
    try:
        # 1. Fetch all recent call records
        all_recent_calls = fetch_recent_audio_urls()
        total_calls_found = len(all_recent_calls)

        if all_recent_calls:
            # 2. Filter out ones that have already been processed
            unprocessed_calls = filter_unprocessed_urls(all_recent_calls)
            already_processed_count = total_calls_found - len(unprocessed_calls)
            new_calls_to_process = len(unprocessed_calls)

            if unprocessed_calls:
                # 3. Process the new, unprocessed calls
                asyncio.run(process_all_audio_files(unprocessed_calls, concurrency=3))
                successfully_processed = new_calls_to_process
            else:
                logging.info("No new audio URLs to process at this time.")
        else:
            logging.warning(
                f"No audio URLs found in the last {os.environ['FETCH_SINCE_HOURS']} hours."
            )
    except Exception as e:
        failure_reason = str(e)
        status = "failed"
        logging.critical(
            f"An unexpected error occurred in the main pipeline: {e}", exc_info=True
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
        job_log_collection.insert_one(job_log_doc)
