import os
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional # Dict is imported here
import requests
import json

import uuid
import time
from datetime import datetime

app = FastAPI(title="Meeting Minutes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://meeting-minutes-frontend.vercel.app"], # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (would use a database in production)
transcripts = {}
summaries = {}

# Hugging Face API endpoints
HF_ASR_API = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
HF_LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

class TranscriptRequest(BaseModel):
    audio_url: str

class SummaryRequest(BaseModel):
    transcript_id: str

class Speaker(BaseModel):
    id: str
    name: str

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str

class SummarySection(BaseModel):
    title: str
    blocks: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"message": "Meeting Minutes API is running"}

@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Process audio file and return transcript ID"""
    # Generate a unique ID for this transcript
    transcript_id = str(uuid.uuid4())

    # Save the file temporarily
    file_location = f"/tmp/{transcript_id}.{file.filename.split('.')[-1]}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process in background
    if background_tasks:
        background_tasks.add_task(process_audio_file, file_location, transcript_id)
    else:
        # For testing, process synchronously
        await process_audio_file(file_location, transcript_id)

    return {"transcript_id": transcript_id, "status": "processing"}

async def process_audio_file(file_path: str, transcript_id: str):
    """Process audio file using Hugging Face Whisper API"""
    try:
        # Read the audio file
        with open(file_path, "rb") as f:
            data = f.read()

        # === Sanity checks and logging ===  <- This block replaces the old headers/requests.post
        print(f"[whisper] HF_API_TOKEN set: {bool(HF_API_TOKEN)}")

        headers: Dict[str, str] = {}
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
            print("[whisper] Using Bearer auth on Whisper endpoint")
        else:
            print("[whisper] ⚠️ No HF_API_TOKEN provided; calling without auth")

        # Ensure correct content type for audio (adjust if not WAV)
        # Assuming WAV based on common usage, adjust header if your audio format differs
        headers["Content-Type"] = f"audio/{file_path.split('.')[-1]}" # Dynamically set content-type based on extension
        print(f"[whisper] Setting Content-Type: {headers['Content-Type']}")


        response = requests.post(
            HF_ASR_API,
            headers=headers,
            data=data
        )

        # Log status + first 200 chars of response body
        print(f"[whisper] status={response.status_code} body={response.text[:200]}…")
        # === End of replaced block ===


        if response.status_code != 200:
            error_detail = f"Transcription failed: Status {response.status_code}, Body: {response.text}"
            print(f"[whisper] Error: {error_detail}") # Log the full error
            transcripts[transcript_id] = {
                "error": error_detail,
                "status": "failed"
            }
            # Clean up temp file even on failure
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"[whisper] Error removing temp file {file_path}: {e}")
            return

        result = response.json()

        # Simulate speaker diarization (in a real app, we would use a proper diarization model)
        segments = []
        speakers = []

        # Simple rule-based speaker assignment for demonstration
        text = result.get("text", "")
        sentences = text.split(". ")

        current_time = 0
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Assign a speaker based on simple pattern
            speaker_id = f"speaker_{i % 3}"  # Rotate between 3 speakers

            # Add speaker if not already in the list
            if speaker_id not in [s["id"] for s in speakers]:
                speakers.append({
                    "id": speaker_id,
                    "name": f"Speaker {i % 3 + 1}"
                })

            # Create segment
            duration = len(sentence.split()) * 0.3  # Rough estimate: 0.3 seconds per word
            segments.append({
                "start": current_time,
                "end": current_time + duration,
                "text": sentence.strip(),
                "speaker": speaker_id
            })
            current_time += duration

        # Store the transcript
        transcripts[transcript_id] = {
            "segments": segments,
            "speakers": speakers,
            "status": "completed",
            "created_at": datetime.now().isoformat()
        }

        # Clean up
        os.remove(file_path)

    except Exception as e:
        print(f"[whisper] Exception during processing {transcript_id}: {e}") # Log exceptions
        transcripts[transcript_id] = {
            "error": f"An unexpected error occurred: {str(e)}",
            "status": "failed"
        }
        # Attempt cleanup even on general exception
        try:
            if os.path.exists(file_path):
                 os.remove(file_path)
        except OSError as e_rm:
            print(f"[whisper] Error removing temp file {file_path} after exception: {e_rm}")


@app.get("/api/transcript/{transcript_id}")
def get_transcript(transcript_id: str):
    """Get transcript by ID"""
    if transcript_id not in transcripts:
        raise HTTPException(status_code=404, detail="Transcript not found")

    return transcripts[transcript_id]

@app.post("/api/generate-summary")
async def generate_summary(request: SummaryRequest, background_tasks: BackgroundTasks = None):
    """Generate meeting minutes from transcript"""
    if request.transcript_id not in transcripts:
        raise HTTPException(status_code=404, detail="Transcript not found")

    transcript = transcripts[request.transcript_id]
    if transcript.get("status") != "completed":
         # Check for error state as well
        if transcript.get("status") == "failed":
             raise HTTPException(status_code=400, detail=f"Transcript processing failed: {transcript.get('error', 'Unknown error')}")
        raise HTTPException(status_code=400, detail="Transcript processing not completed or status unknown")


    # Generate a unique ID for this summary
    summary_id = str(uuid.uuid4())

    # Process in background
    if background_tasks:
        background_tasks.add_task(generate_meeting_minutes, request.transcript_id, summary_id)
    else:
        # For testing, process synchronously
        await generate_meeting_minutes(request.transcript_id, summary_id)

    return {"summary_id": summary_id, "status": "processing"}

async def generate_meeting_minutes(transcript_id: str, summary_id: str):
    """Generate meeting minutes using Hugging Face LLM API"""
    try:
        transcript = transcripts[transcript_id]

        # Prepare the transcript text
        transcript_text = ""
        for segment in transcript["segments"]:
            speaker_name = next((s["name"] for s in transcript["speakers"] if s["id"] == segment["speaker"]), "Unknown")
            transcript_text += f"{speaker_name}: {segment['text']}\n"

        # Prepare the prompt for the LLM
        prompt = f"""
        You are a meeting summarizer. Your task is to analyze the following meeting transcript and create structured meeting minutes with these sections:
        1. Summary: A concise summary of the meeting in bullet points.
        2. Actions: List of actions assigned during the meeting, formatted as bullet points.
        3. Decisions: Bullet points outlining any decisions made during the meeting.
        4. Detailed Notes: An expanded version of the summary in bullet points, providing granular detail.

        Format your response as JSON with the following structure:
        {{
            "Summary": {{
                "blocks": [
                    {{"content": "Point 1"}},
                    {{"content": "Point 2"}},
                    ...
                ]
            }},
            "Actions": {{
                "blocks": [
                    {{"content": "Action 1"}},
                    {{"content": "Action 2"}},
                    ...
                ]
            }},
            "Decisions": {{
                "blocks": [
                    {{"content": "Decision 1"}},
                    {{"content": "Decision 2"}},
                    ...
                ]
            }},
            "DetailedNotes": {{
                "blocks": [
                    {{"content": "Note 1"}},
                    {{"content": "Note 2"}},
                    ...
                ]
            }}
        }}

        Here is the meeting transcript:
        {transcript_text}
        """

        # === LLM API Call ===
        print(f"[llm] HF_API_TOKEN set: {bool(HF_API_TOKEN)}")
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        if HF_API_TOKEN:
             print("[llm] Using Bearer auth on LLM endpoint")
        else:
             print("[llm] ⚠️ No HF_API_TOKEN provided; calling LLM without auth")


        response = requests.post(
            HF_LLM_API,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 1024, "return_full_text": False}}
        )
        print(f"[llm] status={response.status_code} body={response.text[:200]}…") # Log LLM response


        if response.status_code != 200:
            error_detail = f"Summary generation failed: Status {response.status_code}, Body: {response.text}"
            print(f"[llm] Error: {error_detail}") # Log the full error
            summaries[summary_id] = {
                "error": error_detail,
                "status": "failed"
            }
            return

        result = response.json()

        # Extract the JSON from the response
        text_response = ""
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
             text_response = result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
             text_response = result["generated_text"]
        else:
             print(f"[llm] Unexpected LLM response structure: {result}")
             text_response = str(result) # Fallback


        # Find JSON in the response
        json_start = text_response.find("{")
        json_end = text_response.rfind("}") + 1
        summary_data = None

        if json_start != -1 and json_end != 0 and json_start < json_end:
            json_str = text_response[json_start:json_end]
            try:
                summary_data = json.loads(json_str)
                # Basic validation of expected structure
                if not all(k in summary_data for k in ["Summary", "Actions", "Decisions", "DetailedNotes"]):
                     print(f"[llm] Warning: Parsed JSON missing expected keys. Content: {json_str[:200]}...")
                     summary_data = None # Reset if structure is wrong
            except json.JSONDecodeError as json_e:
                print(f"[llm] JSONDecodeError: {json_e}. Content: {json_str[:200]}...")
                summary_data = None # Failed parsing

        if summary_data is None:
             # If no JSON found or parsing failed, create a structured response manually
            print("[llm] Could not parse valid JSON summary, creating fallback structure.")
            summary_data = {
                "Summary": {
                    "blocks": [{"content": "Meeting summary could not be generated or parsed in the expected JSON format."}]
                },
                "Actions": {
                    "blocks": [{"content": "No actions could be extracted."}]
                },
                "Decisions": {
                    "blocks": [{"content": "No decisions could be extracted."}]
                },
                "DetailedNotes": {
                    "blocks": [{"content": f"Raw LLM response: {text_response}"}] # Include raw response for debugging
                }
            }


        # Store the summary
        summaries[summary_id] = {
            **summary_data,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "transcript_id": transcript_id
        }

    except Exception as e:
        print(f"[llm] Exception during summary generation {summary_id}: {e}") # Log exceptions
        summaries[summary_id] = {
            "error": f"An unexpected error occurred during summary generation: {str(e)}",
            "status": "failed"
        }

@app.get("/api/summary/{summary_id}")
def get_summary(summary_id: str):
    """Get summary by ID"""
    if summary_id not in summaries:
        raise HTTPException(status_code=404, detail="Summary not found")

    return summaries[summary_id]

@app.get("/api/export/{summary_id}/{format}")
def export_summary(summary_id: str, format: str):
    """Export summary in specified format"""
    if summary_id not in summaries:
        raise HTTPException(status_code=404, detail="Summary not found")

    summary = summaries[summary_id]
    if summary.get("status") != "completed":
        if summary.get("status") == "failed":
             raise HTTPException(status_code=400, detail=f"Summary generation failed: {summary.get('error', 'Unknown error')}")
        raise HTTPException(status_code=400, detail="Summary generation not completed or status unknown")


    if format == "markdown":
        # Generate Markdown
        markdown = "# Meeting Minutes\n\n"

        # Summary Section
        markdown += "## Summary\n\n"
        for block in summary.get("Summary", {}).get("blocks", []):
            markdown += f"- {block.get('content', '')}\n"

        # Actions Section
        markdown += "\n## Actions\n\n"
        for block in summary.get("Actions", {}).get("blocks", []):
            markdown += f"- {block.get('content', '')}\n"

        # Decisions Section
        markdown += "\n## Decisions\n\n"
        for block in summary.get("Decisions", {}).get("blocks", []):
            markdown += f"- {block.get('content', '')}\n"

        # Detailed Notes Section
        markdown += "\n## Detailed Notes\n\n"
        for block in summary.get("DetailedNotes", {}).get("blocks", []):
            markdown += f"- {block.get('content', '')}\n"

        return {"content": markdown, "format": "markdown"}

    elif format == "json":
        # Return the raw JSON
        return summary

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

if __name__ == "__main__":
    import uvicorn
    # Recommended: Add reload=True for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Assuming filename is main.py
