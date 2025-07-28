import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import our existing modules
from src.docparse import parse_pdf_to_sections
from src.relevance import RelevanceAnalyzer

app = FastAPI(
    title="DocuMind AI API",
    description="Persona-Driven Document Intelligence API",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the analyzer globally to avoid reloading the model
analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup"""
    global analyzer
    print("Loading AI models...")
    analyzer = RelevanceAnalyzer()
    print("AI models loaded successfully!")


# Pydantic models for request/response
class PlainTextConfig(BaseModel):
    persona: str
    job_to_be_done: str


class AnalysisRequest(BaseModel):
    config_type: str  # "plain-text", "json-file", "yaml-file"
    config_data: Optional[Dict[str, Any]] = None
    plain_text_config: Optional[PlainTextConfig] = None


class AnalysisResponse(BaseModel):
    metadata: Dict[str, Any]
    extracted_sections: List[Dict[str, Any]]
    subsection_analysis: List[Dict[str, Any]]


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "analyzer_loaded": analyzer is not None}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "analyzer_loaded": analyzer is not None,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_documents(
    files: List[UploadFile] = File(...),
    config_type: str = Form(...),
    persona: Optional[str] = Form(None),
    job_to_be_done: Optional[str] = Form(None),
    config_file: Optional[UploadFile] = File(None),
):
    print("[DEBUG] /analyze endpoint called")
    print(f"[DEBUG] config_type: {config_type}")
    print(f"[DEBUG] Uploaded files: {[file.filename for file in files]}")
    if config_file:
        print(f"[DEBUG] Config file uploaded: {config_file.filename}")
    else:
        print("[DEBUG] No config file uploaded")

    global analyzer

    if analyzer is None:
        raise HTTPException(status_code=500, detail="AI models not loaded")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Validate file types
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. Got: {file.filename}",
            )

    # Save uploaded files
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        print(f"[DEBUG] Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    # Process configuration based on type
    if config_type == "plain-text":
        if not persona or not job_to_be_done:
            raise HTTPException(
                status_code=400,
                detail="Persona and job_to_be_done are required for plain-text config",
            )
        config_data = {
            "persona": {"role": persona},
            "job_to_be_done": {"task": job_to_be_done},
        }
    elif config_type in ["json-file", "yaml-file"]:
        if not config_file:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration file is required for {config_type}",
            )

        config_content = await config_file.read()
        if config_type == "json-file":
            try:
                config_data = json.loads(config_content.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid JSON file: {str(e)}"
                )
        else:  # yaml-file
            try:
                import yaml

                config_data = yaml.safe_load(config_content.decode("utf-8"))
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid YAML file: {str(e)}"
                )
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported config_type: {config_type}"
        )

    print(f"[DEBUG] Config data: {config_data}")

    # Create relevance profile
    persona_role = config_data.get("persona", {}).get("role", "User")
    job_task = config_data.get("job_to_be_done", {}).get("task", "analyze documents")
    relevance_profile = f"As a {persona_role}, I need to {job_task}."

    # Parse all documents
    all_sections = []
    file_info = []

    for file_path in saved_files:
        sections = parse_pdf_to_sections(str(file_path), os.path.basename(file_path))
        all_sections.extend(sections)
        file_info.append(
            {"filename": os.path.basename(file_path), "sections_found": len(sections)}
        )

    if not all_sections:
        raise HTTPException(
            status_code=400,
            detail="No sections could be extracted from the uploaded documents",
        )

    # Rank sections using the analyzer
    ranked_sections = analyzer.rank_sections(all_sections, relevance_profile)

    # Perform subsection analysis
    subsection_analysis = analyzer.analyze_subsections(
        ranked_sections, relevance_profile
    )

    # Format response
    response = {
        "metadata": {
            "input_documents": [f["filename"] for f in file_info],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.now().isoformat(),
            "total_sections_extracted": len(all_sections),
            "config_type": config_type,
        },
        "extracted_sections": [
            {
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"],
                "page_number": section["page_number"],
                "relevance_score": round(section.get("relevance_score", 0) * 100, 1),
            }
            for section in ranked_sections[:10]  # Return top 10
        ],
        "subsection_analysis": subsection_analysis,
    }

    return response


@app.post("/upload-test")
async def test_upload(files: List[UploadFile] = File(...)):
    """Test endpoint to verify file upload functionality"""
    return {
        "message": f"Received {len(files)} files",
        "files": [
            {"name": f.filename, "size": f.size, "type": f.content_type} for f in files
        ],
    }


# Add this new endpoint after the existing ones
@app.get("/pdf-info")
async def get_pdf_info():
    """Get list of uploaded PDF files"""
    try:
        files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}


@app.get("/files/{filename}")
def get_pdf_with_headers(filename: str):
    """Serve PDF files with proper headers for Adobe Embed API"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pdf",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*",
            },
        )
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn

    print("Starting DocuMind AI API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
