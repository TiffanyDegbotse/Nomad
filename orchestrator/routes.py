from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any
import pathlib
from ..core.utils import ROOT, read_yaml
from .handlers import PipelineHandlers


app = FastAPI(title="SightSense API")
settings = read_yaml(ROOT / "config" / "settings.yaml")
handlers = PipelineHandlers(settings)


@app.post("/ingest")
async def ingest_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    out_dir = ROOT / settings["paths"]["images_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / file.filename
    out_path.write_bytes(await file.read())

    result = handlers.process_images([out_path])

    return {
        "message": "processed",
        "reply": result["reply"],
        "vision": result["vision"],
        "audio": result["audio"],
    }

# FastAPI endpoint to post images
