from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ml.dataset_exporter import dataset_exporter
from app.ml.trainer import model_trainer, FineTuningJobConfig
from app.ml.rlhf import rlhf_logger, DPOPreferencePair

router = APIRouter(prefix="/training", tags=["training"])

class ExportDatasetRequest(BaseModel):
    conversations: List[List[Dict[str, str]]]
    output_filename: str = "data/fine_tuning_sft.jsonl"

@router.post("/export-dataset")
async def export_dataset_endpoint(request: ExportDatasetRequest):
    formatted = dataset_exporter.format_sharegpt(request.conversations)
    count = dataset_exporter.export_to_jsonl(formatted, request.output_filename)
    return {"status": "success", "records_exported": count, "file_path": request.output_filename}

@router.post("/start-finetune")
async def start_finetune_endpoint(config: FineTuningJobConfig):
    job_info = await model_trainer.start_fine_tuning_job(config)
    return job_info

@router.get("/job-status/{job_id}")
async def get_job_status_endpoint(job_id: str):
    status = model_trainer.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training job not found")
    return status

@router.post("/feedback")
async def submit_rlhf_feedback(pair: DPOPreferencePair):
    success = rlhf_logger.log_preference(pair)
    return {"status": "success" if success else "error", "pair_id": pair.pair_id}
