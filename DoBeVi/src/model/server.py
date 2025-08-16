import os
import asyncio
import torch
import json
import ray
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from model.generator import (
    HuggingFaceTacticGenerator, 
    VllmTacticGenerator, 
    InternlmVllmTacticGenerator,
    DsVllmProofGenerator,
    KiminaVllmProofGenerator,
)
from model.value_net import ValueNetwork

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['RAY_DEDUP_LOGS'] = '0' 
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
os.environ["VLLM_CONFIGURE_LOGGING"] = '0'

class settings:
    GENERATOR_TYPE = "DsVllmProofGenerator"
    ONE_STEP_MODEL_PATH = "/data0/xs/LLM-ATP/model/llm_based_atp"
    # ONE_STEP_MODEL_PATH = "/data0/ljy/hf_home/hub/models--bytedance-research--BFS-Prover/snapshots/8c713e129d05643507aed4948635f81f5dc2d746/"
    # WHOLE_PROOF_MODEL_PATH = "/data0/xs/LLM-ATP/model/kimina-prover-preview-distill-7b"
    WHOLE_PROOF_MODEL_PATH = "/data0/xs/LLM-ATP/model/Goedel-Prover-V2-8B"
    VALUE_NETWORK_MODEL_PATH = "internlm/internlm2_5-step-prover-critic"
    NUM_GPUS = 1
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "info"
    USE_VALUE_NETWORK = False
    RAY_TEMP_DIR = "/data0/zjk/ATP/TEMP-DoBeVi/server_ray_cache"

ray.init(_temp_dir=settings.RAY_TEMP_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.USE_VALUE_NETWORK:
        app.state.value_network = ValueNetwork.remote(
            model_path=settings.VALUE_NETWORK_MODEL_PATH,
            max_length=4096
        )
        app.state.value_network_lock = asyncio.Lock()
    
    if settings.GENERATOR_TYPE == "HuggingFaceTacticGenerator":
        app.state.generators = [
            HuggingFaceTacticGenerator.remote(
                model_path=settings.ONE_STEP_MODEL_PATH,
                length_penalty=1.0,
                max_length=4096,
                gpu_id=i,
            )
            for i in range(settings.NUM_GPUS)
        ]
    elif settings.GENERATOR_TYPE == "VllmTacticGenerator":
        app.state.generators = [
            VllmTacticGenerator.remote(
                model_path=settings.ONE_STEP_MODEL_PATH,
                length_penalty=1.0,
                max_length=4096,
                gpu_id=i,
            )
            for i in range(settings.NUM_GPUS)
        ]
    elif settings.GENERATOR_TYPE == "InternlmVllmTacticGenerator":
        app.state.generators = [
            InternlmVllmTacticGenerator.remote(
                model_path=settings.ONE_STEP_MODEL_PATH,
                length_penalty=1.0,
                max_length=4096,
                gpu_id=i,
            )
            for i in range(settings.NUM_GPUS)
        ]
    elif settings.GENERATOR_TYPE == "DsVllmProofGenerator":
        app.state.generators = [
            DsVllmProofGenerator.remote(
                model_path=settings.WHOLE_PROOF_MODEL_PATH,
                length_penalty=1.0,
                max_length=32768,
                gpu_id=i,
            )
            for i in range(settings.NUM_GPUS)
        ]
    elif settings.GENERATOR_TYPE == "KiminaVllmProofGenerator":
        app.state.generators = [
            KiminaVllmProofGenerator.remote(
                model_path=settings.WHOLE_PROOF_MODEL_PATH,
                length_penalty=1.0,
                max_length=4096,
                gpu_id=i,
            )
            for i in range(settings.NUM_GPUS)
        ]
    else: # invalid settings.GENERATOR_TYPE:
        raise ValueError(f"Invalid settings.GENERATOR_TYPE: {settings.GENERATOR_TYPE}")

    app.state.generator_locks = [
        asyncio.Lock() for _ in range(settings.NUM_GPUS)
    ]

    logger.info(f"Initialized {len(app.state.generators)} tactic generators on GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    try:
        yield
    finally:
        for actor in app.state.generators:
            ray.kill(actor)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "ok"}

class GenerateTacticRequestBody(BaseModel):
    gpu_id: int
    state: str
    num_samples: int

@app.post("/generate_tactic")
async def generate_tactic(request_body: GenerateTacticRequestBody):
    try:
        generator = app.state.generators[request_body.gpu_id]
        lock = app.state.generator_locks[request_body.gpu_id]

        async with lock:
            suggestions = await generator.generate.remote(
                request_body.state,
                request_body.num_samples
            )

        suggestions_json = [
            {
                "tactic": text,
                "score": score
            }
            for text, score in suggestions
        ]

        return {"suggestions": suggestions_json}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
    
@app.post("/generate_tactic_sampling")
async def generate_tactic_sampling(request_body: GenerateTacticRequestBody):
    try:
        generator = app.state.generators[request_body.gpu_id]
        lock = app.state.generator_locks[request_body.gpu_id]

        async with lock:
            suggestions = await generator.generate_sampling.remote(
                request_body.state,
                request_body.num_samples
            )

        suggestions_json = [
            {
                "tactic": text,
                "score": score
            }
            for text, score in suggestions
        ]

        return {"suggestions": suggestions_json}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))

class GenerateProofRequestBody(BaseModel):
    gpu_id: int
    theorem_name: str
    init_state: str
    num_samples: int
    output_dir: str

@app.post("/generate_proof")
async def generate_proof(request_body: GenerateProofRequestBody):
    try:
        generator = app.state.generators[request_body.gpu_id]
        lock = app.state.generator_locks[request_body.gpu_id]

        async with lock:
            proofs = await generator.generate.remote(
                request_body.theorem_name,
                request_body.init_state,
                request_body.num_samples,
                request_body.output_dir
            )

        return {"proofs": json.dumps(proofs)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))

@app.post("/generate_proof_sampling")
async def generate_proof_sampling(request_body: GenerateProofRequestBody):
    try:
        generator = app.state.generators[request_body.gpu_id]
        lock = app.state.generator_locks[request_body.gpu_id]
        async with lock:
            print(request_body.init_state)
            proofs = await generator.generate_sampling.remote(
                request_body.theorem_name,
                request_body.init_state,
                request_body.num_samples,
                request_body.output_dir
            )

        return {"proofs": json.dumps(proofs)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))

class ValueNetworkRequestBody(BaseModel):
    state: str

@app.post("/value_network_score")
async def value_network_score(request_body: ValueNetworkRequestBody):
    try:
        if not settings.USE_VALUE_NETWORK:
            raise HTTPException(status_code=400, detail="Value network is disabled.")
        
        chat = [
            {"role": "user", "content": "Which state is closer to 'no goals'?"},
            {"role": "assistant", "content": request_body.state}
        ]
        
        async with app.state.value_network_lock:
            score = await app.state.value_network.get_score.remote(
                chat
            )
        
        return {"score": score}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
