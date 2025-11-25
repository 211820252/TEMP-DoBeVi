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

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"  # Specify which GPUs to use
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['RAY_DEDUP_LOGS'] = '0' 
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
os.environ["VLLM_CONFIGURE_LOGGING"] = '0'
# os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
# os.environ['VLLM_USE_V1'] = '1'
# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

class settings:
    GENERATOR_TYPE = "VllmTacticGenerator"
    # ONE_STEP_MODEL_PATH = "/data1/xs/ckpt/atp/ckpt9000"
    # ONE_STEP_MODEL_PATH = "/data0/xs/LLM-ATP/LLaMA-Factory/saves/llm_based_atp-8B/full_sft_0831_2030"
    ONE_STEP_MODEL_PATH = "/data1/xs/ATP/train_model/sft_filter_leakage_100_seg_stp_1121"
    # ONE_STEP_MODEL_PATH = "/data1/xs/ATP/train_model/grpo_dobevi_1112/merged_hf_model/"
    # ONE_STEP_MODEL_PATH = "/data0/xs/LLM-ATP/model/llm_based_atp/"
    # ONE_STEP_MODEL_PATH = "/data1/zjk/models/sft_58w_of_8G_ours_1119/"
    # ONE_STEP_MODEL_PATH = "/data0/ljy/hf_home/hub/models--bytedance-research--BFS-Prover/snapshots/8c713e129d05643507aed4948635f81f5dc2d746/"
    # WHOLE_PROOF_MODEL_PATH = "/data0/xs/LLM-ATP/model/kimina-prover-preview-distill-7b"
    WHOLE_PROOF_MODEL_PATH = "/data0/xs/LLM-ATP/model/Goedel-Prover-V2-8B"
    # for HAVE model
    HAVE_ONE_STEP_MODEL_PATH = "/data0/xs/LLM-ATP/LLaMA-Factory/saves/llm_based_atp-8B/full_sft_0903_0930"
    # for value network
    VALUE_NETWORK_MODEL_PATH = "internlm/internlm2_5-step-prover-critic"
    NUM_GPUS = 4
    HOST = "0.0.0.0"
    PORT = 8002
    LOG_LEVEL = "info"
    USE_VALUE_NETWORK = False
    TEST_HAVE_MODEL = False
    RAY_TEMP_DIR = "/data0/xs/LLM-ATP/TEMP-DoBeVi/ray_tmp"

ray.init(_temp_dir=settings.RAY_TEMP_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # print(f"当前的attn实现为: {os.getenv('VLLM_ATTENTION_BACKEND')}")

    if settings.USE_VALUE_NETWORK:
        app.state.value_network = ValueNetwork.remote(
            model_path=settings.VALUE_NETWORK_MODEL_PATH,
            max_length=4096
        )
        app.state.value_network_lock = asyncio.Lock()
    
    if settings.TEST_HAVE_MODEL:
        app.state.have_generator = VllmTacticGenerator.remote(
            model_path=settings.HAVE_ONE_STEP_MODEL_PATH,
            length_penalty=1.0,
            max_length=4096,
            gpu_id=-1
        )
        app.state.have_generator_lock = asyncio.Lock()

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
        print(F"Using VllmTacticGenerator:{settings.ONE_STEP_MODEL_PATH}")
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
                max_length=8192,
                gpu_id=i,
            )
            for i in range(settings.NUM_GPUS)
        ]
    elif settings.GENERATOR_TYPE == "KiminaVllmProofGenerator":
        app.state.generators = [
            KiminaVllmProofGenerator.remote(
                model_path=settings.WHOLE_PROOF_MODEL_PATH,
                length_penalty=1.0,
                max_length=8192,
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
        gpu_id = request_body.gpu_id
        if gpu_id == -1:
            generator = app.state.have_generator
            lock = app.state.have_generator_lock
        else:
            generator = app.state.generators[gpu_id]
            lock = app.state.generator_locks[gpu_id]

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
