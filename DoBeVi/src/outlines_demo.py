from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
from vllm.sampling_params import GuidedDecodingParams

# Set environment variables once
os.environ.update({
    'CUDA_VISIBLE_DEVICES': "3",
    'TOKENIZERS_PARALLELISM': 'false',
    'RAY_DEDUP_LOGS': '0',
    'RAY_memory_monitor_refresh_ms': '0',
    "VLLM_CONFIGURE_LOGGING": '0'
})

class Settings:
    GENERATOR_TYPE = "VllmTacticGenerator"
    ONE_STEP_MODEL_PATH = "/data0/xs/LLM-ATP/model/llm_based_atp"
    WHOLE_PROOF_MODEL_PATH = "/data0/xs/LLM-ATP/model/Goedel-Prover-V2-8B"
    VALUE_NETWORK_MODEL_PATH = "internlm/internlm2_5-step-prover-critic"
    NUM_GPUS = 2
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "info"
    USE_VALUE_NETWORK = False
    RAY_TEMP_DIR = "/data0/zjk/ATP/TEMP-DoBeVi/server_ray_cache"

class Generator(ABC):
    @abstractmethod
    async def generate_sampling(self, state: str, num_samples: int) -> List[Tuple[str, float]]:
        raise NotImplementedError

class VllmTacticGenerator(Generator):
    def __init__(self, model_path: str, length_penalty: float, max_length: int, gpu_id: Optional[int] = None):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate_sampling(self, state: str, num_samples: int) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []

        guided_decoding_params_regex = GuidedDecodingParams(regex=r'^(?:[^h].*|h[^a].*|ha[^v].*|hav[^e].*|h|ha|hav)$')
        guided_decoding_params_regex = GuidedDecodingParams(regex=r"^have.*")
        gen_params = SamplingParams(
            n=num_samples,
            temperature=1.1,
            # top_k=50,
            # top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1024,
            logprobs=0,
            # guided_decoding=guided_decoding_params_regex,
        )
        print(gen_params)
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output

        if not isinstance(final_output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(final_output)}")

        return [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in final_output.outputs]

async def generate_tactic_sampling(state: str, num_samples: int):
    generator = VllmTacticGenerator(
        model_path=Settings.ONE_STEP_MODEL_PATH,
        length_penalty=1.0,
        max_length=4096,
        gpu_id=0,
    )
    suggestions = await generator.generate_sampling(state, num_samples)
    for suggestion in suggestions:
        print(f"Generated Text: {suggestion[0]}")
        print(f"Logprob: {suggestion[1]}")

state = "[GOAL]\n⊢ ∃ A ⊆ Finset.Icc 1 100,\n    A.card = 16 ∧ ∃ a ∈ A, ∃ b ∈ A, ∃ c ∈ A, ∃ d ∈ A, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d\n[PROOFSTEP]\n"

# Run the sampling generation
asyncio.run(generate_tactic_sampling(state, 4))
