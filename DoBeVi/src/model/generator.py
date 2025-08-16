from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict
import torch
import ray
import logging
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
import uuid
import asyncio
import re
import os
import sys
import json
from contextlib import redirect_stdout

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

class ModelEmptyOutputError(Exception):
    """Raised when the model returns no output (empty result)."""
    pass

class Generator(ABC):
    pass

# Generator for tree search
class TacticGenerator(Generator):
    model_path: str
    length_penalty: float
    max_length: int
    gpu_id: Optional[int] = None

    @abstractmethod
    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError
    
    @abstractmethod
    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError
    
@ray.remote(num_gpus=1)
class HuggingFaceTacticGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.device = torch.device("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        try:
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.decoder_only = False
        except ValueError:
            self.generator = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            self.decoder_only = True
        self.generator.eval()

    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        
        gen_kwargs = dict(
            max_length=self.max_length,
            num_beams=num_samples,
            num_return_sequences=num_samples,
            length_penalty=self.length_penalty,
            early_stopping=True,
            do_sample=False,
            repetition_penalty=1.1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.generator.generate(
            **inputs,
            **gen_kwargs,
        )

        outputs_score = outputs.sequences_scores.tolist()
        outputs = outputs.sequences.view(num_samples, -1)

        output_text = []
        output_score = []
        for i in range(num_samples):
            model_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            if self.decoder_only:
                model_output = model_output.replace(state, "").strip().strip("\n").strip()
            else:
                model_output = model_output.strip().strip("\n").strip()
            output_text.append(model_output)
            log_prob_score = outputs_score[i]
            output_score.append(log_prob_score)
        
        return list(zip(output_text, output_score))
    
    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            logging.warning(f"Input length exceeds max_length: {inputs['input_ids'].size(1)}")
            return []
        
        gen_kwargs = dict(
            max_length=self.max_length,
            num_beams=num_samples,
            num_return_sequences=num_samples,
            length_penalty=self.length_penalty,
            early_stopping=True,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=1.1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.generator.generate(
            **inputs,
            **gen_kwargs,
        )

        outputs_score = outputs.sequences_scores.tolist()
        outputs = outputs.sequences.view(num_samples, -1)

        output_text = []
        output_score = []
        for i in range(num_samples):
            model_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            if self.decoder_only:
                model_output = model_output.replace(state, "").strip().strip("\n").strip()
            else:
                model_output = model_output.strip().strip("\n").strip()
            output_text.append(model_output)
            log_prob_score = outputs_score[i]
            output_score.append(log_prob_score)
        
        return list(zip(output_text, output_score))

@ray.remote(num_gpus=1)
class VllmTacticGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # init vllm
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        
        gen_params = SamplingParams(
            n=num_samples,
            temperature=0,
            repetition_penalty=1.1,
            logprobs=0,
        )
        
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        suggestions = [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in output.outputs]
        return suggestions

    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        
        gen_params = SamplingParams(
            n=num_samples,
            temperature=1.1,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=128,
            logprobs=0,
        )

        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        suggestions = [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in output.outputs]
        return suggestions
    
    # async def _run_query(
    #     self,
    #     state: str
    # ) -> Tuple[str, float]:
    #     gen_params = SamplingParams(
    #         n=1,
    #         temperature=0,
    #         top_k=50,
    #         top_p=0.95,
    #         max_tokens=128,
    #         logprobs=0,
    #     )
        
    #     async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
    #         final_output = output
    #     output = final_output
    #     if not isinstance(output, RequestOutput):
    #         raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
    #     resp = output.outputs[0].text.strip('\n') # Can not use strip() here
    #     score = output.outputs[0].cumulative_logprob or 0.0
    #     return state + resp, score
    
    # async def batch_generate_sampling(
    #     self,
    #     states: List[str],
    # ) -> List[Tuple[str, float]]:
    #     inputs = self.tokenizer(states, return_tensors="pt", padding=True).to("cuda")
    #     if inputs["input_ids"].size(1) > self.max_length:
    #         return []
    #     tasks = [asyncio.create_task(self._run_query(state)) for state in states]
    #     suggestions = []
    #     for task in asyncio.as_completed(tasks):
    #         resp, score = await task
    #         suggestions.append((resp, score))
    #     return suggestions
    
@ray.remote(num_gpus=1)
class InternlmVllmTacticGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # init vllm
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        
        gen_params = SamplingParams(
            n=num_samples,
            temperature=0,
            repetition_penalty=1.1,
            logprobs=0,
        )
        
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        suggestions = [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in output.outputs]
        return suggestions

    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        
        def prompt_style_internlm_chat_stepprover_extractor(result:str):
            return result
    
        def _unique_sorted(texts, scores):
            texts_ = []
            scores_ = []
            for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
                if t not in texts_:
                    texts_.append(t)
                    scores_.append(s)
            return texts_, scores_
    
        texts, scores = [], []
        params = SamplingParams(
            n=num_samples,
            temperature=1.1,
            max_tokens=128,
            stop=['<|im_end|>',],
            logprobs=True,
        )
        
        async for output in self.engine.generate(state, params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        for o in output.outputs:
            text = o.text.replace(self.tokenizer.eos_token, '').strip().strip("\n")
            score = o.cumulative_logprob or 0.0
            texts.append(text)
            scores.append(score)

        texts = list(map(prompt_style_internlm_chat_stepprover_extractor,texts))
        texts, scores = _unique_sorted(texts, scores)
        suggestions = []
        for i in range(len(texts)):
            suggestions.append((texts[i], scores[i]))
    
        return suggestions

# Generator for whole proof
class ProofGenerator(Generator):
    model_path: str
    length_penalty: float
    max_length: int
    gpu_id: Optional[int] = None

    @abstractmethod
    async def generate(
        self,
        theorem_name: str,
        init_state: List[Dict[str, str]],
        num_samples: int,
        output_dir: str
    ) -> List[List[str]]:
        raise NotImplementedError
    
    @abstractmethod
    async def generate_sampling(
        self,
        theorem_name: str,
        init_state: List[Dict[str, str]],
        num_samples: int,
        output_dir: str
    ) -> List[List[str]]:
        raise NotImplementedError

@ray.remote(num_gpus=1)
class KiminaVllmProofGenerator(ProofGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # init vllm
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async def generate(
        self,
        theorem_name: str,
        init_state: List[Dict[str, str]],
        num_samples: int,
        output_dir: str
    ) -> List[List[str]]:
        pass

    async def generate_sampling(
        self,
        theorem_name: str,
        init_state: List[Dict[str, str]],
        num_samples: int,
        output_dir: str
    ) -> List[List[str]]:
        
        def extract_leancode(code: str) -> str | None:
            match = re.search(r"```lean4\s*\n(.*?)```", code, re.DOTALL)
            return match.group(1).strip() if match else None
            
        def remove_comments(code: str) -> str:
            # 去除块注释
            code = re.sub(r'/-(.|\s)*?-/\s*', '', code)
            # 去除行注释（注意保留换行和缩进）
            code = re.sub(r'--.*', '', code)
            # 去除多余空白行（只保留非空行）
            lines = code.splitlines()
            non_empty_lines = [line for line in lines if line.strip() != '']
            return '\n'.join(non_empty_lines)
        
        def extract_proof(code: str) -> str | None:
            lines = code.splitlines()
            for idx, line in enumerate(lines):
                if line.strip().startswith("⊢"):
                    # 从 ⊢ 行之后开始收集
                    proof_lines = lines[idx + 1:]
                    # 如果后面确实有 proof 内容
                    if any(line.strip() != "" for line in proof_lines):
                        return "\n".join(proof_lines).rstrip()
                    else:
                        return None
                    
            by_index = code.find(":= by")
            if by_index != -1:
                wholeproof = code[by_index + 6:]  # +6 是为了跳过 " by"
                lines = wholeproof.splitlines()
                # 判断是否是用花括号包裹的 block
                if len(lines) >= 2 and lines[0].strip().startswith("{") and lines[-1].strip().endswith("}"):
                    # 移除首行的 `{` 和末行的 `}`
                    lines[0] = lines[0][lines[0].find("{")+1:]  # 去除 { 后的内容（可能含指令）
                    lines[-1] = lines[-1][:lines[-1].rfind("}")]  # 去除最后一行 } 前的内容
                    lines = [line for line in lines if line.strip() != ""]
                    return "\n".join(lines).rstrip()
                else:
                    return wholeproof

            return None
            
        def dedent_block_remove_blank(code: str) -> str:
            lines = code.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            if not non_empty_lines:
                return ""  
            indents = [len(re.match(r'^\s*', line).group()) for line in non_empty_lines]
            min_indent = min(indents)
            dedented_lines = [line[min_indent:] for line in non_empty_lines]
            return "\n".join(dedented_lines)
        
        # 处理包含<;>的连续代码块
        def is_sequenced_tactic_start(lines: List[str], i: int) -> bool:
            this_line = lines[i].strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return "<;>" in this_line or next_line.startswith("<;>")
       
        # 处理 rcases, cases, cases', by_cases, match 等
        def is_multibranch_rcases(line: str) -> bool:
            stripped = line.strip()
            return (
                re.match(r"^(rcases|cases|cases')\b.*with\b", stripped) is not None
                or stripped.startswith("by_cases ")
                or stripped.startswith("constructor")
                or stripped.startswith("induction")
                or stripped.startswith("match ") and " with " in stripped
            )
        
        # 处理带by或calc的缩进连续代码块
        def is_by_block_start(line: str) -> bool:
             return (
                re.match(r"^(have|rcases|calc|let|suffices|show)\b.*:=\s*by\b", line)
                or line.startswith("have")
                or line.startswith("field_simp")
                or line.startswith("nlinarith")
                or line.startswith("calc")
                or line.startswith("all_goals")
            )
        
        # 处理refine<...>的代码块
        def is_refine_by_block_start(line: str) -> bool:
            return line.strip().startswith("refine ⟨") and "by" in line
        
        # 处理all goals{...}的代码块
        def is_all_goals_block_start(line: str) -> bool:
            return line.strip().startswith("all_goals {")
        
        def collect_sequenced_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
            block_lines = [lines[start_idx]]
            i = start_idx + 1
            base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            line = lines[start_idx].strip()
            if line.endswith("(") or ( i<len(lines) and lines[i].strip().startswith("(")):
                paren_depth = 1
                while i < len(lines) and paren_depth > 0:
                    next_line = lines[i]
                    block_lines.append(next_line)
                    paren_depth += next_line.count("(") - next_line.count(")")
                    i += 1
                return "\n".join(block_lines).rstrip(), i
            
            i = start_idx + 1
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())
                stripped = next_line.strip()
                if stripped == "":
                    block_lines.append(lines[i])
                    i += 1
                elif next_indent > base_indent:
                    block_lines.append(next_line)
                    i += 1
                elif stripped.startswith("<;>"):
                    block_lines.append(next_line)
                    i += 1
                # elif line[i-1].strip().endswith("<;>"):
                #     block_lines.append(lines[i])
                #     i += 1
                else:
                    break
            
            return "\n".join(block_lines).rstrip(), i

        def collect_multibranch_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
            block_lines = [lines[start_idx]]
            i = start_idx + 1
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith("·") or line.strip().startswith("|") or line.strip().startswith("."):
                    block_lines.append(line)
                    i += 1
                    while i < len(lines):
                        sub_line = lines[i]
                        if sub_line.strip() == "":
                            block_lines.append(sub_line)
                            i += 1
                        elif len(sub_line) - len(sub_line.lstrip()) > 0:
                            block_lines.append(sub_line)
                            i += 1
                        else:
                            break
                else:
                    break
            return "\n".join(block_lines).rstrip(), i

        def collect_by_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
            block_lines = [lines[start_idx]]
            i = start_idx + 1
            base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())
                stripped = next_line.strip()
                if stripped == "":
                    block_lines.append(next_line)
                    i += 1
                elif next_indent > base_indent:
                    block_lines.append(next_line)
                    i += 1
                elif stripped.startswith("<;>"):
                    block_lines.append(next_line)
                    i += 1
                else:
                    break
            return "\n".join(block_lines).rstrip(), i
        
        def collect_refine_by_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
            block_lines = [lines[start_idx]]
            i = start_idx + 1
            open_brackets = 1  # 对应 ⟨

            while i < len(lines) and open_brackets > 0:
                line = lines[i]
                open_brackets += line.count("⟨") - line.count("⟩")
                block_lines.append(line)
                i += 1

            return "\n".join(block_lines).rstrip(), i
        
        def collect_all_goals_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
            block_lines = [lines[start_idx]]
            i = start_idx + 1
            open_brackets = 1  # 对应 {

            while i < len(lines) and open_brackets > 0:
                line = lines[i]
                open_brackets += line.count("{") - line.count("}")
                block_lines.append(line)
                i += 1

            return "\n".join(block_lines).rstrip(), i
        
        def split_tactics(wholeproof: str) -> List[str]:
            lines = wholeproof.splitlines()
            blocks = []
            i = 0
            n = len(lines)

            while i < n:
                line = lines[i]
                stripped = line.strip()

                if not stripped:
                    i += 1
                    continue

                if is_sequenced_tactic_start(lines, i):
                    block, i = collect_sequenced_block(lines, i)
                    blocks.append(block)
                    continue

                if is_multibranch_rcases(stripped):
                    block, i = collect_multibranch_block(lines, i)
                    blocks.append(block)
                    continue

                if is_refine_by_block_start(stripped):
                    block, i = collect_refine_by_block(lines, i)
                    blocks.append(block)
                    continue

                if is_all_goals_block_start(stripped):
                    block, i = collect_all_goals_block(lines, i)
                    blocks.append(block)
                    continue

                if is_by_block_start(stripped):
                    block, i = collect_by_block(lines, i)
                    blocks.append(block)
                    continue

                # fallback: single-line tactic
                blocks.append(line.rstrip())
                i += 1

            return blocks

        proofs=[]
        
        params = SamplingParams(
            n=num_samples,
            temperature=1.1,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=8192,
        )

        text = self.tokenizer.apply_chat_template(
            init_state,
            tokenize=False,
            add_generation_prompt=True
        )

        async for output in self.engine.generate(text, params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{theorem_name}.txt")
        with open(output_path, "a", encoding="utf-8") as f:
            for i, o in enumerate(output.outputs):
                text = o.text.strip()
                f.write(f"==========Output {i+1} ==========\n")
                f.write("=====text=====\n" + text + "\n\n")
                leancode = extract_leancode(text)
                if leancode is not None:
                    leancode = remove_comments(leancode)
                    f.write("=====leancode=====\n" + leancode + "\n")
                    proof = extract_proof(leancode)
                    if proof is not None:
                        f.write("=====proof1=====\n" + proof + "\n")
                        proof = dedent_block_remove_blank(proof)
                        f.write("=====proof2=====\n" + proof + "\n")
                        tactics = split_tactics(proof)
                        f.write("=====tactics=====\n")
                        for tactic in tactics:
                            f.write(tactic.strip() + "\n\n")
                        proofs.append(tactics)
                    else:
                        f.write("=====proof=====\nNone\n")
                else:
                    f.write("=====leancode=====\nNone\n")
            f.flush()
        return proofs            

@ray.remote(num_gpus=1)
class DsVllmProofGenerator(ProofGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # init vllm
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async def generate(
        self,
        theorem_name: str,
        init_state: List[Dict[str, str]],
        num_samples: int,
        output_dir: str
    ) -> List[List[str]]:
        pass

    async def generate_sampling(
        self,
        theorem_name: str,
        init_state: List[Dict[str, str]],
        num_samples: int,
        output_dir: str
    ) -> List[List[str]]:
        
        def extract_leancode(code: str) -> str | None:
            pattern = r"###\s*Complete Lean 4 Proof\s*```lean4?\s*\n(.*?)```"
            match = re.search(pattern, code, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None
            
        def remove_comments(code: str) -> str:
            # 去除块注释
            code = re.sub(r'/-(.|\s)*?-/\s*', '', code)
            # 去除行注释（注意保留换行和缩进）
            code = re.sub(r'--.*', '', code)
            # 去除多余空白行（只保留非空行）
            lines = code.splitlines()
            non_empty_lines = [line for line in lines if line.strip() != '']
            return '\n'.join(non_empty_lines)

        def extract_proof(code: str) -> str | None:            
            by_index = code.find(":= by")
            if by_index != -1:
                wholeproof = code[by_index + 6:]
                return wholeproof
            return None

        def dedent_block_remove_blank(code: str) -> str:
            lines = code.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            if not non_empty_lines:
                return ""  
            indents = [len(re.match(r'^\s*', line).group()) for line in non_empty_lines]
            min_indent = min(indents)
            dedented_lines = [line[min_indent:] for line in non_empty_lines]
            return "\n".join(dedented_lines)
    
        # 处理包含<;>的连续tactic使用
        def is_sequenced_tactic_start(lines: List[str], i: int) -> bool:
            this_line = lines[i].strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return "<;>" in this_line or next_line.startswith("<;>")

        # 处理have...by等缩进块
        def is_by_block_start(line: str) -> bool:
                return (
                re.match(r"^(have|rcases|calc|let|suffices|show)\b.*:=\s*by\b", line)
                or line.startswith("field_simp")
                or line.startswith("nlinarith")
                or line.startswith("calc")
                or line.startswith("all_goals")
            )

        def collect_sequenced_block(lines: List[str], i: int) -> Tuple[str, int]:
            block_lines = [lines[i]]
            base_indent = len(lines[i]) - len(lines[i].lstrip())
            i = i + 1
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())
                stripped = next_line.strip()
                if stripped == "":
                    block_lines.append(lines[i])
                    i += 1
                elif next_indent > base_indent:
                    block_lines.append(next_line)
                    i += 1
                elif stripped.startswith("<;>"):
                    block_lines.append(next_line)
                    i += 1
                elif i > 0 and lines[i - 1].endswith("<;>"):
                    block_lines.append(next_line)
                    i += 1
                else:
                    break
            
            return "\n".join(block_lines).rstrip(), i

        def collect_by_block(lines: List[str], i: int) -> Tuple[str, int]:
            block_lines = [lines[i]]
            base_indent = len(lines[i]) - len(lines[i].lstrip())
            i = i + 1
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())
                stripped = next_line.strip()
                if stripped == "":
                    block_lines.append(next_line)
                    i += 1
                elif next_indent > base_indent:
                    block_lines.append(next_line)
                    i += 1
                elif stripped.startswith("<;>"):
                    block_lines.append(next_line)
                    i += 1
                else:
                    break
            return "\n".join(block_lines).rstrip(), i

        def split_tactics(wholeproof: str) -> List[str]:
            lines = wholeproof.splitlines()
            blocks = []
            i = 0
            n = len(lines)

            while i < n:
                line = lines[i]
                stripped = line.strip()

                if is_sequenced_tactic_start(lines, i):
                    block, i = collect_sequenced_block(lines, i)
                    blocks.append(block)
                    continue

                if is_by_block_start(stripped):
                    block, i = collect_by_block(lines, i)
                    blocks.append(block)
                    continue

                blocks.append(line.rstrip())
                i += 1
            
            return blocks
        
        # with open('output815.txt', 'w') as f:
        #     with redirect_stdout(f):
        #         print("=============================================================")
        #         print(f"{type(init_state)}\n{init_state}\n\n")
        #         print(f"{type(json.loads(init_state))}\n{json.loads(init_state)}")
        #         print("=============================================================\n\n\n\n")

        proofs=[]
        
        params = SamplingParams(
            temperature=1.2,
            top_k=50,
            top_p = 0.95,
            # repetition_penalty=1.1,
            max_tokens=32768,
        )

        text = self.tokenizer.apply_chat_template(
            json.loads(init_state),
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # with open('output815.txt', 'a') as f:
        #     with redirect_stdout(f):
        #         print("=============================================================")
        #         print(f"{text}\n\n")
        #         print("=============================================================\n\n\n\n")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{theorem_name}.txt")

        with open(output_path, "a", encoding="utf-8") as f:
            # none_count = 0
            for idx in range(num_samples):
                async for output in self.engine.generate(text, params, request_id=str(uuid.uuid4().hex)):
                    final_output = output
                output = final_output
                # with open('output815.txt', 'a') as f1:
                #     with redirect_stdout(f1):
                #         print("=============================================================")
                #         print(f"{idx}\n")
                #         for o in output.outputs:
                #             print(f"{o.text}")
                #         print("=============================================================\n\n\n\n")
                if not isinstance(output, RequestOutput):
                    raise ValueError(f"Expected RequestOutput, got {type(output)}")
                # leancode_extract = False
                for o in output.outputs:
                    f.write(f"==========Output {idx+1} ==========\n")
                    f.write("====text====\n" + o.text.strip() + "\n\n")
                    leancode = extract_leancode(o.text.strip())
                    if leancode is not None:
                        # leancode_extract = True
                        leancode = remove_comments(leancode)
                        f.write("=====leancode=====\n" + leancode + "\n")
                        proof = extract_proof(leancode)
                        if proof is not None:
                            f.write("=====proof1=====\n" + proof + "\n")
                            proof = dedent_block_remove_blank(proof)
                            f.write("=====proof2=====\n" + proof + "\n")
                            tactics = split_tactics(proof)
                            f.write("=====tactics=====\n")
                            for tactic in tactics:
                                f.write(tactic.strip() + "\n\n")
                            proofs.append(tactics)
                        else:
                            f.write("=====proof=====\nNone\n")
                    else:
                        f.write("=====leancode=====\nNone\n")
                # if not leancode_extract:
                #     none_count += 1
                #     if none_count >= num_samples // 2:
                #         print(f"leancode 为 None 的次数已达到 {none_count}(>= 一半)，提前结束循环")
                #         break
                f.flush()

        return proofs  