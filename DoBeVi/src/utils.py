import json
import os
from typing import List, Tuple, Dict, Optional
from search.search_algo import SearchResult, Status
from search.best_first_search_algo import BestFirstSearchProver
from search.group_score_search_algo import GroupScoreSearchProver
from search.internlm_bfs_algo import InternlmBFSProver
from search.layer_dropout_algo import LayerDropoutProver
from search.back_propagate_algo import BackPropagateProver
from search.value_net_algo import ValueNetProver
from search.mcts_algo import MCTSProver
from search.deepseek_algo import DeepSeekProver
from search.kimina_wholeproof_algo import KiminaWholeproofProver
from search.ds_wholeproof_algo import DsWholeproofProver

def get_num_gpus(cuda_visible_devices: str) -> int:
    return len(cuda_visible_devices.split(","))

def get_prover_clazz(algo_str: str):
    if algo_str == "best_first":
        return BestFirstSearchProver
    elif algo_str == "group_score":
        return GroupScoreSearchProver
    elif algo_str == "internlm_bfs":
        return InternlmBFSProver
    elif algo_str == "layer_dropout":
        return LayerDropoutProver
    elif algo_str == "back_propagate":
        return BackPropagateProver
    elif algo_str == "mcts":
        return MCTSProver
    elif algo_str == "value_net":
        return ValueNetProver
    elif algo_str == "deepseek":
        return DeepSeekProver
    elif algo_str == "kimina_wholeproof":
        return KiminaWholeproofProver
    elif algo_str == "ds_wholeproof":
        return DsWholeproofProver
    else:
        raise ValueError(f"Invalid algorithm: {algo_str}")

def _get_proof_length(results: List[Optional[SearchResult]]) -> Dict:
    path_len_dict = {}
    for result in results:
        if result == None or result.status != Status.SOLVED:
            continue
        path_len = len(result.proof)
        path_len_dict[path_len] = path_len_dict.get(path_len, 0) + 1
    return path_len_dict
    
def get_stats(results: List[Optional[SearchResult]]) -> Dict:
    stats_dict = {}
    stats_dict['proof_length'] = _get_proof_length(results)
    return stats_dict

def get_leancode_minif2f(name: str) -> str:
    file_path = "/data0/zjk/ATP/graduate/miniF2F-lean4/MiniF2F/Test_ds.lean"
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_lines, body_start_idx = [], 0
    for i, line in enumerate(lines):
        if line.strip().startswith('theorem ') or line.strip().startswith('--'):
            body_start_idx = i
            break
        header_lines.append(line)

    theorem_lines, in_target_theorem = [], False

    print("name:", name)
    for line in lines[body_start_idx:]:
        if line.strip().startswith(f'theorem {name}'):
            in_target_theorem = True

        if in_target_theorem:
            theorem_lines.append(line)

            if line.strip().endswith(':= by sorry'):
                break

    if not theorem_lines:
        raise ValueError(f"❌ Theorem '{name}' not found in {file_path}")

    formal_statement = ''.join(header_lines + theorem_lines)
    print(f"Formal statement for {name}:\n{formal_statement}\n")

    return formal_statement

def get_leancode_stp(name: str) -> str:
    dir_path = "/data0/xs/LLM-ATP/dataset/STPlean/STPlean"
    file_path = os.path.join(dir_path, f"{name}.lean")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        formal_statement = f.read()
        print(f"Formal statement for {name}:\n{formal_statement}\n")
    
    return formal_statement

def get_leancode_FineLeanCorpusLean(name: str) -> str:
    dir_path = "/data0/xs/LLM-ATP/dataset/FineLeanCorpus-lean/FineLeanCorpusLean"
    file_path = os.path.join(dir_path, f"{name}.lean")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        formal_statement = f.read()
        print(f"Formal statement for {name}:\n{formal_statement}\n")
    
    return formal_statement

