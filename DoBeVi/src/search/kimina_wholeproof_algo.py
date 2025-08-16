import asyncio
import logging
import time
import json
from typing import List, Optional, Dict
from datasets import load_dataset
import torch

from dojo import (
    TracedTheorem,
    Dojo,
    DojoCrashError,
    DojoInitError,
    LeanError,
    TacticState,
    ProofFinished,
    ProofGivenUp,
)

from search.search_algo import (
    Prover,
    SearchResult,
    client
)

from search.search_tree import (
    Status,
    SolvedNode,
    UnsolvedNode,
    InvalidNode,
    Edge,
    collect_success_edges,
)

from search.tactic_generator import (
    ModelEmptyOutputError,
)

from search.visual import (
    visualize_proof_tree,
    build_data,
)

class KiminaWholeproofProver(Prover):
    def __init__(
        self,
        actor_id: int,
        num_gpus: int,
        search_timeout: int,
        max_expansions: Optional[int],
        num_sampled_tactics: int,
        result_save_path: str,
    ):
        super().__init__(actor_id, num_gpus, search_timeout, max_expansions, num_sampled_tactics, result_save_path)
        self.dataset = load_dataset("AI-MO/minif2f_test", split="train")
        self.dataset = self.dataset.shuffle(seed=42)

    async def search(self, thm: TracedTheorem) -> None:
        """
        Search for a proof of a theorem using kimina wholeproof search.
        """
        self._prepare()
        await client._test_connection()

        self.thm = thm
    
        try:
            # initialize the current root node
            self.dojo = Dojo(
                thm.root_dir, 
                thm, 
                self.leandojo_tactic_timeout,
                self.leandojo_num_threads,
                self.leandojo_memory_limit
            ).__enter__()

            current_state = self.dojo.init_state
            
            self.root = UnsolvedNode(
                leandojo_state=current_state,
                is_terminal=False,
                priority=0.0, 
                depth=0,
            )
            
            self.nodes = {current_state.id: self.root}
            self.back_edges = []
            self.success_edges = []

            try:
                await self._wholeproof_search()
            except torch.OutOfMemoryError:
                logging.error(f"🚨OOM when sampling theorem: {self.thm.name}")
                torch.cuda.empty_cache()
            except DojoCrashError as e:
                logging.error(f"🚨{e}")

        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise

        except DojoInitError as e:
            logging.error(f"🚨Failed to initialize Dojo: {e}")
        
        except Exception as e:
            logging.error(f"🚨{type(e).__name__}: {e}")
        
        finally: 
            if hasattr(self, "dojo"):
                self.dojo.__exit__(None, None, None)

    async def _wholeproof_search(self) -> None:
        start_time = time.time()

        search_node = self.root
        logging.info(f"Expanding node: {search_node}")

        if isinstance(search_node.leandojo_state, TacticState):
            messages = self._build_prompt(search_node)
        else:
            raise ValueError(f"Invalid leandojo_state for search_node: {type(search_node.leandojo_state)}")
        
        # generate proofs by LLM
        proofs = await self._generate_proofs(messages)
        
        for proof in proofs:
            try:
                await self._run_proof(search_node, proof)
            except (asyncio.TimeoutError, asyncio.CancelledError, DojoCrashError):
                raise
            except Exception as e:
                logging.error(f"🚨{type(e).__name__}: {e}")


            self.elapsed_time = time.time() - start_time
            if self.elapsed_time > self.search_timeout:
                if self.root.status == Status.SOLVED:
                    logging.info("Search complete: proof found.")
                else:
                    logging.info("Exceeded time limit")
                break

            if self.root.status == Status.INVALID:
                logging.info("Search complete: proof failed.")

            if self.root.status == Status.SOLVED:
                logging.info("Search complete: proof found.")

    async def _run_proof(
        self, 
        search_node: UnsolvedNode, 
        tactics: List[str], 
    )-> None:
        start_time = time.time()

        for tactic in tactics:
            leandojo_new_state = await asyncio.to_thread(
                self.dojo.run_tac, search_node.leandojo_state, tactic
            )

            self.dojo_elapsed_time += time.time() - start_time # Tactics triggering timeout will not be counted
            
            # create a new child node
            depth = search_node.depth + 1
            if leandojo_new_state.id not in self.nodes:
                # proof finished
                if isinstance(leandojo_new_state, ProofFinished):
                    child_node = SolvedNode(leandojo_state=leandojo_new_state, depth=depth)
                    self.nodes[leandojo_new_state.id] = child_node
                    edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=0.0, norm_score=0.0)
                    child_node.in_edge = edge
                    search_node.out_edges = (search_node.out_edges or []) + [edge]
                    break
                # invalid tactic
                elif type(leandojo_new_state) in (LeanError, ProofGivenUp):
                    child_node = InvalidNode(leandojo_state=leandojo_new_state, depth=depth)
                    self.nodes[leandojo_new_state.id] = child_node
                    edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=0.0, norm_score=0.0)
                    child_node.in_edge = edge
                    search_node.out_edges = (search_node.out_edges or []) + [edge]
                    break
                # unsolved proof
                else:
                    assert isinstance(leandojo_new_state, TacticState), f"Expected TacticState, got{type(leandojo_new_state)}"
                    child_node = UnsolvedNode(
                        leandojo_state=leandojo_new_state,
                        is_terminal=False,
                        priority=0.0,
                        depth=depth,
                    )
                    self.nodes[leandojo_new_state.id] = child_node
                    edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=0.0, norm_score=0.0)
            else: 
                assert isinstance(leandojo_new_state, TacticState), f"Expected TacticState, got{type(leandojo_new_state)}"
                child_node = self.nodes[leandojo_new_state.id]
                assert isinstance(child_node, UnsolvedNode), f"Expected UnsolvedNode, got {type(child_node)}"
                child_node.depth = min(child_node.depth, depth)
                if await asyncio.to_thread(child_node.is_descendant, search_node):
                    edge = None
                else:
                    edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=0.0, norm_score=0.0)
                    self.back_edges.append(edge)
            
            if isinstance(child_node, UnsolvedNode) and edge is not None:
                child_node.in_edges.append(edge)
                search_node.out_edges = (search_node.out_edges or []) + [edge]
            
            search_node = child_node

    @torch.no_grad()
    async def _generate_proofs(self, messages: List[Dict[str, str]]) -> List[List[str]]:
        start_time = time.time()
        proofs = await client.async_generate_proof_sampling(
            self.select_tac_gen(),
            theorem_name=self.thm.name,
            init_state=messages,
            num_samples=self.num_sampled_tactics,
            output_dir=self.result_save_path + "/outputs"
        )

        proofs = json.loads(proofs['proofs'])
        
        if len(proofs) == 0:
            raise ModelEmptyOutputError("No proof generated.")

        self.model_elapsed_time += time.time() - start_time
        return proofs

    def _build_prompt(
        self,
        search_node: UnsolvedNode,
    ) -> List[Dict[str, str]]:
        filtered_dataset = self.dataset.filter(lambda example: example["name"] == self.thm.name)
        prompt = "Think about and solve the following problem step by step in Lean 4."
        prompt += f"\n# Problem:{filtered_dataset[0]['informal_prefix']}"""
        prompt += f"\n# Formal statement:\n```lean4\n{filtered_dataset[0]['formal_statement']}\n```\n"
        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt}
        ]
        return messages
    
    async def get_result(self, visualize: bool = True) -> Optional[SearchResult]:
        if not self.thm or not self.root:
            return None
        
        # check if the search was successful
        if self.root.status == Status.SOLVED:
            proof = [e.tactic for e in await asyncio.to_thread(self.root.extract_proof)]
            if visualize:
                self.success_edges = await asyncio.to_thread(collect_success_edges, self.root)
        else:
            proof = None
        
        if visualize:
            await asyncio.to_thread(
                visualize_proof_tree,
                list(self.nodes.values()),
                self.success_edges, 
                self.back_edges,
                self.result_save_path + "/visual",
                self.thm.name,
                ['simple','detail']
            )
        
        await asyncio.to_thread(
            build_data,
            list(self.nodes.values()),
            self.success_edges, 
            self.result_save_path + "/data",
            self.thm.name, 
        )

        # box the result
        result = SearchResult(
            theorem=self.thm,
            status=self.root.status,
            proof=proof,
            num_total_nodes=len(self.nodes),
            num_expansions=0,
            elapsed_time=self.elapsed_time,
            dojo_elapsed_time=self.dojo_elapsed_time,
            model_elapsed_time=self.model_elapsed_time,
        )

        return result