import asyncio
import logging
import time
from typing import List, Optional, Tuple
import torch
import math
import json
import os

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
    get_data_for_grpo,
    save_tree,
    load_tree,
)

from search.tactic_generator import (
    ModelEmptyOutputError,
)

from search.visual import (
    visualize_proof_tree,
)

class BestFirstSearchProver(Prover):
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

    async def search(self, thm: TracedTheorem) -> None:
        """
        Search for a proof of a theorem using best-first search.
        """
        self._prepare()
        await client._test_connection()

        self.thm = thm
        self.root = None
        self.dojo_ctx = None
        self.dojo = None

        try:
            # initialize the current root node
            self.dojo_ctx = Dojo(
                thm.root_dir, 
                thm, 
                self.leandojo_tactic_timeout,
                self.leandojo_num_threads,
                self.leandojo_memory_limit
            )
            self.dojo = self.dojo_ctx.__enter__()

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
                await self._best_first_search()
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
            if hasattr(self, "dojo_ctx") and isinstance(self.dojo_ctx, Dojo):
                self.dojo_ctx.__exit__(None, None, None)
                self.dojo_ctx = None
                self.dojo = None
            
    async def _best_first_search(self) -> None:
        start_time = time.time()
        priority_queue = asyncio.PriorityQueue() # lowest priority first
        priority_queue.put_nowait((-self.root.priority, self.root))

        while True:
            if priority_queue.empty():
                logging.info("Search queue is empty.")
                break
            try:
                await self._step(priority_queue)
            except (asyncio.TimeoutError, asyncio.CancelledError, DojoCrashError):
                raise
            except Exception as e:
                logging.error(f"🚨{type(e).__name__}: {e}")

            self.elapsed_time = time.time() - start_time
            if self.max_expansions and self.num_expansions >= self.max_expansions:
                if self.root.status == Status.SOLVED:
                    logging.info("Search complete: proof found.")
                else:
                    logging.info("Exceeded time limit or max expansions.")
                break

            if self.root.status == Status.INVALID:
                logging.info("Search complete: proof failed.")
                break

            if self.root.status == Status.SOLVED:
                logging.info("Search complete: proof found.")
                break

    async def _step(self, priority_queue: asyncio.PriorityQueue[Tuple[float, UnsolvedNode]]) -> None:
        _, search_node = priority_queue.get_nowait()
        
        logging.info(f"Expanding node: {search_node}")

        if isinstance(search_node.leandojo_state, TacticState):
            tactic_state_str = self._build_prompt(search_node)
        else:
            raise ValueError(f"Invalid leandojo_state for search_node: {type(search_node.leandojo_state)}")
        
        # generate tactics by LLM
        suggestions = await self._generate_tactics(tactic_state_str)
        normalized_suggestions = self._normalize_scores(suggestions)
        normalized_suggestions = sorted(normalized_suggestions, key=lambda x: x[2], reverse=True)

        # try all the tactics
        results = []
        
        for tactic, score, norm_score in normalized_suggestions:
            edge, proof_finished = await self._run_tactic(
                search_node, tactic, score, norm_score, priority_queue
            )
            
            if edge is not None:
                results.append(edge)

            if proof_finished:
                break
        
        # expand the search node
        search_node.out_edges = results
        self.num_expansions += 1
        priority_queue.task_done()

    @torch.no_grad()
    async def _generate_tactics(self, tactic_state_str: str) -> List[Tuple[str, float]]:
        start_time = time.time()
        suggestions = await client.async_generate_tactic_sampling(
            self.select_tac_gen(),
            state=tactic_state_str,
            num_samples=self.num_sampled_tactics
        )

        suggestions = [(item['tactic'], item['score']) for item in suggestions['suggestions']]

        if len(suggestions) == 0:
            raise ModelEmptyOutputError("No tactic generated.")

        self.model_elapsed_time += time.time() - start_time
        return suggestions
    
    async def _run_tactic(
        self, 
        search_node: UnsolvedNode, 
        tactic: str, 
        score: float, 
        norm_score: float,
        priority_queue: asyncio.PriorityQueue
    ) -> Tuple[Optional[Edge], bool]:
        # run tactic in Lean server
        start_time = time.time()

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
            # invalid tactic
            elif type(leandojo_new_state) in (LeanError, ProofGivenUp):
                child_node = InvalidNode(leandojo_state=leandojo_new_state, depth=depth)
            # unsolved proof
            else:
                assert isinstance(leandojo_new_state, TacticState), f"Expected TacticState, got{type(leandojo_new_state)}"
                child_node = UnsolvedNode(
                    leandojo_state=leandojo_new_state,
                    is_terminal=False,
                    priority=search_node.priority + score,
                    depth=depth,
                )
                priority_queue.put_nowait((-child_node.priority, child_node))
            self.nodes[leandojo_new_state.id] = child_node
            edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=score, norm_score=norm_score)
        else: 
            assert isinstance(leandojo_new_state, TacticState), f"Expected TacticState, got{type(leandojo_new_state)}"
            child_node = self.nodes[leandojo_new_state.id]
            assert isinstance(child_node, UnsolvedNode), f"Expected UnsolvedNode, got {type(child_node)}"
            child_node.depth = min(child_node.depth, depth)
            if await asyncio.to_thread(child_node.is_descendant, search_node):
                edge = None
            else:
                edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=score, norm_score=norm_score)
                self.back_edges.append(edge)

        if isinstance(child_node, UnsolvedNode) and edge is not None:
            child_node.in_edges.append(edge)
        elif isinstance(child_node, SolvedNode) or isinstance(child_node, InvalidNode):
            child_node.in_edge = edge

        return edge, isinstance(leandojo_new_state, ProofFinished)
    
    def _build_prompt(
        self,
        search_node: UnsolvedNode,
    ) -> str:
        input_template = "[GOAL]\n{state}\n[PROOFSTEP]\n"
        return input_template.format(state=search_node.leandojo_state.pp)
        # return f"{search_node.leandojo_state.pp}:::"

    def _normalize_scores(
        self,
        suggestions: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, float]]:

        exps = [math.exp(score) for _, score in suggestions]
        total = sum(exps)
        
        return [
            (tactic, score, exp_score / total)
            for (tactic, score), exp_score in zip(suggestions, exps)
        ]

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
        # # construct grpo data
        # try:
        #     grpo_data = get_data_for_grpo(self.root)
        #     grpo_dir = f"{self.result_save_path}/grpo"
        #     os.makedirs(grpo_dir, exist_ok=True)  
        #     with open(f"{grpo_dir}/{self.thm.name}_grpo.json", "w", encoding="utf-8") as f:
        #         json.dump(grpo_data, f, ensure_ascii=False, indent=4) 
        # except ValueError as e:
        #     logging.error(f"🚨Failed to construct GRPO data for theorem {self.thm.name}: {e}") 

        try:
            tree_data = save_tree(self.root, str(self.thm.root_dir), str(self.thm.path), self.thm.name)
            tree_dir = f"{self.result_save_path}/tree"
            os.makedirs(tree_dir, exist_ok=True)
            with open(f"{tree_dir}/{self.thm.name}_tree.json", "w", encoding="utf-8") as f:
                json.dump(tree_data, f, ensure_ascii=False, indent=4)
        except ValueError as e:
            logging.error(f"🚨Failed to save tree for theorem {self.thm.name}: {e}")

        # box the result
        result = SearchResult(
            theorem=self.thm,
            status=self.root.status,
            proof=proof,
            num_total_nodes=len(self.nodes),
            num_expansions=self.num_expansions,
            elapsed_time=self.elapsed_time,
            dojo_elapsed_time=self.dojo_elapsed_time,
            model_elapsed_time=self.model_elapsed_time,
        )
        return result

# import asyncio
# import logging
# import time
# from typing import List, Optional, Tuple
# import torch
# import math

# from dojo import (
#     TracedTheorem,
#     Dojo,
#     DojoCrashError,
#     DojoInitError,
#     LeanError,
#     TacticState,
#     ProofFinished,
#     ProofGivenUp,
# )

# from search.search_algo import (
#     Prover,
#     SearchResult,
#     client
# )

# from search.search_tree import (
#     Status,
#     SolvedNode,
#     UnsolvedNode,
#     InvalidNode,
#     Edge,
#     collect_success_edges,
# )

# from search.tactic_generator import (
#     ModelEmptyOutputError,
# )

# from search.visual import (
#     visualize_proof_tree,
# )

# class BestFirstSearchProver(Prover):
#     def __init__(
#         self,
#         actor_id: int,
#         num_gpus: int,
#         search_timeout: int,
#         max_expansions: Optional[int],
#         num_sampled_tactics: int,
#         result_save_path: str,
#     ):
#         super().__init__(actor_id, num_gpus, search_timeout, max_expansions, num_sampled_tactics, result_save_path)

#     async def search(self, thm: TracedTheorem) -> None:
#         """
#         Search for a proof of a theorem using best-first search.
#         """
#         self._prepare()
#         await client._test_connection()

#         self.thm = thm

#         try:
#             # initialize the current root node
#             self.dojo = Dojo(
#                 thm.root_dir, 
#                 thm, 
#                 self.leandojo_tactic_timeout,
#                 self.leandojo_num_threads,
#                 self.leandojo_memory_limit
#             ).__enter__()

#             current_state = self.dojo.init_state
            
#             self.root = UnsolvedNode(
#                 leandojo_state=current_state,
#                 is_terminal=False,
#                 priority=0.0, 
#                 depth=0,
#             )
#             self.root.value = (await client.async_get_score(current_state.pp))["score"]

#             self.nodes = {current_state.id: self.root}
#             self.back_edges = []
#             self.success_edges = []

#             try:
#                 await self._best_first_search()
#             except torch.OutOfMemoryError:
#                 logging.error(f"🚨OOM when sampling theorem: {self.thm.name}")
#                 torch.cuda.empty_cache()
#             except DojoCrashError as e:
#                 logging.error(f"🚨{e}")
        
#         except (asyncio.TimeoutError, asyncio.CancelledError):
#             raise

#         except DojoInitError as e:
#             logging.error(f"🚨Failed to initialize Dojo: {e}")
        
#         except Exception as e:
#             logging.error(f"🚨{type(e).__name__}: {e}")
        
#         finally: 
#             if hasattr(self, "dojo"):
#                 self.dojo.__exit__(None, None, None)
            
#     async def _best_first_search(self) -> None:
#         start_time = time.time()
#         priority_queue = asyncio.PriorityQueue() # lowest priority first
#         priority_queue.put_nowait((-self.root.value, self.root))

#         while True:
#             if priority_queue.empty():
#                 logging.info("Search queue is empty.")
#                 break
#             try:
#                 await self._step(priority_queue)
#             except (asyncio.TimeoutError, asyncio.CancelledError, DojoCrashError):
#                 raise
#             except Exception as e:
#                 logging.error(f"🚨{type(e).__name__}: {e}")

#             self.elapsed_time = time.time() - start_time
#             if self.max_expansions and self.num_expansions >= self.max_expansions:
#                 if self.root.status == Status.SOLVED:
#                     logging.info("Search complete: proof found.")
#                 else:
#                     logging.info("Exceeded time limit or max expansions.")
#                 break

#             if self.root.status == Status.INVALID:
#                 logging.info("Search complete: proof failed.")
#                 break

#             if self.root.status == Status.SOLVED:
#                 logging.info("Search complete: proof found.")
#                 break

#     async def _step(self, priority_queue: asyncio.PriorityQueue[Tuple[float, UnsolvedNode]]) -> None:
#         _, search_node = priority_queue.get_nowait()
        
#         logging.info(f"Expanding node: {search_node}")

#         if isinstance(search_node.leandojo_state, TacticState):
#             tactic_state_str = self._build_prompt(search_node)
#         else:
#             raise ValueError(f"Invalid leandojo_state for search_node: {type(search_node.leandojo_state)}")
        
#         # generate tactics by LLM
#         suggestions = await self._generate_tactics(tactic_state_str)
#         normalized_suggestions = self._normalize_scores(suggestions)
#         normalized_suggestions = sorted(normalized_suggestions, key=lambda x: x[2], reverse=True)

#         # try all the tactics
#         results = []
        
#         for tactic, score, norm_score in normalized_suggestions:
#             edge, proof_finished = await self._run_tactic(
#                 search_node, tactic, score, norm_score, priority_queue
#             )
            
#             if edge is not None:
#                 results.append(edge)

#             if proof_finished:
#                 break
        
#         # expand the search node
#         search_node.out_edges = results
#         self.num_expansions += 1
#         priority_queue.task_done()

#     @torch.no_grad()
#     async def _generate_tactics(self, tactic_state_str: str) -> List[Tuple[str, float]]:
#         start_time = time.time()
#         suggestions = await client.async_generate_tactic_sampling(
#             self.select_tac_gen(),
#             state=tactic_state_str,
#             num_samples=self.num_sampled_tactics,
#         )

#         suggestions = [(item['tactic'], item['score']) for item in suggestions['suggestions']]
        
#         if len(suggestions) == 0:
#             raise ModelEmptyOutputError("No tactic generated.")

#         self.model_elapsed_time += time.time() - start_time
#         return suggestions
    
#     async def _run_tactic(
#         self, 
#         search_node: UnsolvedNode, 
#         tactic: str, 
#         score: float, 
#         norm_score: float,
#         priority_queue: asyncio.PriorityQueue
#     ) -> Tuple[Optional[Edge], bool]:
#         # run tactic in Lean server
#         start_time = time.time()

#         leandojo_new_state = await asyncio.to_thread(
#             self.dojo.run_tac, search_node.leandojo_state, tactic
#         )
        
#         self.dojo_elapsed_time += time.time() - start_time # Tactics triggering timeout will not be counted

#         # create a new child node
#         depth = search_node.depth + 1
#         if leandojo_new_state.id not in self.nodes:
#             # proof finished
#             if isinstance(leandojo_new_state, ProofFinished):
#                 child_node = SolvedNode(leandojo_state=leandojo_new_state, depth=depth)
#             # invalid tactic
#             elif type(leandojo_new_state) in (LeanError, ProofGivenUp):
#                 child_node = InvalidNode(leandojo_state=leandojo_new_state, depth=depth)
#             # unsolved proof
#             else:
#                 assert isinstance(leandojo_new_state, TacticState), f"Expected TacticState, got{type(leandojo_new_state)}"
#                 child_node = UnsolvedNode(
#                     leandojo_state=leandojo_new_state,
#                     is_terminal=False,
#                     priority=search_node.priority + score,
#                     depth=depth,
#                 )
#                 try:
#                     child_node.value = (await client.async_get_score(leandojo_new_state.pp))["score"]
#                 except Exception as e:
#                     logging.error(f"🚨Failed to get score for new state: {e}\nlen:{len(leandojo_new_state.pp)}")
#                     child_node.value = -1.0
#                 priority_queue.put_nowait((-child_node.value, child_node))
#             self.nodes[leandojo_new_state.id] = child_node
#             edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=score, norm_score=norm_score)
#         else: 
#             assert isinstance(leandojo_new_state, TacticState), f"Expected TacticState, got{type(leandojo_new_state)}"
#             child_node = self.nodes[leandojo_new_state.id]
#             assert isinstance(child_node, UnsolvedNode), f"Expected UnsolvedNode, got {type(child_node)}"
#             child_node.depth = min(child_node.depth, depth)
#             if await asyncio.to_thread(child_node.is_descendant, search_node):
#                 edge = None
#             else:
#                 edge = Edge(src=search_node, dst=child_node, tactic=tactic, score=score, norm_score=norm_score)
#                 self.back_edges.append(edge)

#         if isinstance(child_node, UnsolvedNode) and edge is not None:
#             child_node.in_edges.append(edge)
#         elif isinstance(child_node, SolvedNode) or isinstance(child_node, InvalidNode):
#             child_node.in_edge = edge

#         return edge, isinstance(leandojo_new_state, ProofFinished)
    
#     def _build_prompt(
#         self,
#         search_node: UnsolvedNode,
#     ) -> str:
#         input_template = "[GOAL]\n{state}\n[PROOFSTEP]\n"
#         return input_template.format(state=search_node.leandojo_state.pp)

#     def _normalize_scores(
#         self,
#         suggestions: List[Tuple[str, float]]
#     ) -> List[Tuple[str, float, float]]:

#         exps = [math.exp(score) for _, score in suggestions]
#         total = sum(exps)
        
#         return [
#             (tactic, score, exp_score / total)
#             for (tactic, score), exp_score in zip(suggestions, exps)
#         ]

#     async def get_result(self, visualize: bool = True) -> Optional[SearchResult]:
#         if not self.thm or not self.root:
#             return None
        
#         # check if the search was successful
#         if self.root.status == Status.SOLVED:
#             proof = [e.tactic for e in await asyncio.to_thread(self.root.extract_proof)]
#             if visualize:
#                 self.success_edges = await asyncio.to_thread(collect_success_edges, self.root)
#         else:
#             proof = None
        
#         if visualize:
#             await asyncio.to_thread(
#                 visualize_proof_tree,
#                 list(self.nodes.values()),
#                 self.success_edges, 
#                 self.back_edges,
#                 self.result_save_path + "/visual",
#                 self.thm.name,
#                 ['simple','detail']
#             )

#         # box the result
#         result = SearchResult(
#             theorem=self.thm,
#             status=self.root.status,
#             proof=proof,
#             num_total_nodes=len(self.nodes),
#             num_expansions=self.num_expansions,
#             elapsed_time=self.elapsed_time,
#             dojo_elapsed_time=self.dojo_elapsed_time,
#             model_elapsed_time=self.model_elapsed_time,
#         )

#         return result