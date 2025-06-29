import asyncio
import logging
import time
from typing import List, Optional, Tuple, Dict
import torch
import math

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
    TacticGenerator,
    ModelEmptyOutputError,
)

from search.visual import (
    visualize_proof_tree,
)


class InternlmBFSProver(Prover):
    def __init__(
        self,
        tactic_generators: List[TacticGenerator],
        actor_id: int,
        search_timeout: int,
        max_expansions: Optional[int],
        num_sampled_tactics: int,
        result_save_path: str,
    ):
        super().__init__(tactic_generators, actor_id, search_timeout, max_expansions, num_sampled_tactics, result_save_path)

    async def search(self, thm: TracedTheorem) -> None:
        """
        Search for a proof of a theorem using best-first search.
        """
        self._prepare()
    
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
            if hasattr(self, "dojo"):
                self.dojo.__exit__(None, None, None)
    
    async def _best_first_search(self) -> None:
        start_time = time.time()
        priority_queue = asyncio.PriorityQueue() # lowest priority first
        priority_queue.put_nowait((-self.root.priority, self.root, []))

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
            if (self.elapsed_time > self.search_timeout) or (self.max_expansions and self.num_expansions >= self.max_expansions):
                if self.root.status == Status.SOLVED:
                    logging.info("Search complete: proof found.")
                else:
                    logging.info("Exceeded time limit or max expansions.")
                break

            if self.root.status == Status.INVALID:
                logging.info("Invalid tactic generated.")
                break

            if self.root.status == Status.SOLVED:
                logging.info("Search complete: proof found.")
                break

    async def _step(self, priority_queue: asyncio.PriorityQueue[Tuple[float, UnsolvedNode, List]]) -> None:
        _, search_node, steps = priority_queue.get_nowait()
        
        logging.info(f"Expanding node: {search_node}")

        if isinstance(search_node.leandojo_state, TacticState):
            tactic_state_str = self._build_prompt(search_node, steps)
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
                search_node, tactic, steps, score, norm_score, priority_queue
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
        tac_gen = self.select_tac_gen()
        start_time = time.time()

        suggestions = await tac_gen.generate_sampling.remote(
            state=tactic_state_str,
            num_samples=self.num_sampled_tactics,
        )
        if len(suggestions) == 0:
            raise ModelEmptyOutputError("No tactic generated.")
        self.model_elapsed_time += time.time() - start_time
        return suggestions
    
    async def _run_tactic(
        self, 
        search_node: UnsolvedNode, 
        tactic: str, 
        steps: List[str],
        score: float,
        norm_score: float, 
        priority_queue: asyncio.PriorityQueue
    ) -> Tuple[Edge, bool]:
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
            elif type(leandojo_new_state) in (
                LeanError, ProofGivenUp
            ):
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
                priority_queue.put_nowait((-child_node.priority, child_node, steps + [tactic]))
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
        steps: List[str],
    ) -> str:
        proof_before = '\n'.join(steps)
        input_template = (  f"---\nNAME: {self.thm.name}\n\n"
                            f"---\nPROOF_BEFORE: {proof_before}\n\n"
                            f"---\nSTATE_BEFORE: {search_node.leandojo_state.pp}\n\n"
                            f"---\nTACTIC: "
                        )
        prompt_list = [{"role": "user", "content": input_template}]
        
        result = ""
        total_step = len(prompt_list)
        for i, message in enumerate(prompt_list):
            result += ('<|im_start|>' + message['role'] +
                    '\n' + message['content'])
            if i+1 != total_step:
                result += '<|im_end|>\n'
            elif message['role'] == 'user':
                result += '<|im_end|>\n<|im_start|>assistant\n'
        
        return result
    
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