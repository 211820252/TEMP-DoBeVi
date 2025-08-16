import asyncio
import logging
import time
from typing import List, Optional, Tuple, Union
import torch
import math
import queue

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
)

class MCTSProver(Prover):
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
            self.root.value = 0.0
            self.root.vis_cnt = 0

            self.nodes = {current_state.id: self.root}
            self.back_edges = []
            self.success_edges = []

            try:
                await self._mcts_search()
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

    async def _mcts_search(self) -> None:
        start_time = time.time()

        while True:
            try:
                await self._step()
            except (asyncio.TimeoutError, asyncio.CancelledError, DojoCrashError):
                raise
            except Exception as e:
                logging.error(f"🚨{type(e).__name__}: {e}")

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

        self.elapsed_time = time.time() - start_time

    def compute_ucb(self, node: UnsolvedNode) -> float:
        if not isinstance(node, UnsolvedNode):
            return 0.0
        if node.vis_cnt == 0:
            return float('inf')
        if not node.in_edges:
            return 0.0 
        N = min(edge.src.vis_cnt for edge in node.in_edges)
        if N <= 0:
            return 0.0
        return node.value / node.vis_cnt +  math.sqrt(math.log(N) / node.vis_cnt)

    # def _mcts_selection(self) -> Optional[UnsolvedNode]:
    #     """
    #     Select a node to expand using MCTS selection strategy.
    #     """
    #     try:
    #         cur_node = self.root
    #         while cur_node.out_edges:
    #             assert isinstance(cur_node, UnsolvedNode), f"Expected UnsolvedNode, got '{type(cur_node)}'"
    #             valid_out_edges = [edge for edge in cur_node.out_edges if isinstance(edge.dst, UnsolvedNode)]
    #             if not valid_out_edges:
    #                 self._mcts_back_propagation(cur_node, 0.0)
    #                 return None
    #             cur_node = max(valid_out_edges, key=lambda edge: self.compute_ucb(edge.dst)).dst
    #             assert isinstance(cur_node, UnsolvedNode), f"Expected UnsolvedNode, got '{type(cur_node)}'"
    #         return cur_node
            
    #     except (asyncio.TimeoutError, asyncio.CancelledError):
    #         raise
        
    #     except Exception as e:
    #         logging.error(f"🚨{type(e).__name__}: {e}")
    #         return None
    
    def _mcts_selection(self):
        try:
            best_node = None
            best_ucb  = float("-inf")

            stack: list = [self.root]
            visited: set = set()

            while stack:
                node = stack.pop()

                # skip already-seen nodes (DAG guarantees acyclic, but this prevents re-work)
                if node in visited:
                    continue
                visited.add(node)

                # --- push children (if any) for further traversal ------------------
                if getattr(node, "out_edges", None):     # ''None'' or [] are both false-y
                    for edge in node.out_edges:
                        child = edge.dst
                        if isinstance(child, UnsolvedNode) and child not in visited:
                            stack.append(child)

                # --- check whether this node is an Unsolved leaf -------------------
                if isinstance(node, UnsolvedNode) and not node.out_edges:
                    ucb = self.compute_ucb(node)
                    if ucb > best_ucb:
                        best_ucb  = ucb
                        best_node = node

            return best_node
            
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise
        
        except Exception as e:
            logging.error(f"🚨{type(e).__name__}: {e}")
            return None

    async def _mcts_expansion(self, search_node: UnsolvedNode) -> UnsolvedNode:
        """
        Expand the search node by adding new edges based on the tactics generated by the LLM.
        """
        assert isinstance(search_node, UnsolvedNode), f"Expected UnsolvedNode, got '{type(search_node)}'"
        
        if search_node.vis_cnt == 0:
            return search_node
        
        if isinstance(search_node.leandojo_state, TacticState):
            tactic_state_str = self._build_prompt(search_node)
        else:
            raise ValueError(f"Invalid leandojo_state for search_node: {type(search_node.leandojo_state)}")
        
        # generate tactics by LLM
        try: 
            suggestions = await self._generate_tactics(tactic_state_str)
            normalized_suggestions = self._normalize_scores(suggestions)
            normalized_suggestions = sorted(normalized_suggestions, key=lambda x: x[2], reverse=True)
        except ModelEmptyOutputError as e:
            logging.error(f"🚨{type(e).__name__}: {e}")
            self._mcts_back_propagation(search_node, 0.0)
            suggestions = []
            normalized_suggestions = []

        # try all the tactics
        results = []
        
        for tactic, score, norm_score in normalized_suggestions:
            edge, proof_finished = await self._run_tactic(
                search_node, tactic, score, norm_score
            )
            
            if edge is not None:
                results.append(edge)

        # expand the search node
        search_node.out_edges = results
        self.num_expansions += 1
        
        return next((e.dst for e in search_node.out_edges if isinstance(e.dst, Union[SolvedNode, UnsolvedNode])), None)

    async def _mcts_simulation(self, simulation_node: Union[SolvedNode, UnsolvedNode]) -> float:
        """
        Simulate the search node to get a value estimate.
        """
        assert isinstance(simulation_node, Union[SolvedNode, UnsolvedNode]), f"Expected UnsolvedNode or SolvedNode, got '{type(simulation_node)}'"
        
        if simulation_node.vis_cnt == 0:
            return 0.0
        
        # simulate the value of the node
        try:
            value = (await client.async_get_score(
                simulation_node.leandojo_state.pp if isinstance(simulation_node, UnsolvedNode) else "no goals"
            ))["score"]
            return value
        except Exception as e:
            logging.error(f"🚨{type(e).__name__}: {e}")
            return 0.0

    def _mcts_back_propagation(self, simulation_node: UnsolvedNode, value: float) -> None:
        """
        Back propagate the value from the simulation node to the search node.
        """
        assert isinstance(simulation_node, UnsolvedNode), f"Expected UnsolvedNode, got '{type(simulation_node)}'"
        
        # back propagate the value to the parent nodes
        cur_node = simulation_node
        while cur_node:
            cur_node.vis_cnt += 1
            cur_node.value += value
            if not cur_node.in_edges:
                break
            cur_node = min(cur_node.in_edges, key=lambda edge: edge.src.vis_cnt).src

    async def _step(self) -> None:
        search_node = self._mcts_selection()
        if search_node is None:
            logging.info("No valid search node found, skipping MCTS step.")
            return
        simulation_node = await self._mcts_expansion(search_node)
        if simulation_node is None:
            logging.info("No valid simulation node found, skipping MCTS step.")
            return
        value = await self._mcts_simulation(simulation_node)
        self._mcts_back_propagation(simulation_node, value)

    @torch.no_grad()
    async def _generate_tactics(self, tactic_state_str: str) -> List[Tuple[str, float]]:
        start_time = time.time()
        suggestions = await client.async_generate_tactic_sampling(
            self.select_tac_gen(),
            state=tactic_state_str,
            num_samples=self.num_sampled_tactics,
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
                child_node.vis_cnt = 0
                child_node.value = 0.0
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

    def _normalize_scores(
        self,
        suggestions: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, float]]:
        if not suggestions:
            return []

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