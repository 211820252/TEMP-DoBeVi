import math
import json
from enum import Enum
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Union, Set, Dict, Tuple

from dojo import (
    TacticState,
    ProofFinished,
    LeanError,
    ProofGivenUp,
)

class Status(Enum):
    """
    Status of a node in the search tree.
    """
    SOLVED = 0
    UNSOLVED = 1
    INVALID = 2

class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @property
    def id(self):
        return self.leandojo_state.id
    
@dataclass
class SolvedNode(Node):
    """
    A node in the search tree that has been solved.
    """
    leandojo_state: ProofFinished
    in_edge: Optional['Edge'] = None 
    status: Status = Status.SOLVED
    distance_to_proof: float = 0.0
    is_terminal: bool = True
    depth: Optional[int] = None

@dataclass
class InvalidNode(Node):
    """
    A node in the search tree that is invalid.
    """
    leandojo_state: Union[LeanError, ProofGivenUp]
    in_edge: Optional['Edge'] = None 
    status: Status = Status.INVALID
    distance_to_proof: float = math.inf
    is_terminal: bool = True
    depth: Optional[int] = None
    
@total_ordering
@dataclass(unsafe_hash=True)
class UnsolvedNode(Node):
    """
    A node in the search tree that has not been solved.
    """
    leandojo_state: TacticState = field(compare=True)
    status: Status = field(default=Status.UNSOLVED, compare=False, repr=True)
    is_terminal: bool = False
    priority: float = field(default=0.0, compare=False)
    depth: Optional[int] = field(default=None, compare=False, repr=False)

    in_edges: List['Edge'] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    _out_edges: Optional[List['Edge']] = field(
        default=None, init=False, compare=False, repr=False
    )

    _distance_to_proof: float = field(default=math.inf, init=False, compare=False, repr=False)

    success_edges_list =[]

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    @property
    def out_edges(self) -> Optional[List['Edge']]:
        return self._out_edges
    
    @out_edges.setter
    def out_edges(self, edges: Iterable['Edge']) -> None:
        self._out_edges = list(edges)
        self._recompute_status()
        self._recompute_distance_to_proof()

    def _recompute_status(self):
        assert self.out_edges is not None
        # recompute current node status
        if self.status != Status.UNSOLVED:
            return
        if any(edge.dst.status == Status.SOLVED for edge in self.out_edges):
            self.status = Status.SOLVED
        if all(edge.dst.status == Status.INVALID for edge in self.out_edges):
            self.status = Status.INVALID

        # update the status of the parent node
        if self.status != Status.UNSOLVED:
            for edge in self.in_edges:
                edge.src._recompute_status()

    def _recompute_distance_to_proof(self):
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()


    def __lt__(self, other: "UnsolvedNode") -> bool:
        return self.priority < other.priority

    def extract_proof(self) -> Optional[List["Edge"]]:
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.SOLVED:
            return None
        
        # return []

        proving_path = min(
            self.out_edges,
            key=Edge.distance_to_proof
        )

        if proving_path.dst.is_terminal:
            assert isinstance(proving_path.dst, SolvedNode)
            return [proving_path]
        else:
            assert isinstance(proving_path.dst, UnsolvedNode)
            remaining_proof = proving_path.dst.extract_proof()
            return [proving_path] + remaining_proof
    
    def is_descendant(self, potential_descendant: Node) -> bool:
        assert isinstance(potential_descendant, UnsolvedNode), f"potential_descendant should be UnsolvedNode, got {type(potential_descendant)}"
        if self == potential_descendant:
            return True
        if isinstance(self, UnsolvedNode) and self.out_edges:
            for edge in self.out_edges:
                if isinstance(edge.dst, UnsolvedNode) and edge.dst.is_descendant(potential_descendant):
                    return True
        return False
    
@dataclass(eq=False)
class Edge:
    """
    An edge in the search tree, representing a tactic.
    """
    src: Node = field(repr=False)
    dst: Node = field(repr=False)
    tactic: str = ""
    score: float = float('-inf')
    norm_score: float = 0.00

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof
    
    def __repr__(self) -> str:
        src_id = getattr(self.src, 'id', 'unknown')
        dst_id = getattr(self.dst, 'id', 'unknown')
        return f"Edge({src_id} -> {dst_id})"
    
def print_search_tree(root: Node) -> str:
    lines = []
    visited = set()

    def dfs(node: Node, depth: int):
        indent = "  " * depth
        node_id = node.id

        if node_id in visited:
            lines.append(f"{indent}🔁 Node (revisited)")
            return
        visited.add(node_id)

        # 打印当前节点
        if isinstance(node, UnsolvedNode):
            lines.append(f"{indent}📍 UnsolvedNode: {str(node.id)}")
            if node.out_edges:
                for edge in node.out_edges:
                    tactic_str = edge.tactic
                    lines.append(f"{indent}  ├─🔧 Tactic: {tactic_str}")
                    dfs(edge.dst, depth + 2)
        elif isinstance(node, SolvedNode):
            lines.append(f"{indent}✅ SolvedNode:{str(node.id)}")
        elif isinstance(node, InvalidNode):
            lines.append(f"{indent}❌ InvalidNode: {str(node.id)}")
        else:
            lines.append(f"{indent}❓ Unknown Node Type")

    dfs(root, 0)
    return "\n".join(lines)

def collect_success_edges(root_node: Node) -> List['Edge']:
    success_edges: List['Edge'] = []
    edge_path: List['Edge'] = []

    def dfs(node: Node):
        if isinstance(node, SolvedNode):
            for edge in edge_path:
                if edge not in success_edges:
                    success_edges.append(edge)
            return
        if isinstance(node, UnsolvedNode) and node.out_edges:
            for edge in node.out_edges:
                edge_path.append(edge)
                dfs(edge.dst)
                edge_path.pop()

    dfs(root_node)
    return success_edges

def check_is_DAG(root_node: Node) -> bool:
    visited: Set[int] = set()
    rec_stack: Set[int] = set()

    def dfs(node: Node) -> bool:
        node_id = node.id
        if node_id in rec_stack:
            return False  # Cycle detected
        if node_id in visited:
            return True  # Already processed

        visited.add(node_id)
        rec_stack.add(node_id)

        if isinstance(node, UnsolvedNode) and node.out_edges:
            for edge in node.out_edges:
                if not dfs(edge.dst):
                    return False

        rec_stack.remove(node_id)
        return True

    return dfs(root_node)

def save_tree(root: Node, repo_path: str, file_path: str, theorem_name: str) -> Dict:
    """
    Save the tree structure to a JSON file.

    Args:
        root (Node): The root node of the tree.
        file_path (str): The file path where the tree will be saved.
    """
    if not check_is_DAG(root):
        raise ValueError("The search tree must be a DAG.")
    # assert root.status == Status.SOLVED, "The root node must be solved."

    nodes_data = []
    edges_data = []
    visited = set()

    def dfs(node: Node) -> None:
        node_id = node.id
        if node_id in visited:
            return
        visited.add(node_id)

        # Serialize node data
        node_data = {
            "id": node_id,
            "type": node.__class__.__name__,
            "status": node.status.name,
            "is_terminal": node.is_terminal,
            "depth": node.depth,
            "tactic_state": node.leandojo_state.pp if isinstance(node, UnsolvedNode) else node.leandojo_state.__class__.__name__,
        }

        nodes_data.append(node_data)

        if isinstance(node, UnsolvedNode) and node.out_edges:
            for edge in node.out_edges:
                edge_data = {
                    "src": edge.src.id,
                    "dst": edge.dst.id,
                    "tactic": edge.tactic,
                    "score": edge.score,
                    "norm_score": edge.norm_score,
                }
                edges_data.append(edge_data)
                dfs(edge.dst)

    dfs(root)

    # Combine nodes and edges into a single dictionary
    tree_data = {
        "repo_path": repo_path,
        "file_path": file_path,
        "theorem_name": theorem_name,
        "nodes": nodes_data,
        "edges": edges_data,
    }

    return tree_data

def load_tree(json_path: str) -> Tuple[Node, str, str, str]:
    """
    Load the tree structure from a JSON file.

    Args:
        json_path (str): The file path from which the tree will be loaded.

    Returns:
        Node: The root node of the loaded tree.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    nodes_dict = {}
    for node_data in tree_data["nodes"]:
        assert "type" in node_data, "Node data must contain 'type' field."
        assert node_data["id"] not in nodes_dict, f"Duplicate node id found: {node_data['id']}"

        node_type = node_data["type"]
        if node_type == "UnsolvedNode":
            node = UnsolvedNode(
                leandojo_state=TacticState(pp=node_data["tactic_state"], id=node_data["id"]),
                status=Status[node_data["status"]],
                is_terminal=node_data["is_terminal"],
                depth=node_data["depth"]
            )
        elif node_type == "SolvedNode":
            node = SolvedNode(
                leandojo_state=ProofFinished(id=node_data["id"]),
                status=Status[node_data["status"]],
                is_terminal=node_data["is_terminal"],
                depth=node_data["depth"]
            )
        elif node_type == "InvalidNode":
            node = InvalidNode(
                leandojo_state=LeanError(id=node_data["id"], error=''),
                status=Status[node_data["status"]],
                is_terminal=node_data["is_terminal"],
                depth=node_data["depth"]
            )
        else:
            raise ValueError(f"Unknown node type: {node_type}")

        nodes_dict[node_data["id"]] = node

    for edge_data in tree_data["edges"]:
        src_node = nodes_dict[edge_data["src"]]
        dst_node = nodes_dict[edge_data["dst"]]
        edge = Edge(
            src=src_node,
            dst=dst_node,
            tactic=edge_data["tactic"],
            score=edge_data["score"],
            norm_score=edge_data["norm_score"]
        )
        if isinstance(src_node, UnsolvedNode):
            if src_node.out_edges is None:
                src_node.out_edges = []
            src_node.out_edges.append(edge)
            if dst_node.is_terminal:
                assert isinstance(dst_node, (SolvedNode, InvalidNode))
                dst_node.in_edge = edge
            else:
                assert isinstance(dst_node, UnsolvedNode)
                dst_node.in_edges.append(edge)
    
    assert not nodes_dict[0].in_edges, "The root node should not have any incoming edges."

    if 'repo_path' not in tree_data or 'file_path' not in tree_data or 'theorem_name' not in tree_data:
        return nodes_dict[0], "", "", ""
        
    return nodes_dict[0], tree_data["repo_path"], tree_data["file_path"], tree_data["theorem_name"]

def get_data_for_grpo(json_path: str) -> List[Dict]:
    root, repo_path, file_path, theorem_name = load_tree(json_path)
    visited = set()
    grpo_data = []

    def set_node_path(u: Node) -> None:
        if u.id in visited:
            return
        visited.add(u.id)

        assert isinstance(u, UnsolvedNode) and u.out_edges is not None
        for edge in u.out_edges:
            v = edge.dst
            if isinstance(v, UnsolvedNode) and v.out_edges is not None:
                v.path = u.path + [edge.tactic]
                set_node_path(v)

    root.path = []
    set_node_path(root)
    visited.clear()

    def compute_reward(node: Node) -> float:
        if hasattr(node, 'reward') and node.reward is not None:
            assert node.num_child_types is not None
            return node.reward # Reward already computed

        if isinstance(node, InvalidNode):
            node.num_child_types = 0
            node.reward = -1.0
        elif isinstance(node, SolvedNode):
            node.num_child_types = 0
            node.reward = 1.0
        elif isinstance(node, UnsolvedNode):
            if node.out_edges is None or len(node.out_edges) == 0:
                node.num_child_types = 0
                node.reward = 0.0
            else:
                num = len(node.out_edges)
                total_reward = 0.0
                types = set()
                node.rewards = []
                node.num_solved_children = 0
                node.num_invalid_children = 0
                node.num_unsolved_children = 0
                for edge in node.out_edges:
                    if isinstance(edge.dst, UnsolvedNode):
                        node.num_unsolved_children += 1
                    elif isinstance(edge.dst, SolvedNode):
                        node.num_solved_children += 1
                    elif isinstance(edge.dst, InvalidNode):     
                        node.num_invalid_children += 1
                    types.add(edge.dst.__class__.__name__)
                    total_reward += compute_reward(edge.dst)
                    node.rewards.append(edge.dst.reward)
                node.num_child_types = len(types)
                node.reward = total_reward / num
        else:
            node.num_child_types = 0
            node.reward = 0.0  # Default case, should not happen
    
        return node.reward

    compute_reward(root)

    child_types_num_threshold = 2
    child_num_threshold = 4
    reward_range = (-1, 1)

    def extract_data_for_grpo(node: Node) -> None:
        if node.id in visited:
            return
        visited.add(node.id)

        assert isinstance(node, UnsolvedNode) and node.out_edges is not None

        if len(node.out_edges) >= child_num_threshold and reward_range[0] <= node.reward <= reward_range[1]: # and node.status == Status.SOLVED:
            child_types_num = len(set(edge.dst.__class__.__name__ for edge in node.out_edges))
            if child_types_num >= child_types_num_threshold:
                edge_data = []
                for edge in node.out_edges:
                    edge_info = {
                        'tactic': edge.tactic,
                        'state_after': edge.dst.leandojo_state.pp if isinstance(edge.dst, UnsolvedNode) else edge.dst.leandojo_state.__class__.__name__,
                        'child_type': edge.dst.__class__.__name__,
                        'child_status': edge.dst.status.name,
                        'child_reward': edge.dst.reward
                    }
                    edge_data.append(edge_info)

                grpo_data.append({
                    'repo_path': repo_path,
                    'file_path': file_path,
                    'theorem_name': theorem_name,
                    'id': node.id,
                    'path': node.path,
                    'self_reward': node.reward,
                    'self_status': node.status.name,
                    'num_child_types': child_types_num,
                    'state_before': node.leandojo_state.pp,
                    'data': edge_data
                })
            for edge in node.out_edges:
                if isinstance(edge.dst, UnsolvedNode) and edge.dst.out_edges is not None:
                    extract_data_for_grpo(edge.dst)

    extract_data_for_grpo(root)
    
    return grpo_data

#  num_child_types: 子节点类型的数量，默认是2
def choose_good_nodes(root: Node, theorem_id: int) -> List[Dict]:
    """
    遍历树，找到所有“好节点”（子节点有两种类型的节点），并记录其 tactic_state。

    Args:
        root (Node): 树的根节点。

    Returns:
        List[Dict]: 包含所有好节点的 tactic_state 的列表。
    """
    visited = set() 
    good_nodes = []

    def dfs(node: Node):
        # 仅处理 UnsolvedNode 类型的节点
        if node.id in visited:
            return
        visited.add(node.id)
        if isinstance(node, UnsolvedNode) and node.out_edges:
            # 统计子节点的类型
            child_types = set(edge.dst.__class__.__name__ for edge in node.out_edges)

            # 如果子节点类型有两种或以上，则认为是好节点
            if len(child_types) >= 2:
                good_nodes.append({
                    "id": node.id,
                    "theorem_id": theorem_id,
                    "tactic_state": node.leandojo_state.pp,
                    "child_types": list(child_types)  # 记录子节点的类型
                })

        # 递归遍历子节点
        if isinstance(node, UnsolvedNode) and node.out_edges:
            for edge in node.out_edges:
                dfs(edge.dst)

    # 从根节点开始遍历
    dfs(root)

    return good_nodes



