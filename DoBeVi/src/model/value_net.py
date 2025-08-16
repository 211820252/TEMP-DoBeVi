from transformers import (
    AutoTokenizer,
    AutoModel
)
import torch
import ray

@ray.remote(num_gpus=1)
class ValueNetwork:
    def __init__(self, model_path: str, max_length: int = 4096):
        torch.cuda.set_per_process_memory_fraction(0.9)

        self.model = AutoModel.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_length = max_length

    @torch.no_grad()
    def get_score(self, chat: list) -> float:
        if sum(len(item["content"]) for item in chat) > self.max_length:
            return -100000.0
        score = self.model.get_score(self.tokenizer, chat)
        return score

