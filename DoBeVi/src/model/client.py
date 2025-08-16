import asyncio
import logging
import json
from urllib.parse import urljoin, urlparse, urlunparse
from loguru import logger
import aiohttp
from typing import Dict, List
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

class GeneratorClient:
    url: str

    def __init__(self, base_url: str) -> None:
        self.url = base_url

    def generate_tactic(self, gpu_id: int, state: str, num_samples: int) -> Dict:
        """
        Generate tactics based on the provided state and number of samples.

        Args:
            gpu_id (int): The ID of the GPU to use for generation.
            state (str): The state to generate tactics from.
            num_samples (int): The number of samples to generate.

        Returns:
            Dict: A dictionary containing the generated tactics.
        """
        return asyncio.run(self.async_generate_tactic(gpu_id, state, num_samples))

    async def async_generate_tactic(self, gpu_id: int, state: str, num_samples: int) -> Dict:
        """
        Asynchronously generate tactics based on the provided state and number of samples.

        Args:
            gpu_id (int): The ID of the GPU to use for generation.
            state (str): The state to generate tactics from.
            num_samples (int): The number of samples to generate.

        Returns:
            Dict: A dictionary containing the generated tactics.
        """
        return await self._query(
            "post",
            "/generate_tactic",
            json_data={
                "gpu_id": gpu_id,
                "state": state,
                "num_samples": num_samples
            }
        )

    def generate_tactic_sampling(self, gpu_id: int, state: str, num_samples: int) -> Dict:
        """
        Generate tactics with sampling based on the provided state and number of samples.

        Args:
            gpu_id (int): The ID of the GPU to use for generation.
            state (str): The state to generate tactics from.
            num_samples (int): The number of samples to generate.

        Returns:
            Dict: A dictionary containing the generated tactics.
        """
        return asyncio.run(self.async_generate_tactic_sampling(gpu_id, state, num_samples))

    async def async_generate_tactic_sampling(self, gpu_id: int, state: str, num_samples: int) -> Dict:
        """
        Asynchronously generate tactics with sampling based on the provided state and number of samples.

        Args:
            gpu_id (int): The ID of the GPU to use for generation.
            state (str): The state to generate tactics from.
            num_samples (int): The number of samples to generate.

        Returns:
            Dict: A dictionary containing the generated tactics.
        """
        return await self._query(
            "post",
            "/generate_tactic_sampling",
            json_data={
                "gpu_id": gpu_id,
                "state": state,
                "num_samples": num_samples
            }
        )

    def generate_proof(self, gpu_id: int, theorem_name: str, init_state: List[Dict[str, str]], num_samples: int, output_dir: str) -> Dict: 
        return asyncio.run(self.async_generate_proof(gpu_id, theorem_name, init_state, num_samples, output_dir))

    async def async_generate_proof(self, gpu_id: int, theorem_name: str, init_state: List[Dict[str, str]], num_samples: int, output_dir: str) -> Dict:
        return await self._query(
            "post",
            "/generate_proof",
            json_data={
                "gpu_id": gpu_id,
                "theorem_name": theorem_name,
                "init_state": json.dumps(init_state),
                "num_samples": num_samples,
                "output_dir": output_dir
            }
        )
    
    def generate_proof_sampling(self, gpu_id: int, theorem_name: str, init_state: List[Dict[str, str]], num_samples: int, output_dir: str) -> Dict: 
        return asyncio.run(self.async_generate_proof_sampling(gpu_id, theorem_name, init_state, num_samples, output_dir))

    async def async_generate_proof_sampling(self, gpu_id: int, theorem_name: str, init_state: List[Dict[str, str]], num_samples: int, output_dir: str) -> Dict:
        return await self._query(
            "post",
            "/generate_proof_sampling",
            json_data={
                "gpu_id": gpu_id,
                "theorem_name": theorem_name,
                "init_state": json.dumps(init_state),
                "num_samples": num_samples,
                "output_dir": output_dir
            }
        )

    def get_score(self, state: str) -> Dict:
        """
        Get the score for a given state using the value network.

        Args:
            state (str): The state to evaluate.

        Returns:
            Dict: A dictionary containing the score.
        """
        return asyncio.run(self.async_get_score(state))
    
    async def async_get_score(self, state: str) -> Dict:
        """
        Asynchronously get the score for a given state using the value network.

        Args:
            state (str): The state to evaluate.

        Returns:
            Dict: A dictionary containing the score.
        """
        return await self._query(
            "post",
            "/value_network_score",
            json_data={"state": state}
        )

    async def _query(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        n_retries: int = 3,
    ) -> dict:
        """
        One single method for sending all requests, with retry behavior controlled by the caller.

        Args:
            method: The HTTP method to use (e.g., "get", "post").
            endpoint: The endpoint to call.
            json_data: The data to send in the request.
            n_retries: Number of retry attempts.

        Returns:
            response: The response from the server.
        """

        # Create retry decorator with dynamic n_retries
        @retry(
            stop=stop_after_attempt(
                n_retries
            ),  # Dynamic retries based on the caller's argument
            wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
            before_sleep=before_sleep_log(
                logger, logging.ERROR
            ),  # Optional logging of each retry
        )
        async def query_with_retries(url):
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Create a session with trust_env set to True
            async with aiohttp.ClientSession(
                trust_env=True, timeout=aiohttp.ClientTimeout(total=1800)
            ) as session:
                async with session.request(
                    method,
                    self._ensure_url_has_scheme(str(urljoin(url, endpoint))),
                    headers=headers,
                    json=json_data,  # Directly send the JSON data
                ) as response:
                    # Get the response body asynchronously and parse it as JSON
                    res = await response.json()

            return res

        # Call the query function with retries
        return await query_with_retries(self.url)

    def _ensure_url_has_scheme(self, url, default_scheme="http"):
        """Ensure URL has a scheme (http/https) prefix.

        Args:
            url (str): The URL to check and potentially modify.
            default_scheme (str, optional): The scheme to add if none exists. Defaults to "http".

        Returns:
            str: URL with a scheme.
        """
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f"{default_scheme}://{url}")
        return urlunparse(parsed)

    async def _test_connection(self):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # 已经在 event loop 里
                response = await self._query("get", "/")
            else:
                # 普通脚本里
                response = asyncio.run(self._query("get", "/"))
        except RetryError as e:
            raise Exception(f"The server {self.url} cannot be connected:{e}")

        if response.get("status") != "ok":
            raise Exception(
                f"The server {self.url} cannot be available. {response}"
            )