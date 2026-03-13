from dataclasses import dataclass
import numpy as np
from loguru import logger
from numpy._typing import NDArray
from openai import OpenAI


@dataclass
class EmbeddingClient:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-8B",
        base_url: str = "http://localhost:8000/v1",
    ):
        logger.info(f"Configuring Embedding Client for {model_id}")
        self.model_id = model_id
        self.base_url = base_url
        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
        self.prompt = "Instruct: Represent the entity mention for retrieval.\nMention: "

    def embed(self, mentions: list[str]) -> list[NDArray[np.float32]]:
        """
        Generate embeddings for a list of mentions using the vLLM hosted model.
        :param mentions: List of strings to embed
        :return: List of embeddings (each embedding is an NDArray[float32])
        """

        logger.info(f"Generating embeddings for {len(mentions)} mentions")

        queries = [f"{self.prompt}{mention}" for mention in mentions]

        embedding_response = self.client.embeddings.create(
            model=self.model_id,
            input=queries,
            encoding_format="float",
        )

        logger.info(f"Generated embeddings for {len(mentions)} mentions")

        return [
            np.array(embedding.embedding, dtype=np.float32)
            for embedding in embedding_response.data
        ]
