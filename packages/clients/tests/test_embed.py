from clients.embed import EmbeddingClient


def test_embed() -> None:
    client = EmbeddingClient(
        model_id="Qwen/Qwen3-Embedding-8B", base_url="http://localhost:8000/v1"
    )
    mentions = ["Hello this is a test", "And so is this!"]
    embeddings = client.embed(mentions=mentions)
    print(embeddings)
