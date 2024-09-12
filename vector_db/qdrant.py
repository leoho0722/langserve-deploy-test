import os
from langchain_community.vectorstores.qdrant import Qdrant


def get_qdrant_client(chunks, embeddings_model):
    qdrant_cloud_endpoint = os.environ["QDRANT_ENDPOINT"]
    qdrant = Qdrant.from_documents(
        chunks,
        embeddings_model,
        url=f"{qdrant_cloud_endpoint}:6333",
        api_key=os.environ["QDRANT_API_KEY"],
        collection_name="subsidy_qa",
        force_recreate=True,
    )

    return qdrant
