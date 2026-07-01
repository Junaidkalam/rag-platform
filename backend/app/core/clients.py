from functools import lru_cache

from google import genai
from pinecone import Pinecone

from app.core.config import get_settings


@lru_cache
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


@lru_cache
def get_pinecone_index():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [i.name for i in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=3072,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": settings.pinecone_environment}},
        )

    return pc.Index(settings.pinecone_index_name)