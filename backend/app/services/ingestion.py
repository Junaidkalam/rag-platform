import fitz
import httpx
import re
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.clients import get_gemini_client, get_pinecone_index
from app.core.config import get_settings
from app.models.db_models import Document

log = structlog.get_logger()
settings = get_settings()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc):
        if page_num >= settings.max_pages_per_doc:
            break
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


async def extract_text_from_url(url: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url, headers={"User-Agent": "RAGPlatform/1.0"})
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def chunk_text(text: str) -> list[str]:
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_gemini_client()
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=batch,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        all_embeddings.extend([e.values for e in result.embeddings])
    return all_embeddings


def upsert_to_pinecone(
    document_id: str,
    workspace_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    index = get_pinecone_index()
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{document_id}__chunk_{i}",
            "values": embedding,
            "metadata": {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "chunk_index": i,
                "text": chunk[:1000],
            },
        })
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])


async def ingest_document(
    db: AsyncSession,
    document_id: str,
    text: str,
) -> dict:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {document_id} not found")

    try:
        doc.status = "processing"
        await db.commit()

        chunks = chunk_text(text)
        log.info("chunked", document_id=document_id, chunks=len(chunks))

        embeddings = await embed_texts(chunks)
        token_count = sum(len(c.split()) for c in chunks)

        upsert_to_pinecone(
            document_id=document_id,
            workspace_id=doc.workspace_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        doc.status = "ready"
        doc.chunk_count = len(chunks)
        doc.token_count = token_count
        await db.commit()

        log.info("ingestion_complete", document_id=document_id, chunks=len(chunks))
        return {"chunks": len(chunks), "tokens": token_count}

    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
        await db.commit()
        log.error("ingestion_failed", document_id=document_id, error=str(e))
        raise


async def delete_document_vectors(document_id: str) -> None:
    index = get_pinecone_index()
    index.delete(filter={"document_id": {"$eq": document_id}})