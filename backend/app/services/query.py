import json
from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.clients import get_gemini_client, get_pinecone_index
from app.core.config import get_settings
from app.models.db_models import Document, QueryLog
from app.models.schemas import SourceChunk

log = structlog.get_logger()
settings = get_settings()

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based strictly on the provided context documents.

Rules:
- Answer ONLY from the provided context. Never add information from outside.
- If the context doesn't contain enough information, say so clearly.
- Cite which document(s) your answer is based on.
- Be concise and precise.
- Use markdown formatting in your response."""


async def embed_query(question: str) -> list[float]:
    client = get_gemini_client()
    result = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=question,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    return result.embeddings[0].values


async def retrieve_chunks(
    question_embedding: list[float],
    workspace_id: str,
    top_k: int = 5,
) -> list[dict]:
    index = get_pinecone_index()
    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        filter={"workspace_id": {"$eq": workspace_id}},
        include_metadata=True,
    )
    return results.matches


async def build_context(
    chunks: list[dict],
    db: AsyncSession,
) -> tuple[str, list[SourceChunk]]:
    doc_ids = list({c.metadata["document_id"] for c in chunks})
    result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
    docs = {doc.id: doc for doc in result.scalars().all()}

    context_parts = []
    source_chunks = []

    for match in chunks:
        doc_id = match.metadata["document_id"]
        doc = docs.get(doc_id)
        title = doc.title if doc else "Unknown document"
        text = match.metadata.get("text", "")
        context_parts.append(f"[Source: {title}]\n{text}")
        source_chunks.append(SourceChunk(
            document_id=doc_id,
            document_title=title,
            chunk_text=text,
            score=float(match.score),
        ))

    return "\n\n---\n\n".join(context_parts), source_chunks


async def query_rag(
    db: AsyncSession,
    question: str,
    workspace_id: str,
    user_id: str,
    top_k: int = 5,
) -> tuple[str, list[SourceChunk], int, int]:
    q_embedding = await embed_query(question)
    chunks = await retrieve_chunks(q_embedding, workspace_id, top_k)

    if not chunks:
        return "I couldn't find any relevant information in your documents.", [], 0, 0

    context, sources = await build_context(chunks, db)

    client = get_gemini_client()
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 1500,
        },
    )

    answer = response.text
    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count

    log_entry = QueryLog(
        workspace_id=workspace_id,
        user_id=user_id,
        question=question,
        answer=answer,
        source_doc_ids=json.dumps([s.document_id for s in sources]),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    db.add(log_entry)
    await db.commit()

    return answer, sources, prompt_tokens, completion_tokens


async def query_rag_stream(
    db: AsyncSession,
    question: str,
    workspace_id: str,
    user_id: str,
    top_k: int = 5,
) -> AsyncGenerator[str, None]:
    q_embedding = await embed_query(question)
    chunks = await retrieve_chunks(q_embedding, workspace_id, top_k)

    if not chunks:
        yield 'data: {"type":"error","content":"No relevant documents found."}\n\n'
        return

    context, sources = await build_context(chunks, db)

    sources_payload = [s.model_dump() for s in sources]
    yield f'data: {json.dumps({"type": "sources", "content": sources_payload})}\n\n'

    client = get_gemini_client()
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"

    full_answer = []
    response_stream = client.models.generate_content_stream(
        model=settings.gemini_chat_model,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 1500,
        },
    )

    for chunk in response_stream:
        token = chunk.text
        if token:
            full_answer.append(token)
            yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'

    yield 'data: {"type":"done"}\n\n'

    log_entry = QueryLog(
        workspace_id=workspace_id,
        user_id=user_id,
        question=question,
        answer="".join(full_answer),
        source_doc_ids=json.dumps([s.document_id for s in sources]),
    )
    db.add(log_entry)
    await db.commit()