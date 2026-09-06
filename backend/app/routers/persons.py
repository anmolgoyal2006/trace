"""
persons.py — /api/persons
CRUD for registered persons + reference photo enrollment + image-based search.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models.orm import Person
from backend.app.models.schemas import (
    PersonCreate,
    PersonOut,
    PersonSearchResult,
    PersonUpdate,
)
from backend.app.services.embedding_service import embedding_service

router = APIRouter(prefix="/api/persons", tags=["persons"])

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """Save an uploaded file to dest_dir with a unique name. Returns the saved path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename).suffix.lower() if upload.filename else ".jpg"
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Allowed: {_ALLOWED_EXTENSIONS}",
        )
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = dest_dir / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("", response_model=list[PersonOut])
async def list_persons(
    watchlist_only: bool = False,
    db: AsyncSession = Depends(get_db),
) -> list[PersonOut]:
    """List all active persons, optionally filtered to watchlist entries."""
    q = select(Person).where(Person.is_active == True)
    if watchlist_only:
        q = q.where(Person.watchlist_status != "none")
    result = await db.execute(q)
    persons = result.scalars().all()
    return [PersonOut.from_orm_with_embedding(p) for p in persons]


# ---------------------------------------------------------------------------
# Get one
# ---------------------------------------------------------------------------

@router.get("/{person_id}", response_model=PersonOut)
async def get_person(
    person_id: int,
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    person = await db.get(Person, person_id)
    if not person or not person.is_active:
        raise HTTPException(status_code=404, detail="Person not found")
    return PersonOut.from_orm_with_embedding(person)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@router.post("", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def create_person(
    name: str = Form(...),
    alias: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    watchlist_status: str = Form(default="none"),
    photo: Optional[UploadFile] = File(default=None),
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    """
    Register a new person.  Optionally accepts a reference photo which
    will be immediately enrolled (OSNet embedding generated).
    """
    if watchlist_status not in ("none", "missing", "suspect", "poi"):
        raise HTTPException(status_code=400, detail="Invalid watchlist_status")

    person = Person(
        name=name,
        alias=alias,
        description=description,
        watchlist_status=watchlist_status,
    )
    db.add(person)
    await db.flush()  # get person.id

    if photo is not None:
        saved = _save_upload(photo, settings.uploads_dir / "reference_photos")
        person.reference_image_path = str(saved)
        if embedding_service.is_ready:
            try:
                vec = embedding_service.embed_single(saved)
                person.embedding_vector = vec
                logger.info(f"[persons] Enrolled embedding for person {person.id}")
            except Exception as e:
                logger.warning(f"[persons] Embedding failed for person {person.id}: {e}")

    await db.commit()
    return PersonOut.from_orm_with_embedding(person)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

@router.put("/{person_id}", response_model=PersonOut)
async def update_person(
    person_id: int,
    payload: PersonUpdate,
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    person = await db.get(Person, person_id)
    if not person or not person.is_active:
        raise HTTPException(status_code=404, detail="Person not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(person, field, value)
    await db.commit()
    return PersonOut.from_orm_with_embedding(person)


# ---------------------------------------------------------------------------
# Delete (soft)
# ---------------------------------------------------------------------------

@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(
    person_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    person = await db.get(Person, person_id)
    if not person or not person.is_active:
        raise HTTPException(status_code=404, detail="Person not found")
    person.is_active = False
    await db.commit()


# ---------------------------------------------------------------------------
# Enroll reference photo
# ---------------------------------------------------------------------------

@router.post("/{person_id}/enroll", response_model=PersonOut)
async def enroll_photo(
    person_id: int,
    photo: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    """
    Upload a reference photo and generate an OSNet embedding for this person.
    Replaces any existing embedding.
    """
    person = await db.get(Person, person_id)
    if not person or not person.is_active:
        raise HTTPException(status_code=404, detail="Person not found")

    if not embedding_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded. Server is still starting up.",
        )

    saved = _save_upload(photo, settings.uploads_dir / "reference_photos")
    person.reference_image_path = str(saved)

    try:
        vec = embedding_service.embed_single(saved)
        person.embedding_vector = vec
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    await db.commit()
    logger.info(f"[persons] Re-enrolled person {person_id}")
    return PersonOut.from_orm_with_embedding(person)


# ---------------------------------------------------------------------------
# Image-based search
# ---------------------------------------------------------------------------

@router.post("/search", response_model=list[PersonSearchResult])
async def search_by_image(
    photo: UploadFile = File(...),
    top_k: int = 5,
    db: AsyncSession = Depends(get_db),
) -> list[PersonSearchResult]:
    """
    Upload a photo and find matching registered persons by OSNet similarity.
    Returns up to top_k results above the no-match threshold.
    """
    if not embedding_service.is_ready:
        raise HTTPException(status_code=503, detail="Embedding model not loaded.")

    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=".jpg", dir=settings.uploads_dir, delete=False
    ) as tmp:
        shutil.copyfileobj(photo.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        query_vec = embedding_service.embed_single(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Load all persons with enrolled embeddings
    result = await db.execute(
        select(Person).where(
            Person.is_active == True,
            Person.reference_embedding.is_not(None),
        )
    )
    persons = result.scalars().all()

    if not persons:
        return []

    from ai_pipeline.reid.similarity import cosine_similarity
    from ai_pipeline.reid.confidence_scaling import similarity_to_confidence

    scored: list[tuple[Person, float]] = []
    for p in persons:
        vec = p.embedding_vector
        if not vec:
            continue
        try:
            sim = cosine_similarity(query_vec, vec)
        except Exception:
            continue
        if sim >= settings.no_match_threshold:
            scored.append((p, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    return [
        PersonSearchResult(
            person=PersonOut.from_orm_with_embedding(p),
            similarity=round(sim, 6),
            confidence=round(similarity_to_confidence(sim), 1),
        )
        for p, sim in top
    ]
