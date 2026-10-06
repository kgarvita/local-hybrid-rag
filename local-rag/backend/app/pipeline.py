"""Single executable ingestion pipeline.

Order is intentionally:
    extract -> normalize -> structure -> multimodal enrichment -> chunk -> embed/store

Structure is inferred from original extracted text/layout. OCR/Vision is supplemental
and therefore cannot accidentally manufacture section boundaries.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid

from .chunking import build_parent_child_chunks
from .config import CHILD_CHUNK_SIZE, DOCUMENTS_DIR, OUTPUT_DIR, EMBEDDING_MODEL, PARENT_CHUNK_SIZE
from .enrich_extraction import enrich_document
from .extractor import extract_pdf
#from .ingest import ensure_schema, get_connection, upsert_document_and_chunks
from .normalize import normalize_document
from .sectionize import build_section_hierarchy
from .utils import banner, log, save_json
from .ingest import (
    ensure_schema,
    get_connection,
    upsert_document_and_chunks,
    delete_missing_documents
)


def process_document(
    pdf_path: str,
    *,
    output_dir: str,
    force: bool,
    skip_vision: bool,
    conn,
) -> dict:
    filename = os.path.basename(pdf_path)
    doc_output = os.path.join(output_dir, os.path.splitext(filename)[0])
    os.makedirs(doc_output, exist_ok=True)

    banner(f"DOCUMENT: {filename}")
    started = time.perf_counter()

    raw = extract_pdf(pdf_path)
    save_json(os.path.join(doc_output, "document.json"), raw)
    log(f"[EXTRACT] pages={raw['stats']['pages']} elements={raw['stats']['elements']} visuals={raw['stats'].get('visuals', 0)}")

    normalized = normalize_document(raw)
    save_json(os.path.join(doc_output, "document_normalized.json"), normalized)
    log(f"[NORMALIZE] elements={len(normalized.get('elements', []))}")

    structured = build_section_hierarchy(normalized)
    save_json(os.path.join(doc_output, "document_structure.json"), structured)
    log(f"[STRUCTURE] sections={structured['stats']['sections']}")

    if skip_vision:
        enriched = structured
        enriched.setdefault("stats", {})["enriched_visuals"] = 0
        log("[VISION] SKIPPED")
    else:
        log("[VISION] Enriching visual elements")
        enriched = enrich_document(structured)
        log(f"[VISION] enriched={enriched['stats'].get('enriched_visuals', 0)}")
    save_json(os.path.join(doc_output, "document_enriched.json"), enriched)

    chunked = build_parent_child_chunks(enriched)
    parents = chunked["parents"]
    children = chunked["children"]
    save_json(
        os.path.join(doc_output, "chunks.json"),
        {"parents": parents, "children": children, "parent_count": len(parents), "child_count": len(children)},
    )
    multimodal = sum(c.get("chunk_type") == "multimodal" for c in children)
    log(f"[CHUNK] parents={len(parents)} children={len(children)} multimodal_children={multimodal} parent_size={PARENT_CHUNK_SIZE} child_size={CHILD_CHUNK_SIZE}")

    result = upsert_document_and_chunks(conn, pdf_path, chunked, force=force)
    log(
        f"[STORE] action={result['action']} document_id={result['document_id']} "
        f"version={result['version']} chunks={result.get('chunks_inserted', 0)} "
        f"embeddings={result.get('embeddings_generated', 0)}"
    )
    log(f"[DONE] {filename} in {time.perf_counter() - started:.2f}s")
    return result


def find_pdfs(documents_dir: str) -> list[str]:
    return sorted(
        os.path.join(documents_dir, name)
        for name in os.listdir(documents_dir)
        if name.lower().endswith(".pdf")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multimodal RAG ingestion")
    parser.add_argument("--documents", default=DOCUMENTS_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = str(uuid.uuid4())
    started = time.perf_counter()

    banner("RAG INGESTION RUN")
    log(f"run_id={run_id}")
    log(f"documents={args.documents}")
    log(f"output={args.output}")
    log(f"embedding_model={EMBEDDING_MODEL}")
    log(f"force={args.force} skip_vision={args.skip_vision}")

    # ---------------------------------------------------------------
    # Discover documents on disk
    # ---------------------------------------------------------------

    pdf_files = find_pdfs(args.documents)
    log(f"[DISCOVERY] found={len(pdf_files)} PDF(s)")

    conn = get_connection()

    try:
        # -----------------------------------------------------------
        # Make sure database schema exists
        # -----------------------------------------------------------

        ensure_schema(conn)
        conn.commit()

        # -----------------------------------------------------------
        # Detect files that were deleted from the documents directory
        # -----------------------------------------------------------

        deleted_count = delete_missing_documents(conn, pdf_files)
        conn.commit()

        log(f"[DELETE] removed={deleted_count}")

        # -----------------------------------------------------------
        # Nothing to ingest
        # -----------------------------------------------------------

        if not pdf_files:
            banner("INGESTION SUMMARY")
            log("documents=0")
            log("created=0")
            log("updated=0")
            log("skipped=0")
            log("chunks=0")
            log("embeddings=0")
            log(f"total_time={time.perf_counter() - started:.2f}s")
            return

        # -----------------------------------------------------------
        # Process discovered PDFs
        # -----------------------------------------------------------

        results = []

        for pdf_path in pdf_files:
            try:
                result = process_document(
                    pdf_path,
                    output_dir=args.output,
                    force=args.force,
                    skip_vision=args.skip_vision,
                    conn=conn,
                )

                conn.commit()
                log("[COMMIT] Success")

                results.append(result)

            except Exception as exc:
                conn.rollback()
                log(f"[ROLLBACK] {type(exc).__name__}: {exc}")
                raise

    finally:
        conn.close()

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    banner("INGESTION SUMMARY")

    log(f"documents={len(results)}")
    log(
        f"created={sum(r['action'] == 'CREATED' for r in results)}"
    )
    log(
        f"updated={sum(r['action'] == 'UPDATED' for r in results)}"
    )
    log(
        f"reactivated={sum(r['action'] == 'REACTIVATED' for r in results)}"
    )
    log(
        f"skipped={sum(r['action'] == 'SKIPPED_UNCHANGED' for r in results)}"
    )
    log(
        f"chunks={sum(r.get('chunks_inserted', 0) for r in results)}"
    )
    log(
        f"embeddings={sum(r.get('embeddings_generated', 0) for r in results)}"
    )
    log(
        f"embeddings_reused={sum(r.get('embeddings_reused', 0) for r in results)}"
    )
    log(f"total_time={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
