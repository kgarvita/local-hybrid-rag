import argparse

from .context import build_context
from .generate import generate_answer
from .hybrid import search_hybrid
from .ingest import get_connection
from .reranking import rerank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the RAG retrieval and generation pipeline"
    )

    parser.add_argument(
        "question",
        help="Question to ask the RAG system",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    question = args.question

    conn = get_connection()

    try:
        # -----------------------------------------------------------
        # 1. Hybrid retrieval
        # -----------------------------------------------------------

        candidates = search_hybrid(
            conn,
            question,
            retrieval_k=10,
            top_k=10,
        )

        print("\n" + "=" * 80)
        print("HYBRID RETRIEVAL")
        print("=" * 80)

        for i, result in enumerate(candidates, start=1):
            print(
                f"[{i}] "
                f"rrf={result['rrf_score']:.6f} "
                f"vector_rank={result['vector_rank']} "
                f"bm25_rank={result['bm25_rank']} "
                f"chunk={result['chunk_index']}"
            )

        # -----------------------------------------------------------
        # 2. Reranking
        # -----------------------------------------------------------

        reranked = rerank(
            question,
            candidates,
            top_k=5,
        )

        print("\n" + "=" * 80)
        print("RERANKED RESULTS")
        print("=" * 80)

        for i, result in enumerate(reranked, start=1):
            print(f"\n[{i}]")
            print(f"rerank_score={result['rerank_score']:.4f}")
            print(f"rrf_score={result['rrf_score']:.6f}")
            print(f"vector_rank={result['vector_rank']}")
            print(f"bm25_rank={result['bm25_rank']}")
            print(f"chunk_index={result['chunk_index']}")
            print(f"content={result['content'][:500]}")

        # -----------------------------------------------------------
        # 3. Parent/context expansion
        # -----------------------------------------------------------

        contexts = build_context(
            conn,
            reranked,
        )

        print("\n" + "=" * 80)
        print("FINAL CONTEXT")
        print("=" * 80)

        for i, context in enumerate(contexts, start=1):
            print(f"\n[{i}]")
            print(f"parent_id={context['parent_id']}")
            print(f"file={context['filename']}")
            print(f"page={context['page']}")
            print(f"content={context['content'][:1000]}")

        # -----------------------------------------------------------
        # 4. LLM generation
        # -----------------------------------------------------------

        answer = generate_answer(
            question,
            contexts,
        )

        print("\n" + "=" * 80)
        print("FINAL RAG ANSWER")
        print("=" * 80)
        print(answer)

    finally:
        conn.close()


if __name__ == "__main__":
    main()