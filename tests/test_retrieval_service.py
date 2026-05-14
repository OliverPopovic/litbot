from langchain_core.documents import Document

from litbot.retrieval.service import RetrievalService, _to_pgvector_filter


def test_to_pgvector_filter_translates_flat_filters() -> None:
    assert _to_pgvector_filter({"work": "Frankenstein", "author": "Mary Shelley"}) == {
        "$and": [
            {"work": {"$eq": "Frankenstein"}},
            {"author": {"$eq": "Mary Shelley"}},
        ]
    }


def test_merge_rows_assigns_labels_after_hybrid_ranking() -> None:
    service = object.__new__(RetrievalService)
    vector_doc = Document(
        id="vector",
        page_content="Semantic match.",
        metadata={"chunk_id": "vector", "source_id": "source"},
    )
    lexical_doc = Document(
        id="lexical",
        page_content="Exact phrase.",
        metadata={"chunk_id": "lexical", "source_id": "source"},
    )

    chunks = service._merge_rows(
        vector_rows=[(vector_doc, 0.9)],
        lexical_rows=[(lexical_doc, 1.0)],
        limit=2,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["lexical", "vector"]
    assert [chunk.label for chunk in chunks] == ["S1", "S2"]
    assert chunks[0].lexical_score == 1.0
