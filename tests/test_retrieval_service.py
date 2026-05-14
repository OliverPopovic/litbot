from langchain_core.documents import Document

from litbot.retrieval.service import (
    RetrievalService,
    _metadata_where_clause,
    _normalize_filters,
    _to_pgvector_filter,
)


def test_to_pgvector_filter_translates_flat_filters() -> None:
    assert _to_pgvector_filter({"work": "Frankenstein", "author": "Mary Shelley"}) == {
        "$and": [
            {"work": {"$eq": "Frankenstein"}},
            {"author": {"$eq": "Mary Shelley"}},
        ]
    }


def test_normalize_filters_drops_none_values_before_translation() -> None:
    filters = _normalize_filters({"work": "Frankenstein", "author": None})

    assert filters == {"work": "Frankenstein"}
    assert _to_pgvector_filter(filters) == {"work": {"$eq": "Frankenstein"}}


def test_metadata_where_clause_handles_jsonb_and_scalar_filters() -> None:
    where, params = _metadata_where_clause(
        {"work": "Frankenstein", "tags": ["gothic"], "edition": {"volume": 1}}
    )

    assert where == (
        "e.cmetadata->>%s = %s AND e.cmetadata @> %s::jsonb "
        "AND e.cmetadata @> %s::jsonb"
    )
    assert params == [
        "work",
        "Frankenstein",
        '{"tags": ["gothic"]}',
        '{"edition": {"volume": 1}}',
    ]


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
