import pytest

from litbot.retrieval.service import (
    RetrievalService,
    _candidate_limit,
    _metadata_where_clause,
    _normalize_filters,
    _normalize_top_k,
    _rrf_score,
    document_to_retrieved_chunk,
)


def test_normalize_filters_drops_none_values_before_translation() -> None:
    filters = _normalize_filters({"work": "Frankenstein", "author": None})

    assert filters == {"work": "Frankenstein"}


def test_normalize_top_k_uses_default_only_when_unspecified() -> None:
    assert _normalize_top_k(None, 8) == 8
    assert _normalize_top_k(3, 8) == 3


def test_normalize_top_k_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        _normalize_top_k(0, 8)


def test_candidate_limit_applies_multiplier_floor_and_ceiling() -> None:
    assert _candidate_limit(3, multiplier=8, minimum=50, maximum=200) == 50
    assert _candidate_limit(10, multiplier=8, minimum=50, maximum=200) == 80
    assert _candidate_limit(40, multiplier=8, minimum=50, maximum=200) == 200


def test_candidate_limit_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        _candidate_limit(8, multiplier=0, minimum=50, maximum=200)
    with pytest.raises(ValueError, match="min_candidates"):
        _candidate_limit(8, multiplier=8, minimum=0, maximum=200)
    with pytest.raises(ValueError, match="max_candidates"):
        _candidate_limit(8, multiplier=8, minimum=50, maximum=10)


def test_rrf_score_uses_rank_and_constant() -> None:
    assert _rrf_score(1, 60) == pytest.approx(1 / 61)
    assert _rrf_score(10, 60) == pytest.approx(1 / 70)


def test_metadata_where_clause_handles_jsonb_and_scalar_filters() -> None:
    where, params = _metadata_where_clause(
        {"work": "Frankenstein", "tags": ["gothic"], "edition": {"volume": 1}}
    )

    assert where == (
        "metadata->>%s = %s AND metadata @> %s::jsonb "
        "AND metadata @> %s::jsonb"
    )
    assert params == [
        "work",
        "Frankenstein",
        '{"tags": ["gothic"]}',
        '{"edition": {"volume": 1}}',
    ]


def test_document_to_retrieved_chunk_round_trips_scores_and_text() -> None:
    row = {
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "text": "Evidence text.",
        "metadata": {"title": "Work"},
    }

    chunk = document_to_retrieved_chunk(
        row,
        label="S1",
        vector_score=0.9,
        lexical_score=0.2,
        combined_score=0.725,
        reason="hybrid vector + lexical match",
    )

    assert chunk.label == "S1"
    assert chunk.chunk_id == "chunk-1"
    assert chunk.source_id == "source-1"
    assert chunk.text == "Evidence text."
    assert chunk.combined_score == 0.725
    assert chunk.reason == "hybrid vector + lexical match"


def test_merge_assigns_labels_after_rrf_ranking() -> None:
    service = object.__new__(RetrievalService)

    vector_row = {
        "chunk_id": "vector",
        "source_id": "source",
        "text": "Semantic match.",
        "metadata": {},
        "vector_score": 0.1,
    }
    lexical_row = {
        "chunk_id": "lexical",
        "source_id": "source",
        "text": "Exact phrase.",
        "metadata": {},
        "lexical_score": 1.0,
    }

    chunks = service._merge(
        vector_rows=[vector_row],
        lexical_rows=[lexical_row],
        limit=2,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["vector", "lexical"]
    assert [chunk.label for chunk in chunks] == ["S1", "S2"]
    assert chunks[0].reason == "vector match only"
    assert chunks[1].reason == "lexical match only"
    assert chunks[0].combined_score == chunks[1].combined_score == pytest.approx(1 / 61)


def test_merge_prefers_shared_rrf_candidates_and_preserves_raw_scores() -> None:
    service = object.__new__(RetrievalService)

    shared_vector_row = {
        "chunk_id": "shared",
        "source_id": "source",
        "text": "Shared match.",
        "metadata": {},
        "vector_score": 0.2,
    }
    vector_only_row = {
        "chunk_id": "vector",
        "source_id": "source",
        "text": "Vector match.",
        "metadata": {},
        "vector_score": 0.99,
    }
    shared_lexical_row = {
        "chunk_id": "shared",
        "source_id": "source",
        "text": "Shared match.",
        "metadata": {},
        "lexical_score": 0.3,
    }

    chunks = service._merge(
        vector_rows=[vector_only_row, shared_vector_row],
        lexical_rows=[shared_lexical_row],
        limit=2,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["shared", "vector"]
    assert chunks[0].reason == "hybrid vector + lexical match"
    assert chunks[0].vector_score == 0.2
    assert chunks[0].lexical_score == 0.3
    assert chunks[0].combined_score == pytest.approx(1 / 62 + 1 / 61)
