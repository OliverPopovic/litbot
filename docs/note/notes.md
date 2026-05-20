# Global Reading Notes

LitBot supports global, corpus-grounded reading notes through the existing `/chat` API route and
`litbot ask` CLI command. There is no note listing or fetch route in this phase.

## Classification

Every input is classified by `IntentService` as `question` or `note` with a confidence score. A
classification below `LITBOT_INTENT_CONFIDENCE_THRESHOLD` routes to normal question answering, even
when the classifier guessed `note`. This favors avoiding accidental writes over aggressive note
capture.

The classifier may extract note text and a named work. The note workflow retrieves with the request
filters supplied by the caller; if no `filters.work` is present, it retrieves broadly and requires
the rewrite step to infer the work from corpus evidence.

## Grounding And Rewriting

`NoteService` retrieves candidate chunks with the existing hybrid retriever, then asks the LLM for a
structured `NoteProcessingPayload`:

- `should_save`: whether the note is grounded enough to store.
- `rewritten_note`: the concise factual note to store and embed.
- `inferred_work`: the corpus work associated with the note.
- `selected_chunk_ids`: retrieved chunk IDs that support the note.
- `citation_map`: model-provided claim-to-source bookkeeping.
- `rejection_reason`: a user-facing explanation when the note should not be saved.

The service rejects a note if retrieval returns no chunks, the rewritten note is blank,
`should_save=false`, any selected chunk ID was not retrieved, no selected chunks remain, or no work
can be inferred. Blank rejection reasons use the default: “The note could not be grounded in the
corpus.”

## Storage

`migrations/003_global_notes.sql` adds:

- `notes`: one row per saved note, keyed by `note_id` UUID. The row stores the original input,
  rewritten note, inferred work, optional source metadata, rewritten-note embedding, LLM model,
  prompt version, trace ID, status, and timestamps.
- `note_chunks`: a join table from `notes.note_id` to `chunks.chunk_id`, preserving retrieved rank
  and source label.

The service embeds the rewritten note, not the original input. The original input is retained only
for audit and transparency.

## Transaction Guarantee

Note insertion and chunk-link insertion run in a single database transaction. If any `note_chunks`
insert fails, the note row is rolled back with it. This prevents saved notes from existing without
their grounding evidence.

## Response Shape

Question responses keep the existing answer fields and add optional `intent` and
`intent_confidence`. Note responses set:

- `note_status`: `saved` or `not_saved`.
- `note_id`: present only for saved notes.
- `note`: the rewritten note that was stored, or the attempted note text for rejected notes.
- `original_note`: the user’s raw input.
- `note_work`: the inferred corpus work when saved.
- `note_chunk_ids`: selected supporting chunks.
- `note_rejection_reason`: present for rejected notes.

## Future Retrieval

The saved note embedding and `note_chunks` table are intended for future note retrieval and citation
display. Notes are global in v1: there is no user, session, namespace, or ownership field yet.
