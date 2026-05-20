import json
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from litbot.config import Settings, get_settings
from litbot.langchain import make_chat_model
from litbot.models import IntentClassification

logger = structlog.get_logger(__name__)

INTENT_SYSTEM_PROMPT = """
You classify LitBot user inputs before retrieval. Return note only when the user is trying to
store an observation, annotation, margin note, thesis fragment, or reading note about a literary
work. Return question for requests that ask for an answer, explanation, comparison, quote lookup,
or anything ambiguous. Low confidence should be reflected in confidence below 0.65.
""".strip()

INTENT_DEVELOPER_PROMPT = """
Return valid structured data with intent, confidence, extracted_note_text, extracted_work, and
reason. If intent is note, extracted_note_text should be the note content to ground and rewrite,
not command boilerplate. If a work is clearly named, put it in extracted_work; otherwise null.
""".strip()

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INTENT_SYSTEM_PROMPT),
        ("system", INTENT_DEVELOPER_PROMPT),
        ("human", "{user_payload}"),
    ]
)


class IntentService:
    """Structured LLM classifier for question-vs-note routing."""

    def __init__(self, settings: Settings | None = None, model: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or make_chat_model(self.settings).with_structured_output(
            IntentClassification
        )

    def classify(self, user_input: str) -> IntentClassification:
        payload = {"input": user_input}
        try:
            result = self.model.invoke(
                INTENT_PROMPT.invoke({"user_payload": json.dumps(payload, ensure_ascii=False)})
            )
            classification = _classification_from_payload(result)
        except Exception as exc:
            logger.warning("intent_classification_failed", error=str(exc))
            classification = IntentClassification(
                intent="question",
                confidence=0.0,
                reason="Intent classification failed; defaulted to question.",
            )

        logger.info(
            "intent_classified",
            intent=classification.intent,
            confidence=classification.confidence,
            extracted_work=classification.extracted_work,
        )
        return classification


def _classification_from_payload(payload: object) -> IntentClassification:
    if isinstance(payload, IntentClassification):
        return payload
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        return IntentClassification(
            intent="question",
            confidence=0.0,
            reason="Intent classifier returned an unsupported payload.",
        )
    try:
        return IntentClassification.model_validate(payload)
    except ValidationError:
        return IntentClassification(
            intent="question",
            confidence=0.0,
            reason="Intent classifier returned invalid structured data.",
        )
