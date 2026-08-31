import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.

    """
    global _rails

    
    guard_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0
)

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)

    logfire.info(
    "🛡️ NeMo Guardrails initialised ."
    )
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True, rail_response)  -> a rail fired; skip RAG.
        (False, None)          -> message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(
            messages=[{"role": "user", "content": message}],
            options={"log": {"activated_rails": True}},
        )

        # NeMo 0.23.0 returns a GenerationResponse object.
        activated_rails = []

        if result.log and result.log.activated_rails:
            activated_rails = result.log.activated_rails

        # A stopped input rail means the request was blocked.
        input_blocked = any(
            rail.type == "input" and rail.stop
            for rail in activated_rails
        )

        if input_blocked:
            rail_response = ""

            if result.response:
                first_response = result.response[0]

                if isinstance(first_response, dict):
                    rail_response = first_response.get("content", "")

            logfire.info(
                f"🛡️ Guardrails fired | query='{message[:80]}'"
            )

            return True, rail_response

        logfire.info("✅ Guardrails passed.")
        return False, None