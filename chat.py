"""Actor generation, Critic evaluation and the bounded revision loop."""

from datetime import datetime
from uuid import uuid4

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from context import (
    Background, CRITIC_SYSTEM_PROMPT,
    build_actor_prompt, build_critic_message, build_revision_message,
)
from execution_log import save_execution_log


MAX_ATTEMPTS = 3
TECHNICAL_ERROR_MESSAGE = "A technical error prevented an approved answer. Check the API configuration, connection and execution log."
EXHAUSTED_MESSAGE = "No answer was approved within the limit of three attempts. You can ask another question."


class Evaluation(BaseModel):
    """A schema-valid decision. Validation alone does not prove factual accuracy."""

    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    is_acceptable: bool
    feedback: str = Field(min_length=1, description="Brief evaluation feedback written in English.")


def prepare_history(history: list[dict]) -> list[dict]:
    """Give both models the same text conversation without Gradio metadata."""
    # Gradio 6 can supply text as content blocks and include UI-only metadata.
    # Pass only the conversation role and plain text to OpenAI.
    conversation = []
    for entry in history:
        if entry["role"] not in {"user", "assistant"}:
            raise ValueError("Unsupported conversation role.")
        
        content = entry["content"]
        if isinstance(content, list):
            content = "\n".join(block["text"] for block in content if block["type"] == "text")
        conversation.append({"role": entry["role"], "content": content})
    
    return conversation


def generate_answer(
    question: str,
    history: list[dict],
    background: Background,
    *,
    client: OpenAI,
    actor_model: str,
    previous_draft: str | None = None,
    feedback: str | None = None,
) -> str:
    """Send the system prompt, prepared history and current question to the Actor."""
    if not question.strip():
        raise ValueError("The question must not be empty.")
    if (previous_draft is None) != (feedback is None):
        raise ValueError("A revision requires both the rejected draft and Critic feedback.")
    if previous_draft is not None and (not previous_draft.strip() or not feedback.strip()):
        raise ValueError("The rejected draft and Critic feedback must not be empty.")

    messages = [{"role": "system", "content": build_actor_prompt(background)}] + history + [{"role": "user", "content": question}]
    if previous_draft is not None:
        messages.extend([{"role": "assistant", "content": previous_draft},{"role": "user", "content": build_revision_message(feedback)}])

    response = client.chat.completions.create(model=actor_model, messages=messages)
    if not response.choices:
        raise ValueError("The Actor did not return an answer.")
    
    completion = response.choices[0]
    answer = completion.message.content
    if completion.finish_reason != "stop" or completion.message.refusal or not answer or not answer.strip():
        raise ValueError("The Actor did not return a complete text answer.")
    return answer.strip()


def evaluate_answer(
    question: str,
    candidate_answer: str,
    history: list[dict],
    background: Background,
    *,
    client: OpenAI,
    critic_model: str,
) -> Evaluation:
    """Evaluate an answer using prepared history and require a validated Evaluation."""
    if not question.strip() or not candidate_answer.strip():
        raise ValueError("Evaluation requires a question and a candidate answer.")
    
    response = client.chat.completions.parse(
        model=critic_model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": build_critic_message(question, candidate_answer, history, background)}
        ],
        response_format=Evaluation,
    )

    if not response.choices:
        raise ValueError("The Critic did not return an evaluation.")
    completion = response.choices[0]
    if completion.finish_reason != "stop" or completion.message.refusal or completion.message.parsed is None:
        raise ValueError("The Critic did not return a complete, validated evaluation.")

    return completion.message.parsed


def process_question(
    question: str,
    history: list[dict],
    *,
    background: Background,
    client: OpenAI,
    actor_model: str,
    critic_model: str,
    scenario_id: str = "interactive",
    session_id: str | None = None,
    fake_first_draft: str | None = None,
    reject_all: bool = False,
) -> dict:
    """Run up to three Actor-Critic attempts, save the log and return the result."""
    if fake_first_draft is not None and (reject_all or not fake_first_draft.strip()):
        raise ValueError("Provide a non-empty fake draft or reject-all mode, not both.")
    
    if fake_first_draft is not None:
        scenario_id = "4"
    elif reject_all:
        scenario_id = "5"
    
    execution_log = {
        "run_id": uuid4().hex[:8],
        "session_id": session_id or uuid4().hex[:8],
        "scenario_id": scenario_id,
        "asked_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "question": question,
        "history": [],
        "models": {"actor": actor_model, "critic": "simulated" if reject_all else critic_model},
        "attempts": [],
        "status": "exhausted",
        "displayed_output": EXHAUSTED_MESSAGE,
    }
    previous_draft = feedback = None

    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        attempt = {"attempt": attempt_number, "candidate_answer": None, "evaluation": None}
        execution_log["attempts"].append(attempt)
        
        stage = "actor"
        try:
            # Prepare once, retaining technical-error handling for invalid history.
            if attempt_number == 1:
                execution_log["history"] = prepare_history(history)
            
            if fake_first_draft is not None and attempt_number == 1:
                if not question.strip():
                    raise ValueError("The question must not be empty.")
                candidate_answer = fake_first_draft.strip()
                attempt["candidate_source"] = "injected_data"
            else:
                candidate_answer = generate_answer(
                    question, execution_log["history"], background, client=client, actor_model=actor_model,
                    previous_draft=previous_draft, feedback=feedback
                )
            attempt["candidate_answer"] = candidate_answer

            stage = "critic"
            if reject_all:
                evaluation = Evaluation(is_acceptable=False, feedback="Simulated rejection for control-flow testing.")
                attempt["evaluation_source"] = "simulated"
            else:
                evaluation = evaluate_answer(
                    question, candidate_answer, execution_log["history"], background,
                    client=client, critic_model=critic_model,
                )
            attempt["evaluation"] = evaluation.model_dump()
            attempt["status"] = "approved" if evaluation.is_acceptable else "rejected"
        except (OpenAIError, ValueError) as exception:
            # Preserve the technical cause without copying credentials or raw responses.
            error = {
                "stage": stage,
                "type": type(exception).__name__,
                "reason": "Model request failed or returned invalid, refused or incomplete output.",
            }

            if getattr(exception, "status_code", None) is not None:
                error["http_status"] = exception.status_code

            attempt["status"] = "technical_error"
            execution_log.update(status="technical_error", error=error, displayed_output=TECHNICAL_ERROR_MESSAGE)
            break

        if evaluation.is_acceptable:
            execution_log.update(status="approved", displayed_output=candidate_answer)
            break
        previous_draft, feedback = candidate_answer, evaluation.feedback

    try:
        save_execution_log(execution_log)
    except OSError as error:
        execution_log.update(
            status="technical_error", displayed_output=TECHNICAL_ERROR_MESSAGE,
            error={"stage": "logging", "type": type(error).__name__, "reason": "Could not save the execution log."},
        )
    
    return execution_log
