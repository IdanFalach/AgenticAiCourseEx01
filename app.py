"""Personal chat with separate Actor generation and validated Critic evaluation."""

import argparse
from dataclasses import dataclass, field
import os

from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

from chat import process_question
from context import Background, PROJECT_DIR, load_background
from styles import CSS, JS, EXAMPLES

DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    actor_model: str
    critic_model: str


def load_settings() -> Settings:
    """Read this project's .env without replacing existing environment variables."""
    load_dotenv(PROJECT_DIR / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Copy .env.example to .env and set your key.")

    return Settings(
        api_key=api_key,
        actor_model=os.getenv("ACTOR_MODEL", "").strip() or DEFAULT_MODEL,
        critic_model=os.getenv("CRITIC_MODEL", "").strip() or DEFAULT_MODEL,
    )


def handle_message(
    message: str,
    history: list[dict],
    request: gr.Request = None,
    *,
    background: Background,
    client: OpenAI,
    actor_model: str,
    critic_model: str,
    fake_first_draft: str | None = None,
    reject_all: bool = False,
) -> str:
    """Run the Actor-Critic flow for a message and return its displayed output."""
    execution_log = process_question(
        message, history, background=background, client=client, actor_model=actor_model,
        critic_model=critic_model,
        session_id=request.session_hash.replace("-", "")[:8] if request and request.session_hash else None,
        fake_first_draft=fake_first_draft, reject_all=reject_all,
    )
    
    return execution_log["displayed_output"]


def build_app(
    settings: Settings, background: Background, *, client: OpenAI,
    fake_first_draft: str | None = None, reject_all: bool = False,
) -> gr.ChatInterface:
    """Build the Gradio interface and connect the message handler."""
    def chat(message: str, history: list[dict], request: gr.Request = None) -> str:
        return handle_message(
            message, history, request,
            background=background, client=client,
            actor_model=settings.actor_model, critic_model=settings.critic_model,
            fake_first_draft=fake_first_draft, reject_all=reject_all,
        )

    description = "Talk to my AI twin about my career"
    if fake_first_draft is not None:
        description = "Scenario 4 - first draft is injected data. The Critic and revisions use language models."
    elif reject_all:
        description = "Scenario 5 - control-flow test: a simulated Critic rejects every model-generated draft."
    
    return gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Personal Actor Critic Chat",
        description=description,
        chatbot=gr.Chatbot(show_label=False, elem_classes=["chatbot"]),
        analytics_enabled=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fake-first-draft", metavar="TEXT", help="Inject this text as the first draft of each question (scenario 4)")
    modes.add_argument("--reject-all", action="store_true", help="Simulate Critic rejection on every attempt (scenario 5)")
    arguments = parser.parse_args()
    if arguments.fake_first_draft is not None and not arguments.fake_first_draft.strip():
        parser.error("--fake-first-draft must not be empty.")
    
    settings = load_settings()
    background = load_background()

    print(f"Background loaded: LinkedIn {len(background.linkedin)} characters; summary {len(background.summary)} characters.")
    print(f"Models: Actor={settings.actor_model}; Critic={settings.critic_model}")

    with OpenAI(api_key=settings.api_key, timeout=60.0, max_retries=0) as client:
        build_app(settings, background, client=client,
                  fake_first_draft=arguments.fake_first_draft, reject_all=arguments.reject_all).launch(
            css=CSS, js=JS, theme=gr.themes.Base(),
        )


if __name__ == "__main__":
    main()
