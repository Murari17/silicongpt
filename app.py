"""
Gradio web interface for the Small Language Model.

Features:
  - Chat-style conversation UI
  - Dark theme with modern aesthetics
  - Model info sidebar
  - Retrieval confidence display
  - Works on GPU and CPU
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    sys.exit(1)

from model_def import TransformerModel
from generate import (
    normalize_text,
    create_retriever,
    is_weak_answer,
    _answer_relevance,
    generate_answer,
    load_model,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL: Optional[TransformerModel] = None
SP: Optional[spm.SentencePieceProcessor] = None
RETRIEVER: Optional[object] = None
DEVICE = torch.device("cpu")
MODEL_INFO = {}
IS_HF_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))
HF_MAX_TOKENS_CAP = int(os.getenv("HF_MAX_TOKENS_CAP", "64"))


def init_model():
    global MODEL, SP, RETRIEVER, DEVICE, MODEL_INFO

    tokenizer_path = BASE_DIR / "tokenizer.model"
    best_model_path = BASE_DIR / "model_best.pth"
    model_path = BASE_DIR / "model.pth"
    sentences_path = BASE_DIR / "data" / "cleaned" / "sentences.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found. Run the training pipeline first.")

    active_path = best_model_path if best_model_path.exists() else model_path
    if not active_path.exists():
        raise FileNotFoundError("No model checkpoint found. Run training first.")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    SP = spm.SentencePieceProcessor()
    SP.load(str(tokenizer_path))

    checkpoint = torch.load(active_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        config.vocab_size = SP.get_piece_size()
        model = TransformerModel(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        MODEL_INFO = {
            "Parameters": f"{model.count_parameters():,}",
            "Dimensions": config.dim,
            "Layers": config.n_layers,
            "Heads": f"{config.n_heads} (KV: {config.n_kv_heads})",
            "Vocab Size": config.vocab_size,
            "Max Seq Len": config.max_seq_len,
            "Val Loss": f"{checkpoint.get('val_loss', 'N/A'):.4f}" if isinstance(checkpoint.get('val_loss'), float) else "N/A",
            "Val PPL": f"{checkpoint.get('val_ppl', 'N/A'):.1f}" if isinstance(checkpoint.get('val_ppl'), float) else "N/A",
            "Device": str(DEVICE).upper(),
        }
    else:
        raise RuntimeError("Old model format. Please retrain with the new architecture.")

    MODEL = model.to(DEVICE)
    MODEL.eval()

    if sentences_path.exists():
        sentences = [l.strip() for l in sentences_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if sentences:
            RETRIEVER = create_retriever(sentences)
            MODEL_INFO["Retrieval Sentences"] = f"{len(sentences):,}"


@torch.no_grad()
def generate(question: str, temperature: float = 0.7, top_k: int = 40,
             top_p: float = 0.9, max_tokens: int = 128,
             repetition_penalty: float = 1.15) -> tuple[str, str]:
    """Generate answer, returns (answer, source_type)."""
    if MODEL is None or SP is None:
        return "Model not loaded.", "error"

    effective_max_tokens = int(max_tokens)
    effective_top_k = int(top_k)
    if IS_HF_SPACE:
        effective_max_tokens = min(effective_max_tokens, HF_MAX_TOKENS_CAP)
        effective_top_k = min(effective_top_k, 20)

        if RETRIEVER:
            results = RETRIEVER.query(question, top_k=3)
            best = results[0] if results else ("", 0.0)
            retrieved_text, retrieved_score = best
            if retrieved_score >= 0.45 and not is_weak_answer(question, retrieved_text):
                return normalize_text(retrieved_text), f"Retrieved (confidence: {retrieved_score:.0%})"

    answer = generate_answer(
        MODEL, SP, question, DEVICE,
        max_new_tokens=effective_max_tokens,
        temperature=temperature,
        top_k=effective_top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stream=False,
    )

    source = "Generated"

    if RETRIEVER:
        results = RETRIEVER.query(question, top_k=3)
        best = results[0] if results else ("", 0.0)
        retrieved_text, retrieved_score = best

        gen_is_weak = is_weak_answer(question, answer)
        gen_relevance = _answer_relevance(question, answer)

        if gen_is_weak:
            if retrieved_score >= 0.30 and not is_weak_answer(question, retrieved_text):
                answer = normalize_text(retrieved_text)
                source = f"Retrieved (confidence: {retrieved_score:.0%})"
            else:
                answer = "I could not find a clear answer in the loaded notes."
                source = "No confident answer"
        elif gen_relevance < 0.4 and retrieved_score >= 0.45:
            if not is_weak_answer(question, retrieved_text):
                answer = normalize_text(retrieved_text)
                source = f"Retrieved (confidence: {retrieved_score:.0%})"

    if not answer:
        answer = "I need more training data to answer this clearly."
        source = "Insufficient data"

    return answer, source


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def respond(message: str, history: list,
            temperature: float, top_k: int, top_p: float,
            max_tokens: int, rep_penalty: float):
    answer, source = generate(
        message,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        max_tokens=int(max_tokens),
        repetition_penalty=rep_penalty,
    )
    return f"{answer}\n\n*{source}*"


def _new_chat_state(history: list[tuple[str, str]] | None = None, last_message: str = "") -> dict:
    return {
        "history": history or [],
        "last_message": last_message,
    }


def submit_message(message: str, state: dict,
                   temperature: float, top_k: int, top_p: float,
                   max_tokens: int, rep_penalty: float):
    history = list(state.get("history", [])) if state else []
    message = (message or "").strip()
    if not message:
        return history, "", state or _new_chat_state(history)

    answer, source = generate(
        message,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        max_tokens=int(max_tokens),
        repetition_penalty=rep_penalty,
    )
    reply = f"{answer}\n\n*{source}*"
    history.append((message, reply))
    return history, "", _new_chat_state(history, message)


def retry_last(state: dict,
               temperature: float, top_k: int, top_p: float,
               max_tokens: int, rep_penalty: float):
    history = list(state.get("history", [])) if state else []
    last_message = (state or {}).get("last_message", "").strip()
    if not last_message:
        return history, "", state or _new_chat_state(history)

    if history and history[-1][0] == last_message:
        history.pop()

    answer, source = generate(
        last_message,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        max_tokens=int(max_tokens),
        repetition_penalty=rep_penalty,
    )
    reply = f"{answer}\n\n*{source}*"
    history.append((last_message, reply))
    return history, "", _new_chat_state(history, last_message)


def undo_last(state: dict):
    history = list(state.get("history", [])) if state else []
    if history:
        history.pop()
    last_message = history[-1][0] if history else ""
    return history, _new_chat_state(history, last_message), ""


def clear_chat():
    return [], "", _new_chat_state([], "")


def get_model_info() -> str:
    if not MODEL_INFO:
        return "Model not loaded."
    lines = []
    for k, v in MODEL_INFO.items():
        lines.append(f"**{k}:** {v}")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    css = """
    .gradio-container {
        background: radial-gradient(circle at top, rgba(91, 114, 242, 0.12), transparent 34%), linear-gradient(180deg, #0a0a0a 0%, #0d1324 100%);
        color: #ececec;
        min-height: 100vh;
    }
    body {
        background: #0a0a0a;
    }
    #page-shell {
        max-width: 1280px;
        margin: 0 auto;
        padding: 18px 20px 28px;
    }
    #hero {
        padding: 10px 2px 18px;
        text-align: center;
    }
    #hero h1 {
        margin: 0;
        font-size: 34px;
        line-height: 1.05;
        letter-spacing: -0.04em;
        color: #f4f7ff;
    }
    #hero p {
        margin: 8px 0 0;
        color: #c7d0ee;
        font-size: 14px;
    }
    #main-panel {
        background: rgba(15, 22, 42, 0.95);
        border: 1px solid rgba(119, 133, 176, 0.16);
        border-radius: 22px;
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
        overflow: hidden;
    }
    #main-panel > .block {
        background: transparent;
    }
    .welcome-shell {
        padding: 24px 24px 18px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 18px;
        min-height: 260px;
    }
    .welcome-logo {
        width: 92px;
        height: 92px;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .welcome-logo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .welcome-shell h2 {
        margin: 0;
        font-size: 30px;
        letter-spacing: -0.03em;
        color: #f4f7ff;
        text-align: center;
    }
    .suggestions-grid {
        width: 100%;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        max-width: 820px;
    }
    @media (max-width: 720px) {
        .suggestions-grid {
            grid-template-columns: 1fr;
        }
        .welcome-shell h2 {
            font-size: 24px;
        }
    }
    .suggestion-card {
        width: 100%;
        border: 1px solid rgba(119, 133, 176, 0.16) !important;
        background: rgba(23, 31, 53, 0.92) !important;
        color: #e9eeff !important;
        border-radius: 18px !important;
        padding: 16px 18px !important;
        text-align: left !important;
        box-shadow: none !important;
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }
    .suggestion-card:hover {
        transform: translateY(-1px);
        border-color: rgba(91, 114, 242, 0.55) !important;
        background: rgba(28, 38, 64, 0.96) !important;
    }
    .suggestion-card span {
        display: block;
        font-size: 15px;
        line-height: 1.45;
    }
    .suggestion-card svg {
        float: right;
        opacity: 0.85;
    }
    #chatbot {
        min-height: 560px;
        background: transparent;
    }
    #chatbot .wrap {
        background: transparent;
    }
    #chatbot .message-row {
        margin: 10px 0;
    }
    #chatbot .message,
    #chatbot .message.bot,
    #chatbot .message.user {
        border-radius: 20px !important;
        padding: 16px 18px !important;
        max-width: min(82%, 860px) !important;
    }
    #chatbot .message.user {
        margin-left: auto !important;
        background: #2f2f2f !important;
        color: white !important;
    }
    #chatbot .message.bot {
        margin-right: auto !important;
        background: rgba(23, 31, 53, 0.95) !important;
        color: #ececec !important;
        border: 1px solid rgba(145, 160, 204, 0.12);
    }
    #chatbot .message-content {
        font-size: 16px;
        line-height: 1.65;
    }
    #chatbot .message-content p {
        margin: 0;
    }
    #chatbot .message-content em {
        display: inline-block;
        margin-top: 12px;
        color: #aab7dd;
        font-style: italic;
        font-size: 13px;
    }
    #input-row {
        padding: 16px 18px 10px;
        gap: 10px;
    }
    #user-input textarea {
        min-height: 54px !important;
        background: #2f2f2f !important;
        color: #f4f7ff !important;
        border: 1px solid rgba(146, 160, 204, 0.16) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
        resize: none !important;
    }
    #user-input textarea::placeholder {
        color: #8894b6 !important;
    }
    #submit-btn button,
    #retry-btn button,
    #undo-btn button,
    #clear-btn button {
        border-radius: 12px !important;
        min-height: 52px !important;
        font-weight: 700;
    }
    #submit-btn button {
        background: linear-gradient(135deg, #5b72f2, #6b58f2) !important;
        color: white !important;
    }
    .footer-note {
        padding: 2px 18px 16px;
        color: #9aa8cf;
        font-size: 12px;
        text-align: center;
    }
    """

    with gr.Blocks(theme=theme, title="Small Language Model", css=css) as app:
        with gr.Column(elem_id="page-shell"):
            with gr.Column(elem_id="hero"):
                gr.Markdown(
                    """
                    # SiliconGPT Core 1.0
                    *Powered by a locally-trained GPT-style transformer with retrieval fallback*
                    """
                )

            with gr.Column(elem_id="main-panel"):
                state = gr.State(_new_chat_state())
                default_temperature = 0.7
                default_top_k = 40
                default_top_p = 0.9
                default_max_tokens = 128
                default_rep_penalty = 1.15

                with gr.Column(elem_classes=["welcome-shell"], visible=True) as welcome:
                    gr.Markdown("<h2>How can I help you today?</h2>")
                    with gr.Row(elem_classes=["suggestions-grid"]):
                        suggestion_buttons = [
                            gr.Button("What is hardness of water?", elem_classes=["suggestion-card"]),
                            gr.Button("Explain environment & human health", elem_classes=["suggestion-card"]),
                            gr.Button("What is an aquifer?", elem_classes=["suggestion-card"]),
                            gr.Button("What is a wave?", elem_classes=["suggestion-card"]),
                        ]

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label=None,
                    height=560,
                    show_copy_button=True,
                    bubble_full_width=False,
                    visible=False,
                )

                with gr.Row(elem_id="input-row"):
                    message_box = gr.Textbox(
                        placeholder="Ask SiliconGPT...",
                        lines=1,
                        elem_id="user-input",
                        scale=10,
                        show_label=False,
                    )
                    submit_btn = gr.Button("Send", elem_id="submit-btn", variant="primary", scale=2)

                gr.Markdown("<div class='footer-note'>SiliconGPT can make mistakes. Check important information.</div>")

                def _refresh_view(history: list[tuple[str, str]], state_value: dict, message_value: str = ""):
                    has_messages = bool(history)
                    return (
                        gr.update(visible=not has_messages),
                        gr.update(visible=has_messages, value=history),
                        message_value,
                        state_value,
                    )

                def _submit_and_refresh(message: str, state_value: dict):
                    history, cleared_message, new_state = submit_message(
                        message,
                        state_value,
                        default_temperature,
                        default_top_k,
                        default_top_p,
                        default_max_tokens,
                        default_rep_penalty,
                    )
                    return _refresh_view(history, new_state, cleared_message)

                def _retry_and_refresh(state_value: dict):
                    history, cleared_message, new_state = retry_last(
                        state_value,
                        default_temperature,
                        default_top_k,
                        default_top_p,
                        default_max_tokens,
                        default_rep_penalty,
                    )
                    return _refresh_view(history, new_state, cleared_message)

                def _undo_and_refresh(state_value: dict):
                    history, new_state, cleared_message = undo_last(state_value)
                    return _refresh_view(history, new_state, cleared_message)

                def _clear_and_refresh():
                    history, cleared_message, new_state = clear_chat()
                    return _refresh_view(history, new_state, cleared_message)

                message_box.submit(
                    _submit_and_refresh,
                    inputs=[message_box, state],
                    outputs=[welcome, chatbot, message_box, state],
                )
                submit_btn.click(
                    _submit_and_refresh,
                    inputs=[message_box, state],
                    outputs=[welcome, chatbot, message_box, state],
                )

                with gr.Row():
                    retry_btn = gr.Button("Retry last", elem_id="retry-btn")
                    undo_btn = gr.Button("Undo last", elem_id="undo-btn")
                    clear_btn = gr.Button("Clear chat", elem_id="clear-btn")

                def _make_suggestion_handler(query: str):
                    return lambda state_value: _submit_and_refresh(query, state_value)

                for button, text in zip(suggestion_buttons, [
                    "What is hardness of water?",
                    "Explain environment & human health",
                    "What is an aquifer?",
                    "What is a wave?",
                ]):
                    button.click(
                        fn=_make_suggestion_handler(text),
                        inputs=[state],
                        outputs=[welcome, chatbot, message_box, state],
                    )

                retry_btn.click(
                    _retry_and_refresh,
                    inputs=[state],
                    outputs=[welcome, chatbot, message_box, state],
                )
                undo_btn.click(
                    lambda state_value: _refresh_view(*undo_last(state_value)),
                    inputs=[state],
                    outputs=[welcome, chatbot, message_box, state],
                )
                clear_btn.click(
                    _clear_and_refresh,
                    inputs=[],
                    outputs=[welcome, chatbot, message_box, state],
                )

    return app


def main() -> None:
    print("Initializing model...")
    init_model()
    print("Starting web interface...")
    app = build_ui()
    port = int(os.getenv("PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
