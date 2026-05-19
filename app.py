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

    answer = generate_answer(
        MODEL, SP, question, DEVICE,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
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
        background: radial-gradient(circle at top, rgba(91, 114, 242, 0.14), transparent 35%), linear-gradient(180deg, #0b1020 0%, #0d1324 100%);
        color: #e7ecff;
    }
    #page-shell {
        max-width: 1440px;
        margin: 0 auto;
        padding: 24px 20px 28px;
    }
    #hero {
        padding: 4px 2px 20px;
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
        background: rgba(15, 22, 42, 0.9);
        border: 1px solid rgba(119, 133, 176, 0.18);
        border-radius: 18px;
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        overflow: hidden;
    }
    #chatbot {
        min-height: 560px;
    }
    #chatbot .wrap {
        background: transparent;
    }
    #chatbot .message-row {
        margin: 12px 0;
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
        background: linear-gradient(135deg, #4755d7, #5b72f2) !important;
        color: white !important;
    }
    #chatbot .message.bot {
        margin-right: auto !important;
        background: rgba(25, 33, 55, 0.95) !important;
        color: #eef3ff !important;
        border: 1px solid rgba(145, 160, 204, 0.14);
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
    #chatbot .label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        background: linear-gradient(135deg, #5b72f2, #6b58f2);
        border-radius: 10px;
        color: white;
        font-weight: 700;
        font-size: 14px;
        margin: 12px 0 10px;
    }
    #input-row {
        padding: 16px 18px 6px;
        gap: 10px;
    }
    #user-input textarea {
        min-height: 52px !important;
        background: #3a4763 !important;
        color: #f4f7ff !important;
        border: 1px solid rgba(146, 160, 204, 0.16) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
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
    #sidebar {
        padding-left: 10px;
    }
    #sidebar .gr-markdown {
        color: #e9eeff;
    }
    #sidebar h3 {
        margin-top: 0;
        margin-bottom: 12px;
    }
    #sidebar .panel {
        background: rgba(15, 22, 42, 0.82);
        border: 1px solid rgba(119, 133, 176, 0.14);
        border-radius: 16px;
        padding: 18px;
    }
    """

    with gr.Blocks(theme=theme, title="Small Language Model", css=css) as app:
        with gr.Column(elem_id="page-shell"):
            with gr.Column(elem_id="hero"):
                gr.Markdown(
                    """
                    # Small Language Model
                    *Powered by a locally-trained GPT-style transformer with retrieval fallback*
                    """
                )

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(elem_id="main-panel"):
                        with gr.Accordion(label="Generation Settings", open=False):
                            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                            top_k = gr.Slider(1, 100, value=40, step=1, label="Top-K")
                            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (Nucleus)")
                            max_tokens = gr.Slider(32, 256, value=128, step=16, label="Max Tokens")
                            rep_penalty = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")

                        gr.Markdown("<div class='label'>Chatbot</div>")
                        chatbot = gr.Chatbot(elem_id="chatbot", label=None, show_copy_button=False)

                        state = gr.State(_new_chat_state())

                        with gr.Row(elem_id="input-row"):
                            message_box = gr.Textbox(
                                placeholder="Type a message...",
                                lines=1,
                                elem_id="user-input",
                                scale=10,
                                show_label=False,
                            )
                            submit_btn = gr.Button("Submit", elem_id="submit-btn", variant="primary", scale=2)

                        with gr.Row():
                            retry_btn = gr.Button("Retry", elem_id="retry-btn")
                            undo_btn = gr.Button("Undo", elem_id="undo-btn")
                            clear_btn = gr.Button("Clear", elem_id="clear-btn")

                    message_box.submit(
                        submit_message,
                        inputs=[message_box, state, temperature, top_k, top_p, max_tokens, rep_penalty],
                        outputs=[chatbot, message_box, state],
                    )
                    submit_btn.click(
                        submit_message,
                        inputs=[message_box, state, temperature, top_k, top_p, max_tokens, rep_penalty],
                        outputs=[chatbot, message_box, state],
                    )
                    retry_btn.click(
                        retry_last,
                        inputs=[state, temperature, top_k, top_p, max_tokens, rep_penalty],
                        outputs=[chatbot, message_box, state],
                    )
                    undo_btn.click(
                        undo_last,
                        inputs=[state],
                        outputs=[chatbot, state, message_box],
                    )
                    clear_btn.click(
                        clear_chat,
                        inputs=[],
                        outputs=[chatbot, message_box, state],
                    )

                with gr.Column(scale=1, min_width=250, elem_id="sidebar"):
                    gr.Markdown("### Model Info")
                    model_info_display = gr.Markdown(value=get_model_info())
                    gr.Markdown(
                        """
                        <div class='panel'>
                        ### Tips
                        - Ask definition questions: *"What is hardness?"*
                        - Ask explanatory questions: *"Explain corrosion"*
                        - *Retrieved* means a sentence from your PDFs was used
                        - *Generated* means the model composed the answer
                        </div>
                        """
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
