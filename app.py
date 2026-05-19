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

    with gr.Blocks(theme=theme, title="Small Language Model") as app:
        gr.Markdown(
            """
            # Small Language Model
            ### Ask questions about your PDF documents
            *Powered by a locally-trained GPT-style transformer with retrieval fallback*
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion(label="Generation Settings", open=False):
                    temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                    top_k = gr.Slider(1, 100, value=40, step=1, label="Top-K")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (Nucleus)")
                    max_tokens = gr.Slider(32, 256, value=128, step=16, label="Max Tokens")
                    rep_penalty = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")

                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[temperature, top_k, top_p, max_tokens, rep_penalty],
                )

            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Model Info")
                model_info_display = gr.Markdown(value=get_model_info)
                gr.Markdown(
                    """
                    ---
                    ### Tips
                    - Ask definition questions: *"What is hardness?"*
                    - Ask explanatory questions: *"Explain corrosion"*
                    - *Retrieved* means a sentence from your PDFs was used
                    - *Generated* means the model composed the answer
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
