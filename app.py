import os
import re
from typing import Tuple

import gradio as gr
import language_tool_python
import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from language_tool_python.utils import RateLimitError

# ------------------------------
# Runtime setup
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("FLAN_T5_MODEL", "google/flan-t5-base")

_tool = None
_tokenizer = None
_model = None


def get_tool():
    global _tool
    if _tool is None:
        # Use local LanguageTool. If LT_REMOTE_URL is set, connect to that server.
        lt_remote = os.environ.get("LT_REMOTE_URL")
        if lt_remote:
            _tool = language_tool_python.LanguageTool("en-US", remote_server=lt_remote, timeout=8)
        else:
            _tool = language_tool_python.LanguageTool("en-US", timeout=8)
    return _tool


def load_model() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(DEVICE)
        _model.eval()
    return _tokenizer, _model


# ------------------------------
# Core functions
# ------------------------------

def correct_grammar(text: str) -> str:
    if not text or not text.strip():
        return ""
    # Allow bypass via env if API is rate-limited or offline
    if os.environ.get("DISABLE_GRAMMAR", "").lower() in ("1", "true", "yes"):  # pragma: no cover
        return text.strip()
    try:
        tool = get_tool()
        corrected = tool.correct(text)
    except RateLimitError:
        corrected = text
    except requests.exceptions.RequestException:
        corrected = text
    except Exception:
        corrected = text
    # Normalize whitespace a bit
    corrected = re.sub(r"\s+", " ", corrected).strip()
    return corrected


def rewrite_tone(text: str, tone: str) -> str:
    if not text:
        return ""
    tokenizer, model = load_model()
    # Prompt tailored for FLAN-T5 instruction style
    prompt = (
        f"Rewrite the following text in a {tone} tone while preserving meaning and improving clarity.\n"
        f"Text: {text}\n"
        f"Rewritten:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    with torch.no_grad():
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            eos_token_id=eos_id,
        )
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text_out = text_out.strip()
    if not text_out:
        return text
    return text_out


def process(text: str, tone: str) -> Tuple[str, str]:
    try:
        if not text or not text.strip():
            return "**Corrected Text:**\n\n", "**Rewritten Text:**\n\n"

        # 1) Grammar correction
        corrected = correct_grammar(text)

        # 2) Tone rewriting from corrected text
        tone = (tone or "formal").strip().lower()
        rewritten = rewrite_tone(corrected, tone)

        corrected_md = f"**Corrected Text:**\n\n{corrected}"
        rewritten_md = f"**Rewritten Text ({tone.capitalize()} Tone):**\n\n{rewritten}"
        return corrected_md, rewritten_md
    except Exception as e:
        err = str(e).strip()[:500]
        return (
            "**Corrected Text:**\n\n(An error occurred during grammar correction.)",
            f"**Rewritten Text:**\n\nError: {err}",
        )


# ------------------------------
# UI
# ------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Grammar & Tone Corrector Bot") as demo:
        gr.Markdown("# Grammar & Tone Corrector Bot")
        gr.Markdown("Fix your grammar and adjust your writing tone instantly.")

        with gr.Row():
            with gr.Column():
                inp_text = gr.Textbox(label="Your text", placeholder="Type or paste your text here...", lines=8)
                tone = gr.Dropdown(
                    label="Tone",
                    choices=["polite", "formal", "friendly", "assertive"],
                    value="polite",
                )
                run_btn = gr.Button("Correct & Rewrite")
            with gr.Column():
                with gr.Row():
                    out_corrected = gr.Markdown()
                    out_rewritten = gr.Markdown()

                run_btn.click(fn=process, inputs=[inp_text, tone], outputs=[out_corrected, out_rewritten])
    demo.queue()
    return demo


if __name__ == "__main__":
    # Preload only the FLAN-T5 model to reduce first-click latency;
    # avoid preloading LanguageTool to prevent downloads/network calls at startup.
    try:
        _ = load_model()
    except Exception:
        pass

    demo = build_ui()
    # Launch for IDE/preview environments
    demo.launch(
        debug=True,
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", 7861)),
        share=False,
    )
