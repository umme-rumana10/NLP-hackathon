import os
import re
from typing import Tuple

import gradio as gr
import language_tool_python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        # Use hosted Public API to avoid requiring local Java
        _tool = language_tool_python.LanguageToolPublicAPI("en-US")
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
    tool = get_tool()
    corrected = tool.correct(text)
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
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text_out.strip()


def process(text: str, tone: str) -> Tuple[str, str]:
    if not text or not text.strip():
        return "", ""

    # 1) Grammar correction
    corrected = correct_grammar(text)

    # 2) Tone rewriting from corrected text
    tone = (tone or "formal").strip().lower()
    rewritten = rewrite_tone(corrected, tone)

    corrected_md = f"**Corrected Text:**\n\n{corrected}"
    rewritten_md = f"**Rewritten Text ({tone.capitalize()} Tone):**\n\n{rewritten}"
    return corrected_md, rewritten_md


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
    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Launch for IDE/preview environments
    demo.launch(debug=True,host='0.0.0.0',port=7861)
