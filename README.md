# Grammar & Tone Corrector Bot

Fix your grammar and adjust your writing tone instantly.

## Features
- Grammar correction using LanguageTool (Public API via `language-tool-python`).
- Tone rewriting using `google/flan-t5-base` (Transformers + Torch).
- Simple Gradio UI to input text, choose tone, and see results side-by-side.

## Tones
- polite
- formal
- friendly
- assertive

## Requirements
See `requirements.txt`.

## Run
```bash
python app.py
```

The app launches a Gradio interface titled "Grammar & Tone Corrector Bot" and should open in Windsurf’s preview panel.
