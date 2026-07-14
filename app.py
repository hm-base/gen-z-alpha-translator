"""Run the trained translator as a chat app — WITHOUT retraining.

Loads the base model + your saved LoRA adapter (from training) and launches a
little Gradio chat box. Use this instead of re-running the whole notebook.

Usage:
    uv run python app.py

Requires that training has been run at least once (so the adapter folder exists).
"""

from pathlib import Path
import sys

from unsloth import FastLanguageModel
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from translate_core import translate as core_translate  # noqa: E402

ADAPTER_DIR = Path(__file__).resolve().parent / "genz_lora_adapter"
MAX_SEQ_LEN = 1024

if not ADAPTER_DIR.exists():
    sys.exit(
        f"No trained adapter found at {ADAPTER_DIR}.\n"
        "Run the training notebook once first (it saves the adapter there)."
    )

print(">> loading fine-tuned model (base + your LoRA adapter)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),   # unsloth loads the base model + your adapter
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print(">> ready")


def translate(text: str, direction: str) -> str:
    # 'Slang -> English' button label starts with 'Slang' => we want to_english;
    # otherwise to_slang. Shared core applies the abstain guard + decoding.
    d = "to_english" if direction.startswith("Slang") else "to_slang"
    return core_translate(model, tokenizer, text, d, use_guard=True)


with gr.Blocks(title="Gen Z Slang Translator") as app:
    gr.Markdown("# Gen Z / Alpha Slang Translator\nType a sentence and pick a direction.")
    direction = gr.Radio(
        ["Slang -> English", "English -> Slang"],
        value="Slang -> English", label="Direction",
    )
    inp = gr.Textbox(label="Your text", placeholder="e.g. bro really ate with that fit, delulu fr")
    out = gr.Textbox(label="Translation")
    btn = gr.Button("Translate", variant="primary")
    btn.click(translate, inputs=[inp, direction], outputs=out)
    inp.submit(translate, inputs=[inp, direction], outputs=out)
    gr.Examples(
        [["bro really ate with that fit, delulu fr", "Slang -> English"],
         ["That outfit is genuinely impressive.", "English -> Slang"]],
        inputs=[inp, direction],
    )

if __name__ == "__main__":
    # share=True gives a public link (valid ~72h) to send to teammates.
    # Set to False for local-only.
    app.launch(share=True)
