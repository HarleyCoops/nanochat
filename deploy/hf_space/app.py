import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure custom config/model are registered before loading from Hub.
import configuration_nanochat  # noqa: F401
import modeling_nanochat  # noqa: F401


MODEL_ID = "HarleyCooper/nanochat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    TORCH_DTYPE = torch.bfloat16 if major >= 8 else torch.float16
else:
    TORCH_DTYPE = torch.float32


try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )
except Exception as exc:
    raise RuntimeError(
        "Failed to load the nanochat tokenizer. Make sure `tokenizer/tokenizer.pkl` "
        "or the expected tokenizer assets are present in the repository."
    ) from exc

# Ensure pad token exists for generation.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
)
model.to(DEVICE)
model.eval()


def build_prompt(history, user_message):
    turns = []
    for user, assistant in history:
        if user:
            turns.append(f"User: {user}")
        if assistant:
            turns.append(f"Assistant: {assistant}")
    turns.append(f"User: {user_message}")
    turns.append("Assistant:")
    return "\n".join(turns)


def generate_response(message, history, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
    prompt = build_prompt(history, message)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        text = "(no response produced)"
    return text


demo = gr.ChatInterface(
    fn=generate_response,
    title="NanoChat (HF Space)",
    description=(
        "Interact with the HarleyCooper/nanochat model served entirely from this Space. "
        "Responses are generated with the custom NanoChat architecture."
    ),
    additional_inputs=[
        gr.Slider(32, 512, value=160, step=16, label="Max new tokens"),
        gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p"),
        gr.Slider(1, 200, value=50, step=1, label="Top-k"),
        gr.Slider(1.0, 2.0, value=1.05, step=0.05, label="Repetition penalty"),
    ],
)


if __name__ == "__main__":
    demo.launch()
