"""
Model by @duyphung for @carperai
Dumb Simple Gradio by @jon-tow
"""
from string import Template

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("CarperAI/vicuna-13b-fine-tuned-rlhf")
model = AutoModelForCausalLM.from_pretrained(
    "CarperAI/vicuna-13b-fine-tuned-rlhf",
    torch_dtype=torch.bfloat16,
)
model.cuda()
max_context_length = model.config.max_position_embeddings
max_new_tokens = 256 


prompt_template = Template("""\
### Human: $human
### Assistant: $bot\
""")


def bot(history):
    history = history or []

    # Hack to inject prompt formatting into the history
    prompt_history = []
    for human, bot in history:
        prompt_history.append(
            prompt_template.substitute(
                human=human, bot=bot if bot is not None else "")
        )

    prompt = "\n\n".join(prompt_history)
    prompt = prompt.rstrip()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    # Use only the most recent context up to the maximum context length with room left over 
    # for the max new tokens
    inputs = {k: v[:, -max_context_length + max_new_tokens:] for k, v in inputs.items()}
    inputs_length = inputs['input_ids'].shape[1]

    # Generate the response
    tokens = model.generate(
        **inputs,
        # Only allow the model to generate up to 512 tokens
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
    )
    # Strip the initial prompt
    tokens = tokens[:, inputs_length:]

    # Process response
    response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    response = response.split("###")[0].strip()

    # Add the response to the history
    history[-1][1] = response
    return history


def user(user_message, history):
    return "", history + [[user_message, None]]


with gr.Blocks() as demo:
    gr.Markdown("""Vicuna-13B RLHF Chatbot""")
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=512)
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    state = gr.State([])

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)
