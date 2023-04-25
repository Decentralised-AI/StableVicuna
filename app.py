import os
from string import Template
from threading import Thread

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


auth_token = os.environ.get("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
    "CarperAI/vicuna-13b-fine-tuned-rlhf",
    use_auth_token=auth_token if auth_token else True,
)
model = AutoModelForCausalLM.from_pretrained(
    "CarperAI/vicuna-13b-fine-tuned-rlhf-fp16",
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="./offload",
    low_cpu_mem_usage=True,  # Not required for demo but leave for now
    use_auth_token=auth_token if auth_token else True,
)
model.cuda()
max_context_length = model.config.max_position_embeddings
max_new_tokens = 500


prompt_template = Template("""\
### Human: $human
### Assistant: $bot\
""")


def bot(history):
    # print(f"History:\n`{history}`")
    history = history or []
    # Hack to inject prompt formatting into the history
    prompt_history = []
    for human, bot in history:
        if bot is not None:
            bot = bot.replace("<br>", "\n")
            bot = bot.rstrip()
        prompt_history.append(
            prompt_template.substitute(
                human=human, bot=bot if bot is not None else "")
        )

    messages = "\n\n".join(prompt_history)
    messages = messages.rstrip()
    # print(f"Messages:\n{messages}")

    # Use only the most recent context up to the maximum context length with room left over
    # for the max new tokens
    inputs = tokenizer(messages, return_tensors='pt').to('cuda')
    inputs = {k: v[:, -max_context_length + max_new_tokens:]
              for k, v in inputs.items()}
    if inputs.get("token_type_ids", None) is not None:
        inputs.pop("token_type_ids")
    # print(f"Inputs: {inputs}")
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    # Generate the response
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
    )

    # print(f"Generating with kwargs: {generate_kwargs}")
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        # Process out the prompt separator. NOTE: we should tune with special tokens for this
        new_text = new_text.replace("<br>", "\n")
        # print(f"New text: `{new_text}`")
        if "###" in new_text:
            new_text = new_text.split("###")[0]
            partial_text += new_text.strip()
            history[-1][1] = partial_text
            break
        else:
            # Filter empty trailing whitespaces
            if new_text.isspace():
                new_text = new_text.strip()
            partial_text += new_text
            history[-1][1] = partial_text
        yield history

    return partial_text


def user(user_message, history):
    return "", history + [[user_message, None]]


with gr.Blocks() as demo:
    gr.Markdown("Chat-RLHF by CarperAI")
    gr.HTML("<a href='https://huggingface.co/CarperAI/vicuna-13b-fine-tuned-rlhf'><code>CarperAI/vicuna-13b-fine-tuned-rlhf</a>")
    gr.HTML('''<center><a href="https://huggingface.co/spaces/CarperAI/chat-rlhf?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to skip the queue and run in a private space</center>''')

    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=512)
    state = gr.State([])
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                             show_label=False).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    submit_event = msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True).then(
        bot, chatbot, chatbot)
    submit_click_event = submit.click(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True).then(
        bot, chatbot, chatbot)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[
               submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, chatbot, queue=True)

demo.queue(max_size=32, concurrency_count=2)
demo.launch(share=True)
