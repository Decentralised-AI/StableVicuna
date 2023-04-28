import os
import gc
from string import Template
from threading import Thread

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding, TextIteratorStreamer


auth_token = os.environ.get("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
    "CarperAI/stable-vicuna-13b-fp16",
    use_auth_token=auth_token if auth_token else True,
)
model = AutoModelForCausalLM.from_pretrained(
    "CarperAI/stable-vicuna-13b-fp16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    use_auth_token=auth_token if auth_token else True,
)
model.eval()


max_context_length = model.config.max_position_embeddings
max_new_tokens = 768


prompt_template = Template("""\
### Human: $human
### Assistant: $bot\
""")


system_prompt = "### Assistant: I am StableVicuna, a large language model created by CarperAI. I am here to chat!"
system_prompt_tokens = tokenizer([f"{system_prompt}\n\n"], return_tensors="pt")
max_sys_tokens = system_prompt_tokens['input_ids'].size(-1)


def bot(history):
    history = history or []

    # Inject prompt formatting into the history
    prompt_history = []
    for human, bot in history:
        if bot is not None:
            bot = bot.replace("<br>", "\n")
            bot = bot.rstrip()
        prompt_history.append(
            prompt_template.substitute(
                human=human, bot=bot if bot is not None else "")
        )

    msg_tokens = tokenizer(
        "\n\n".join(prompt_history).strip(),
        return_tensors="pt",
        add_special_tokens=False  # Use <BOS> from the system prompt
    )

    # Take only the most recent context up to the max context length and prepend the
    # system prompt with the messages
    max_tokens = -max_context_length + max_new_tokens + max_sys_tokens
    inputs = BatchEncoding({
        k: torch.concat([system_prompt_tokens[k], msg_tokens[k][:, max_tokens:]], dim=-1)
        for k in msg_tokens
    }).to('cuda')
    # Remove `token_type_ids` b/c it's not yet supported for LLaMA `transformers` models
    if inputs.get("token_type_ids", None) is not None:
        inputs.pop("token_type_ids")

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=1.0,
        temperature=1.0,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        # Process out the prompt separator
        new_text = new_text.replace("<br>", "\n")
        if "###" in new_text:
            new_text = new_text.split("###")[0]
            partial_text += new_text.strip()
            history[-1][1] = partial_text
            break
        else:
            # Filter empty trailing new lines
            if new_text == "\n":
                new_text = new_text.strip()
            partial_text += new_text
            history[-1][1] = partial_text
        yield history
    return partial_text


def user(user_message, history):
    return "", history + [[user_message, None]]


with gr.Blocks() as demo:
    gr.Markdown("#StableVicuna by CarperAI")
    gr.HTML("<a href='https://huggingface.co/CarperAI/stable-vicuna-13b-delta'><code>CarperAI/stable-vicuna-13b-delta</a>")
    gr.HTML('''<center><a href="https://huggingface.co/spaces/CarperAI/StableVicuna?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to skip the queue and run in a private space</center>''')

    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
    state = gr.State([])
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Send a message",
                placeholder="Send a message",
                show_label=False
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Send")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear History")

    submit_event = msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=bot, inputs=[chatbot], outputs=[chatbot], queue=True)
    submit_click_event = submit.click(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=bot, inputs=[chatbot], outputs=[chatbot], queue=True)

    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, [chatbot], queue=True)

demo.queue(max_size=32, concurrency_count=2)
demo.launch()
