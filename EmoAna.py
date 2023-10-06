import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', top_k=1)


def Analyse(input):
    return classifier(input)[0]


with gr.Blocks() as demo:
    gr.Markdown("<center><h>Emotion Analysis</h></center>")
    with gr.Tab("Emotion Analysis"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                input = gr.Textbox(label="Text", lines=4, max_lines=100, placeholder="Waiting for analysis")
                with gr.Row():
                    analysis_button = gr.Button("Test")
                output = gr.Textbox(label="Result", lines=4, max_lines=100, placeholder="Waiting for analysis")
    analysis_button.click(Analyse, api_name="analytics", inputs=[input], outputs=output)

demo.launch(debug=True, server_name="localhost")
