from transformers import pipeline
import gradio as gr


classifier = pipeline("sentiment-analysis")

def classify_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} ({result['score']:.2f})"

gr.Interface(fn=classify_sentiment,
             inputs="text",
             outputs="text",
             title="Sentiment Classifier").launch(share=True)
