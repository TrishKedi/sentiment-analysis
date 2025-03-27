from transformers import pipeline

# Load a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Try it out
print(classifier("I love Hugging Face!"))

import gradio as gr

def classify_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} ({result['score']:.2f})"

gr.Interface(fn=classify_sentiment,
             inputs="text",
             outputs="text",
             title="Sentiment Classifier").launch(share=True)
