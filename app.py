from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer
import torch

app = Flask(__name__)  # Corrected _name_ to __name__
CORS(app)  # Enable CORS for all routes

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PEGASUS Model
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load BART Model
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load T5 Model
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# PEGASUS Summarization function
def pegasus_summarize(text):
    inputs = pegasus_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    summary_ids = pegasus_model.generate(inputs["input_ids"], max_length=40, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# BART Summarization function
def bart_summarize(text):
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=40, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# T5 Summarization function
def t5_summarize(text):
    inputs = t5_tokenizer("summarize: " + text, return_tensors="pt", truncation=True, padding=True).to(device)
    summary_ids = t5_model.generate(inputs["input_ids"], max_length=40, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Hybrid Summarization function (Pegasus -> BART -> T5)
def hybrid_summarize(text):
    # Step 1: Summarize using Pegasus
    pegasus_summary = pegasus_summarize(text)

    # Step 2: Summarize the Pegasus output using BART
    bart_summary = bart_summarize(pegasus_summary)

    # Step 3: Summarize the BART output using T5
    final_summary = t5_summarize(bart_summary)

    return final_summary

# Flask route to handle summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_text = data.get('inputText', '')

    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    # Summarize using the hybrid method (Pegasus -> BART -> T5)
    final_summary = hybrid_summarize(input_text)

    return jsonify({'t5_summary': final_summary})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
