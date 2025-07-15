from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer 
import torch

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

# Load PEGASUS Model
pegasus_model_name = "google/pegasus-xsum"
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(device)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)

# Load BART Model
bart_model_name = "facebook/bart-large-cnn"
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Load T5 Model
t5_model_name = "google-t5/t5-small"  # or "t5-base", "t5-large"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Pegasus summarization function
def pegasus_summarize(text):
    inputs = pegasus_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    summary_ids = pegasus_model.generate(inputs["input_ids"], max_length=40, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# BART summarization function
def bart_summarize(text):
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=40, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# T5 summarization function
def t5_summarize(text):
    inputs = t5_tokenizer("summarize: " + text, return_tensors="pt", truncation=True).to(device)
    summary_ids = t5_model.generate(inputs["input_ids"], max_length=40, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Hybrid summarization function (combining Pegasus, BART, and T5)
def hybrid_summarize(text):
    # Step 1: Summarize using Pegasus
    pegasus_summary = pegasus_summarize(text)

    # Step 2: Summarize Pegasus output using BART
    bart_summary = bart_summarize(pegasus_summary)

    # Step 3: Summarize BART output using T5
    final_summary = t5_summarize(bart_summary)

    return final_summary



# API endpoint for summarization
@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data.get('inputText')

    # Generate the hybrid summary
    summary = hybrid_summarize(input_text)
    
    return jsonify({'t5_summary': summary})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    
