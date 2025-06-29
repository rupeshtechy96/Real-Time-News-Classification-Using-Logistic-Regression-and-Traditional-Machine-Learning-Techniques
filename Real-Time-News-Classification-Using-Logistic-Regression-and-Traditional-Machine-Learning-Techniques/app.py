from flask import Flask, render_template, request, jsonify, send_file
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
from gtts import gTTS

nltk.download('stopwords', quiet=True)
port_stem = PorterStemmer()

app = Flask(__name__)

with open('vector.pkl', 'rb') as f:
    vector_form = pickle.load(f)
with open('model.pkl', 'rb') as f:
    load_model = pickle.load(f)

def stemming(content):
    """Clean and stem the content."""
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

def fake_news(news):
    """Predict if news is fake or real."""
    news = stemming(news)
    input_data = [news]
    vectorized_input = vector_form.transform(input_data)
    prediction = load_model.predict(vectorized_input)

    confidence = None
    if hasattr(load_model, 'predict_proba'):
        prob = load_model.predict_proba(vectorized_input)
        confidence = prob.max()

    return prediction[0], confidence, input_data[0]

def word_analysis(text):
    """Analyze which words are recognized by the model."""
    words = text.split()
    model_vocab = set(vector_form.get_feature_names_out())
    matched_words = [word for word in words if word in model_vocab]
    unmatched_words = [word for word in words if word not in model_vocab]
    
    influential_words = []
    try:
        if hasattr(load_model, 'coef_'):
            feature_names = vector_form.get_feature_names_out()
            coefs = load_model.coef_[0]
            top_coefs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
            influential_words = [word for word, coef in top_coefs[:10] if word in text.split()]
    except:
        pass
        
    return len(matched_words), len(unmatched_words), matched_words, unmatched_words, influential_words

def generate_speech(text):
    """Generate speech from text and return file path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts = gTTS(text=text, lang='en')
    tts.save(temp_file.name)
    return temp_file.name

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route handler."""
    prediction_class = None
    confidence = None
    matched = unmatched = 0
    matched_words = unmatched_words = influential_words = []

    if request.method == "POST":
        sentence = request.form["news_content"]
        if sentence.strip():
            prediction_class, confidence, processed_text = fake_news(sentence)
            matched, unmatched, matched_words, unmatched_words, influential_words = word_analysis(processed_text)

    return render_template(
        "index.html", 
        prediction_class=prediction_class, 
        confidence=confidence,
        matched=matched, 
        unmatched=unmatched, 
        matched_words=matched_words,
        unmatched_words=unmatched_words,
        influential_words=influential_words
    )

@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    """API endpoint for AJAX requests."""
    data = request.json
    sentence = data.get("news_content", "")
    
    if not sentence.strip():
        return jsonify({"error": "No content provided"}), 400
        
    prediction_class, confidence, processed_text = fake_news(sentence)
    matched, unmatched, matched_words, unmatched_words, influential_words = word_analysis(processed_text)
    
    return jsonify({
        "prediction_class": int(prediction_class),
        "confidence": float(confidence) if confidence is not None else None,
        "matched": matched,
        "unmatched": unmatched,
        "matched_words": matched_words[:20],  # Limit to prevent huge responses
        "unmatched_words": unmatched_words[:20],
        "influential_words": influential_words
    })

@app.route("/api/speech", methods=["POST"])
def speech_to_text():
    """Handle speech audio file and convert to text."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    try:
        import speech_recognition as sr
        
        audio_file = request.files['audio']
        recognizer = sr.Recognizer()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            
        with sr.AudioFile(temp_file.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        os.unlink(temp_file.name)  # Delete the temporary file
        
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/text-to-speech", methods=["POST"])
def text_to_speech():
    """Convert text to speech."""
    data = request.json
    text = data.get("text", "")
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    try:
        file_path = generate_speech(text)
        
        
        app.config["LAST_AUDIO_FILE"] = file_path
        
        return jsonify({"success": True, "file": os.path.basename(file_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<filename>")
def get_audio(filename):
    """Serve the generated audio file."""
    file_path = app.config.get("LAST_AUDIO_FILE")
    if file_path and os.path.basename(file_path) == filename:
        return send_file(file_path, mimetype="audio/mpeg")
    return "File not found", 404

@app.route("/api/read-result", methods=["POST"])
def read_result():
    """Generate speech for the analysis result."""
    data = request.json
    prediction_class = data.get("prediction_class")
    confidence = data.get("confidence")
    
    if prediction_class is None:
        return jsonify({"error": "No prediction data provided"}), 400
    
    # Create result text
    if prediction_class == 0:
        result_text = "This news appears to be Reliable."
    else:
        result_text = "This news appears to be Potentially Misleading."
        
    if confidence:
        result_text += f" Confidence: {round(confidence * 100, 1)}%."
    
    try:
        file_path = generate_speech(result_text)
        
        # Store the file path in the session or a temporary database
        app.config["LAST_AUDIO_FILE"] = file_path
        
        return jsonify({"success": True, "file": os.path.basename(file_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Make sure we have a directory for temporary audio files
    if not os.path.exists('temp'):
        os.makedirs('temp')
        
    app.run(debug=True)