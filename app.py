from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import logging
import os
from functools import lru_cache
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model components
model_lock = threading.Lock()

class ModelManager:
    """Manages multiple model loading and caching for better performance"""
    
    def __init__(self, 
                 detailed_summarizer_path="./models2/bart-large-cnn",
                 oneliner_summarizer_path="./models/pegasus-xsum", 
                 sentiment_path="./models_sentiment/roberta-sentiment"):
        
        # Model paths
        self.detailed_summarizer_path = detailed_summarizer_path
        self.oneliner_summarizer_path = oneliner_summarizer_path
        self.sentiment_path = sentiment_path
        
        # Detailed summarizer components (BART)
        self.detailed_tokenizer = None
        self.detailed_model = None
        self.detailed_summarizer = None
        self.detailed_loaded = False
        
        # One-liner summarizer components (Pegasus)
        self.oneliner_tokenizer = None
        self.oneliner_model = None
        self.oneliner_summarizer = None
        self.oneliner_loaded = False
        
        # Sentiment analyzer components
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.sentiment_analyzer = None
        self.sentiment_loaded = False
    
    def load_detailed_summarizer_model(self):
        """Load detailed summarizer (BART) model components with error handling"""
        try:
            if not os.path.exists(self.detailed_summarizer_path):
                logger.error(f"Detailed summarizer model path {self.detailed_summarizer_path} does not exist")
                return False
            
            logger.info("Loading detailed summarizer (BART) tokenizer...")
            self.detailed_tokenizer = AutoTokenizer.from_pretrained(self.detailed_summarizer_path)
            
            logger.info("Loading detailed summarizer (BART) model...")
            self.detailed_model = AutoModelForSeq2SeqLM.from_pretrained(self.detailed_summarizer_path)
            
            logger.info("Creating detailed summarization pipeline...")
            self.detailed_summarizer = pipeline(
                "summarization", 
                model=self.detailed_model, 
                tokenizer=self.detailed_tokenizer,
                device=-1  # Use CPU, change to 0 for GPU if available
            )
            
            self.detailed_loaded = True
            logger.info("Detailed summarizer (BART) model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading detailed summarizer model: {str(e)}")
            self.detailed_loaded = False
            return False
    
    def load_oneliner_summarizer_model(self):
        """Load one-liner summarizer (Pegasus) model components with error handling"""
        try:
            if not os.path.exists(self.oneliner_summarizer_path):
                logger.error(f"One-liner summarizer model path {self.oneliner_summarizer_path} does not exist")
                return False
            
            logger.info("Loading one-liner summarizer (Pegasus) tokenizer...")
            self.oneliner_tokenizer = AutoTokenizer.from_pretrained(self.oneliner_summarizer_path)
            
            logger.info("Loading one-liner summarizer (Pegasus) model...")
            self.oneliner_model = AutoModelForSeq2SeqLM.from_pretrained(self.oneliner_summarizer_path)
            
            logger.info("Creating one-liner summarization pipeline...")
            self.oneliner_summarizer = pipeline(
                "summarization", 
                model=self.oneliner_model, 
                tokenizer=self.oneliner_tokenizer,
                device=-1  # Use CPU, change to 0 for GPU if available
            )
            
            self.oneliner_loaded = True
            logger.info("One-liner summarizer (Pegasus) model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading one-liner summarizer model: {str(e)}")
            self.oneliner_loaded = False
            return False
    
    def load_sentiment_model(self):
        """Load sentiment analysis model components with error handling"""
        try:
            if not os.path.exists(self.sentiment_path):
                logger.error(f"Sentiment model path {self.sentiment_path} does not exist")
                return False
            
            logger.info("Loading sentiment tokenizer...")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_path)
            
            logger.info("Loading sentiment model...")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_path)
            
            logger.info("Creating sentiment analysis pipeline...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model=self.sentiment_model, 
                tokenizer=self.sentiment_tokenizer,
                device=-1  # Use CPU, change to 0 for GPU if available
            )
            
            self.sentiment_loaded = True
            logger.info("Sentiment analysis model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            self.sentiment_loaded = False
            return False
    
    def get_detailed_summarizer(self):
        """Thread-safe getter for detailed summarizer"""
        with model_lock:
            if not self.detailed_loaded:
                self.load_detailed_summarizer_model()
            return self.detailed_summarizer
    
    def get_oneliner_summarizer(self):
        """Thread-safe getter for one-liner summarizer"""
        with model_lock:
            if not self.oneliner_loaded:
                self.load_oneliner_summarizer_model()
            return self.oneliner_summarizer
    
    def get_sentiment_analyzer(self):
        """Thread-safe getter for sentiment analyzer"""
        with model_lock:
            if not self.sentiment_loaded:
                self.load_sentiment_model()
            return self.sentiment_analyzer

# Initialize model manager
model_manager = ModelManager()

@lru_cache(maxsize=100)
def cached_summarize(text_hash, summary_type="detailed", min_len=10, max_len=50):
    """Cache summaries to avoid re-processing identical text"""
    if summary_type == "oneliner":
        summarizer = model_manager.get_oneliner_summarizer()
    else:
        summarizer = model_manager.get_detailed_summarizer()
    
    if not summarizer:
        return None
    
    try:
        # Convert hash back to text (in real implementation, you'd store text differently)
        # For now, this is a placeholder for the caching concept
        summary = summarizer(text_hash, min_length=min_len, max_length=max_len, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return None

def validate_text(text):
    """Validate input text"""
    if not text or not isinstance(text, str):
        return False, "No text provided"
    
    text = text.strip()
    if len(text) < 10:
        return False, "Text too short (minimum 10 characters)"
    
    if len(text) > 10000:  # Reasonable limit
        return False, "Text too long (maximum 10,000 characters)"
    
    return True, text

def generate_summary(text, summary_type="detailed", min_length=None, max_length=None):
    """Generate summary based on type with appropriate parameters"""
    try:
        if summary_type == "oneliner":
            summarizer = model_manager.get_oneliner_summarizer()
            # Pegasus parameters for one-liner
            min_len = min_length or 15
            max_len = max_length or 35
        else:  # detailed
            summarizer = model_manager.get_detailed_summarizer()
            # BART parameters for detailed summary
            min_len = min_length or 50
            max_len = max_length or 150
        
        if not summarizer:
            return None, f"{summary_type.title()} summarizer model not available"
        
        # Generate summary
        start_time = time.time()
        
        # Adjust parameters based on text length
        text_length = len(text.split())
        if summary_type == "oneliner":
            # For one-liner, be more aggressive with length limits
            max_len = min(max_len, max(15, text_length // 8))
            min_len = min(min_len, max_len - 5)
        else:
            # For detailed, allow longer summaries for longer texts
            if text_length > 500:
                max_len = min(max_len + 50, 200)
                min_len = min(min_len + 20, max_len - 10)
        
        # Ensure min_length < max_length
        if min_len >= max_len:
            min_len = max(1, max_len - 10)
        
        summary_result = summarizer(
            text, 
            min_length=min_len, 
            max_length=max_len, 
            do_sample=False
        )
        
        processing_time = time.time() - start_time
        summary_text = summary_result[0]["summary_text"]
        
        return {
            "summary": summary_text,
            "type": summary_type,
            "original_length": len(text),
            "summary_length": len(summary_text),
            "word_count": len(summary_text.split()),
            "compression_ratio": round(len(summary_text) / len(text), 3),
            "processing_time": round(processing_time, 2),
            "model_used": "Pegasus-XSUM" if summary_type == "oneliner" else "BART-Large-CNN"
        }, None
        
    except Exception as e:
        logger.error(f"Error generating {summary_type} summary: {str(e)}")
        return None, f"Error generating {summary_type} summary: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment of given text"""
    try:
        sentiment_analyzer = model_manager.get_sentiment_analyzer()
        if not sentiment_analyzer:
            return None, "Sentiment analysis model not available"
        
        # Truncate text if too long for sentiment analysis (most models have token limits)
        if len(text) > 512:
            text = text[:512]
        
        result = sentiment_analyzer(text)
        
        # Extract results
        sentiment_result = result[0]
        sentiment_label = sentiment_result['label']
        confidence = sentiment_result['score']
        
        # Map labels to user-friendly names (adjust based on your model's output)
        label_mapping = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral', 
            'LABEL_2': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral',
            'POSITIVE': 'Positive'
        }
        
        sentiment_name = label_mapping.get(sentiment_label, sentiment_label)
        
        return {
            'sentiment': sentiment_name,
            'confidence': confidence,
            'raw_label': sentiment_label,
            'all_results': result
        }, None
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        return None, f"Error analyzing sentiment: {str(e)}"

@app.route("/")
def home():
    """Home page route"""
    return render_template("home.html")

@app.route("/summarizer")
def summarizer_page():
    """Summarizer tool page"""
    return render_template("summarizer_page.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    """Enhanced summarization endpoint with multi-style support"""
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        text = data.get("text", "")
        summary_type = data.get("summary_type", "detailed").lower()  # "oneliner" or "detailed"
        
        # Validate input
        is_valid, result = validate_text(text)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        text = result  # Use cleaned text
        
        # Validate summary type
        if summary_type not in ["oneliner", "detailed"]:
            return jsonify({"error": "Invalid summary type. Use 'oneliner' or 'detailed'"}), 400
        
        # Get optional parameters
        min_length = data.get("min_length")
        max_length = data.get("max_length")
        
        if min_length is not None:
            min_length = min(int(min_length), 200)
        if max_length is not None:
            max_length = min(int(max_length), 300)
        
        # Generate summary
        summary_result, error = generate_summary(text, summary_type, min_length, max_length)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify(summary_result)
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/sentiment_analysis", methods=["GET", "POST"])
def sentiment_analysis():
    """Sentiment analysis page and processing"""
    
    if request.method == "GET":
        # If accessed directly, redirect to summarizer page
        return redirect(url_for('summarizer_page'))
    
    try:
        summary_text = request.form.get('summary_text', '')
        
        if not summary_text.strip():
            return render_template('sentiment_Anyls.html', 
                                 error="No text provided for analysis")
        
        # Perform sentiment analysis
        sentiment_result, error = analyze_sentiment(summary_text)
        
        if error:
            return render_template('sentiment_Anyls.html', 
                                 summary_text=summary_text,
                                 error=error)
        
        return render_template('sentiment_Anyls.html',
                             summary_text=summary_text,
                             sentiment=sentiment_result['sentiment'],
                             confidence=sentiment_result['confidence'] * 100,
                             raw_label=sentiment_result['raw_label'],
                             all_results=sentiment_result['all_results'])
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return render_template('sentiment_Anyls.html', 
                             error=f"Error analyzing sentiment: {str(e)}")

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment_api():
    """API endpoint for sentiment analysis"""
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        text = data.get("text", "")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        # Perform sentiment analysis
        sentiment_result, error = analyze_sentiment(text)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify({
            "sentiment": sentiment_result['sentiment'],
            "confidence": sentiment_result['confidence'],
            "raw_label": sentiment_result['raw_label'],
            "text_length": len(text)
        })
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis API: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "detailed_summarizer_loaded": model_manager.detailed_loaded,
        "oneliner_summarizer_loaded": model_manager.oneliner_loaded,
        "sentiment_loaded": model_manager.sentiment_loaded,
        "timestamp": time.time()
    })

@app.route("/stats")
def get_stats():
    """Get system statistics"""
    return jsonify({
        "detailed_summarizer_status": "loaded" if model_manager.detailed_loaded else "not_loaded",
        "oneliner_summarizer_status": "loaded" if model_manager.oneliner_loaded else "not_loaded",
        "sentiment_status": "loaded" if model_manager.sentiment_loaded else "not_loaded",
        "cache_info": {
            "hits": cached_summarize.cache_info().hits,
            "misses": cached_summarize.cache_info().misses,
            "size": cached_summarize.cache_info().currsize,
            "maxsize": cached_summarize.cache_info().maxsize
        } if hasattr(cached_summarize, 'cache_info') else {}
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def initialize_app():
    """Initialize the application"""
    logger.info("Starting Enhanced News Analysis App...")
    
    # Pre-load models in background
    def load_models_background():
        # Load detailed summarizer model
        model_manager.load_detailed_summarizer_model()
        # Load one-liner summarizer model
        model_manager.load_oneliner_summarizer_model()
        # Load sentiment model
        model_manager.load_sentiment_model()
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=load_models_background)
    model_thread.daemon = True
    model_thread.start()
    
    logger.info("App initialization complete!")

if __name__ == "__main__":
    initialize_app()
    
    # Run with better configuration for development
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True,  # Enable threading for better performance
        use_reloader=False  # Disable reloader to avoid model loading twice
    )