from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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
summarizer = None
model_lock = threading.Lock()

class ModelManager:
    """Manages model loading and caching for better performance"""
    
    def __init__(self, model_path="./models2/bart-large-cnn"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self.is_loaded = False
    
    def load_model(self):
        """Load model components with error handling"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model path {self.model_path} does not exist")
                return False
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.info("Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            logger.info("Creating summarization pipeline...")
            self.summarizer = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=-1  # Use CPU, change to 0 for GPU if available
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def get_summarizer(self):
        """Thread-safe getter for summarizer"""
        with model_lock:
            if not self.is_loaded:
                self.load_model()
            return self.summarizer

# Initialize model manager
model_manager = ModelManager()

@lru_cache(maxsize=100)
def cached_summarize(text_hash, min_len=10, max_len=50):
    """Cache summaries to avoid re-processing identical text"""
    summarizer = model_manager.get_summarizer()
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

@app.route("/")
def home():
    """Home page route"""
    return render_template("home.html")

@app.route("/summarizer")
def summarizer_page()   :
    """Summarizer tool page"""
    return render_template("summarizer_page.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    """Enhanced summarization endpoint with better error handling"""
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        text = data.get("text", "")
        
        # Validate input
        is_valid, result = validate_text(text)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        text = result  # Use cleaned text
        
        # Check if model is available
        summarizer = model_manager.get_summarizer()
        if not summarizer:
            return jsonify({"error": "Summarization model not available"}), 503
        
        # Get optional parameters
        min_length = min(int(data.get("min_length", 10)), 100)
        max_length = min(int(data.get("max_length", 50)), 200)
        
        # Ensure min_length < max_length
        if min_length >= max_length:
            min_length = max(1, max_length - 10)
        
        # Generate summary
        start_time = time.time()
        summary_result = summarizer(
            text, 
            min_length=min_length, 
            max_length=max_length, 
            do_sample=False
        )
        processing_time = time.time() - start_time
        
        summary_text = summary_result[0]["summary_text"]
        
        # Return enhanced response
        return jsonify({
            "summary": summary_text,
            "original_length": len(text),
            "summary_length": len(summary_text),
            "compression_ratio": round(len(summary_text) / len(text), 3),
            "processing_time": round(processing_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "timestamp": time.time()
    })

@app.route("/stats")
def get_stats():
    """Get system statistics"""
    return jsonify({
        "model_status": "loaded" if model_manager.is_loaded else "not_loaded",
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
    logger.info("Starting News Analysis App...")
    
    # Pre-load model in background
    def load_model_background():
        model_manager.load_model()
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=load_model_background)
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