from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import logging
import os
from functools import lru_cache
import threading
import time
import hashlib
import requests  # Add this import for API calls

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# News API Configuration
GNEWS_API_KEY = '133bb93beaeb9f5cfc91ea8efee7b0c4'
GNEWS_BASE_URL = 'https://gnews.io/api/v4'

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
        
        # Add loading status tracking
        self.loading_status = {
            'detailed': False,
            'oneliner': False,
            'sentiment': False
        }
    
    def load_detailed_summarizer_model(self):
        """Load detailed summarizer (BART) model components with error handling"""
        if self.detailed_loaded:
            return True
            
        self.loading_status['detailed'] = True
        try:
            if not os.path.exists(self.detailed_summarizer_path):
                logger.error(f"Detailed summarizer model path {self.detailed_summarizer_path} does not exist")
                return False
            
            logger.info("Loading detailed summarizer (BART) tokenizer...")
            self.detailed_tokenizer = AutoTokenizer.from_pretrained(
                self.detailed_summarizer_path,
                local_files_only=True  # Only use local files
            )
            
            logger.info("Loading detailed summarizer (BART) model...")
            self.detailed_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.detailed_summarizer_path,
                local_files_only=True
            )
            
            logger.info("Creating detailed summarization pipeline...")
            self.detailed_summarizer = pipeline(
                "summarization", 
                model=self.detailed_model, 
                tokenizer=self.detailed_tokenizer,
                device=-1,  # Use CPU, change to 0 for GPU if available
                framework="pt"  # Explicitly specify PyTorch
            )
            
            self.detailed_loaded = True
            logger.info("Detailed summarizer (BART) model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading detailed summarizer model: {str(e)}")
            self.detailed_loaded = False
            return False
        finally:
            self.loading_status['detailed'] = False
    
    def load_oneliner_summarizer_model(self):
        """Load one-liner summarizer (Pegasus) model components with error handling"""
        if self.oneliner_loaded:
            return True
            
        self.loading_status['oneliner'] = True
        try:
            if not os.path.exists(self.oneliner_summarizer_path):
                logger.error(f"One-liner summarizer model path {self.oneliner_summarizer_path} does not exist")
                return False
            
            logger.info("Loading one-liner summarizer (Pegasus) tokenizer...")
            self.oneliner_tokenizer = AutoTokenizer.from_pretrained(
                self.oneliner_summarizer_path,
                local_files_only=True
            )
            
            logger.info("Loading one-liner summarizer (Pegasus) model...")
            self.oneliner_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.oneliner_summarizer_path,
                local_files_only=True
            )
            
            logger.info("Creating one-liner summarization pipeline...")
            self.oneliner_summarizer = pipeline(
                "summarization", 
                model=self.oneliner_model, 
                tokenizer=self.oneliner_tokenizer,
                device=-1,  # Use CPU, change to 0 for GPU if available
                framework="pt"
            )
            
            self.oneliner_loaded = True
            logger.info("One-liner summarizer (Pegasus) model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading one-liner summarizer model: {str(e)}")
            self.oneliner_loaded = False
            return False
        finally:
            self.loading_status['oneliner'] = False
    
    def load_sentiment_model(self):
        """Load sentiment analysis model components with error handling"""
        if self.sentiment_loaded:
            return True
            
        self.loading_status['sentiment'] = True
        try:
            if not os.path.exists(self.sentiment_path):
                logger.error(f"Sentiment model path {self.sentiment_path} does not exist")
                return False
            
            logger.info("Loading sentiment tokenizer...")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                self.sentiment_path,
                local_files_only=True
            )
            
            logger.info("Loading sentiment model...")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_path,
                local_files_only=True
            )
            
            logger.info("Creating sentiment analysis pipeline...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model=self.sentiment_model, 
                tokenizer=self.sentiment_tokenizer,
                device=-1,  # Use CPU, change to 0 for GPU if available
                framework="pt"
            )
            
            self.sentiment_loaded = True
            logger.info("Sentiment analysis model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            self.sentiment_loaded = False
            return False
        finally:
            self.loading_status['sentiment'] = False
    
    def get_detailed_summarizer(self):
        """Thread-safe getter for detailed summarizer"""
        with model_lock:
            if not self.detailed_loaded and not self.loading_status['detailed']:
                self.load_detailed_summarizer_model()
            return self.detailed_summarizer if self.detailed_loaded else None
    
    def get_oneliner_summarizer(self):
        """Thread-safe getter for one-liner summarizer"""
        with model_lock:
            if not self.oneliner_loaded and not self.loading_status['oneliner']:
                self.load_oneliner_summarizer_model()
            return self.oneliner_summarizer if self.oneliner_loaded else None
    
    def get_sentiment_analyzer(self):
        """Thread-safe getter for sentiment analyzer"""
        with model_lock:
            if not self.sentiment_loaded and not self.loading_status['sentiment']:
                self.load_sentiment_model()
            return self.sentiment_analyzer if self.sentiment_loaded else None

# Initialize model manager
model_manager = ModelManager()

# News API caching
news_cache = {}
CACHE_DURATION = 10 * 60  # 10 minutes in seconds

def get_text_hash(text):
    """Generate hash for text caching"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@lru_cache(maxsize=100)
def cached_summarize(text_hash, original_text, summary_type="detailed", min_len=10, max_len=50):
    """Cache summaries to avoid re-processing identical text"""
    if summary_type == "oneliner":
        summarizer = model_manager.get_oneliner_summarizer()
    else:
        summarizer = model_manager.get_detailed_summarizer()
    
    if not summarizer:
        return None
    
    try:
        summary = summarizer(original_text, min_length=min_len, max_length=max_len, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error during cached summarization: {str(e)}")
        return None

def fetch_gnews_data(endpoint, params):
    """Fetch data from GNews API with error handling"""
    try:
        url = f"{GNEWS_BASE_URL}/{endpoint}"
        params['token'] = GNEWS_API_KEY
        
        logger.info(f"Fetching news from: {url}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if API returned an error
        if 'errors' in data:
            logger.error(f"GNews API error: {data['errors']}")
            return None, f"News API error: {data['errors']}"
        
        return data, None
        
    except requests.exceptions.Timeout:
        logger.error("News API request timeout")
        return None, "News service timeout - please try again"
    except requests.exceptions.ConnectionError:
        logger.error("News API connection error")
        return None, "Unable to connect to news service"
    except requests.exceptions.HTTPError as e:
        logger.error(f"News API HTTP error: {e}")
        return None, f"News service error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"News API request error: {e}")
        return None, "News service unavailable"
    except Exception as e:
        logger.error(f"Unexpected error fetching news: {e}")
        return None, "Unexpected error occurred"

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

def validate_length_params(min_length, max_length, text_length):
    """Validate and adjust length parameters based on text length"""
    word_count = len(text_length.split()) if isinstance(text_length, str) else text_length
    
    # Set reasonable defaults based on text length
    if min_length is None:
        min_length = max(5, min(20, word_count // 20))
    
    if max_length is None:
        max_length = max(min_length + 10, min(150, word_count // 3))
    
    # Ensure parameters are reasonable
    min_length = max(1, min(min_length, 200))
    max_length = max(min_length + 5, min(max_length, 300))
    
    return min_length, max_length

def generate_summary(text, summary_type="detailed", min_length=None, max_length=None):
    """Generate summary based on type with appropriate parameters"""
    try:
        if summary_type == "oneliner":
            summarizer = model_manager.get_oneliner_summarizer()
            # Pegasus parameters for one-liner
            min_len = min_length or 10
            max_len = max_length or 40
        else:  # detailed
            summarizer = model_manager.get_detailed_summarizer()
            # BART parameters for detailed summary
            min_len = min_length or 30
            max_len = max_length or 150
        
        if not summarizer:
            return None, f"{summary_type.title()} summarizer model not available. Please check if the model is loaded correctly."
        
        # Validate and adjust parameters
        min_len, max_len = validate_length_params(min_len, max_len, text)
        
        # Generate summary
        start_time = time.time()
        
        # Adjust parameters based on text length for better results
        text_word_count = len(text.split())
        if summary_type == "oneliner":
            # For one-liner, be more aggressive with length limits
            max_len = min(max_len, max(15, text_word_count // 8))
            min_len = min(min_len, max(5, max_len - 5))
        else:
            # For detailed, allow longer summaries for longer texts
            if text_word_count > 500:
                max_len = min(max_len + 50, 250)
                min_len = min(min_len + 20, max_len - 10)
        
        # Ensure min_length < max_length
        if min_len >= max_len:
            min_len = max(1, max_len - 5)
        
        # Use caching for performance
        text_hash = get_text_hash(text)
        cached_result = cached_summarize(text_hash, text, summary_type, min_len, max_len)
        
        if cached_result:
            processing_time = time.time() - start_time
            return {
                "summary": cached_result,
                "type": summary_type,
                "original_length": len(text),
                "original_word_count": text_word_count,
                "summary_length": len(cached_result),
                "summary_word_count": len(cached_result.split()),
                "compression_ratio": round(len(cached_result) / len(text), 3),
                "processing_time": round(processing_time, 2),
                "model_used": "Pegasus-XSUM" if summary_type == "oneliner" else "BART-Large-CNN",
                "cached": True
            }, None
        
        # Generate new summary if not cached
        summary_result = summarizer(
            text, 
            min_length=min_len, 
            max_length=max_len, 
            do_sample=False,
            truncation=True  # Handle long texts
        )
        
        processing_time = time.time() - start_time
        summary_text = summary_result[0]["summary_text"]
        
        return {
            "summary": summary_text,
            "type": summary_type,
            "original_length": len(text),
            "original_word_count": text_word_count,
            "summary_length": len(summary_text),
            "summary_word_count": len(summary_text.split()),
            "compression_ratio": round(len(summary_text) / len(text), 3),
            "processing_time": round(processing_time, 2),
            "model_used": "Pegasus-XSUM" if summary_type == "oneliner" else "BART-Large-CNN",
            "cached": False
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
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
            logger.info(f"Text truncated to {max_length} characters for sentiment analysis")
        
        result = sentiment_analyzer(text)
        
        # Extract results
        sentiment_result = result[0]
        sentiment_label = sentiment_result['label'].upper()
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
            'confidence': round(confidence, 4),
            'raw_label': sentiment_label,
            'text_analyzed_length': len(text),
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

# NEW: News API Endpoints
@app.route("/api/news", methods=["GET"])
def get_latest_news():
    """Get latest news headlines"""
    try:
        # Check cache first
        current_time = time.time()
        if 'headlines' in news_cache:
            cache_time = news_cache['headlines'].get('timestamp', 0)
            if current_time - cache_time < CACHE_DURATION:
                logger.info("Returning cached news headlines")
                return jsonify(news_cache['headlines']['data'])
        
        # Fetch fresh data from GNews API
        params = {
            'lang': 'en',
            'max': 5,
            'sortby': 'publishedAt'
        }
        
        data, error = fetch_gnews_data('top-headlines', params)
        
        if error:
            return jsonify({
                'error': error,
                'articles': [],
                'cached': False
            }), 500
        
        # Cache the results
        news_cache['headlines'] = {
            'data': {
                'articles': data.get('articles', []),
                'totalArticles': data.get('totalArticles', 0),
                'cached': False
            },
            'timestamp': current_time
        }
        
        logger.info(f"Fetched {len(data.get('articles', []))} news articles")
        
        return jsonify(news_cache['headlines']['data'])
        
    except Exception as e:
        logger.error(f"Error in news endpoint: {str(e)}")
        return jsonify({
            'error': 'Unable to fetch news at the moment. Please try again later.',
            'articles': [],
            'cached': False
        }), 500

@app.route("/api/news/search", methods=["GET"])
def search_news():
    """Search news by query"""
    try:
        query = request.args.get('q', '')
        max_results = min(int(request.args.get('max', 10)), 50)  # Limit to 50 max
        
        if not query.strip():
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # Check cache
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}_{max_results}"
        current_time = time.time()
        
        if cache_key in news_cache:
            cache_time = news_cache[cache_key].get('timestamp', 0)
            if current_time - cache_time < CACHE_DURATION:
                logger.info(f"Returning cached search results for: {query}")
                return jsonify(news_cache[cache_key]['data'])
        
        # Fetch from API
        params = {
            'q': query,
            'lang': 'en',
            'max': max_results,
            'sortby': 'publishedAt'
        }
        
        data, error = fetch_gnews_data('search', params)
        
        if error:
            return jsonify({
                'error': error,
                'articles': [],
                'query': query,
                'cached': False
            }), 500
        
        # Cache results
        result_data = {
            'articles': data.get('articles', []),
            'totalArticles': data.get('totalArticles', 0),
            'query': query,
            'cached': False
        }
        
        news_cache[cache_key] = {
            'data': result_data,
            'timestamp': current_time
        }
        
        return jsonify(result_data)
        
    except Exception as e:
        logger.error(f"Error in news search endpoint: {str(e)}")
        return jsonify({
            'error': 'Search unavailable at the moment. Please try again later.',
            'articles': [],
            'query': query if 'query' in locals() else '',
            'cached': False
        }), 500

@app.route("/api/news/categories", methods=["GET"])
def get_news_by_category():
    """Get news by category"""
    try:
        category = request.args.get('category', 'general')
        max_results = min(int(request.args.get('max', 10)), 50)
        
        valid_categories = ['general', 'world', 'nation', 'business', 'technology', 'entertainment', 'sports', 'science', 'health']
        if category not in valid_categories:
            return jsonify({'error': f'Invalid category. Valid options: {", ".join(valid_categories)}'}), 400
        
        # Check cache
        cache_key = f"category_{category}_{max_results}"
        current_time = time.time()
        
        if cache_key in news_cache:
            cache_time = news_cache[cache_key].get('timestamp', 0)
            if current_time - cache_time < CACHE_DURATION:
                return jsonify(news_cache[cache_key]['data'])
        
        # Fetch from API
        params = {
            'category': category,
            'lang': 'en',
            'max': max_results,
            'sortby': 'publishedAt'
        }
        
        data, error = fetch_gnews_data('top-headlines', params)
        
        if error:
            return jsonify({
                'error': error,
                'articles': [],
                'category': category,
                'cached': False
            }), 500
        
        # Cache results
        result_data = {
            'articles': data.get('articles', []),
            'totalArticles': data.get('totalArticles', 0),
            'category': category,
            'cached': False
        }
        
        news_cache[cache_key] = {
            'data': result_data,
            'timestamp': current_time
        }
        
        return jsonify(result_data)
        
    except Exception as e:
        logger.error(f"Error in category news endpoint: {str(e)}")
        return jsonify({
            'error': 'Category news unavailable at the moment.',
            'articles': [],
            'category': category if 'category' in locals() else 'general',
            'cached': False
        }), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    """Enhanced summarization endpoint with multi-style support"""
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "received_content_type": request.content_type}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
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
            try:
                min_length = max(1, min(int(min_length), 200))
            except (ValueError, TypeError):
                return jsonify({"error": "min_length must be a valid integer"}), 400
                
        if max_length is not None:
            try:
                max_length = max(5, min(int(max_length), 300))
            except (ValueError, TypeError):
                return jsonify({"error": "max_length must be a valid integer"}), 400
        
        # Generate summary
        summary_result, error = generate_summary(text, summary_type, min_length, max_length)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify(summary_result)
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({"error": "Internal server error occurred", "details": str(e)}), 500

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
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
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
            "text_length": len(text),
            "text_analyzed_length": sentiment_result['text_analyzed_length']
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
        "loading_status": model_manager.loading_status,
        "news_cache_size": len(news_cache),
        "timestamp": time.time()
    })

@app.route("/stats")
def get_stats():
    """Get system statistics"""
    cache_info = {}
    if hasattr(cached_summarize, 'cache_info'):
        info = cached_summarize.cache_info()
        cache_info = {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": round(info.hits / (info.hits + info.misses) * 100, 2) if (info.hits + info.misses) > 0 else 0
        }
    
    return jsonify({
        "detailed_summarizer_status": "loaded" if model_manager.detailed_loaded else "not_loaded",
        "oneliner_summarizer_status": "loaded" if model_manager.oneliner_loaded else "not_loaded",
        "sentiment_status": "loaded" if model_manager.sentiment_loaded else "not_loaded",
        "loading_status": model_manager.loading_status,
        "cache_info": cache_info,
        "news_cache_entries": len(news_cache)
    })

@app.route('/summarize_pegasus', methods=['POST'])
def summarize_pegasus():
    """Legacy endpoint for Pegasus summarization"""
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
            
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400

        # Validate text
        is_valid, result = validate_text(text)
        if not is_valid:
            return jsonify({'error': result}), 400
        
        text = result

        # Use the improved summarization function
        summary_result, error = generate_summary(text, "oneliner")
        
        if error:
            return jsonify({'error': error}), 500
            
        return jsonify({
            'summary': summary_result['summary'],
            'model_used': summary_result['model_used'],
            'processing_time': summary_result['processing_time']
        })

    except Exception as e:
        logger.error(f"Error in summarize_pegasus endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/summarize_bart', methods=['POST'])
def summarize_bart():
    """Legacy endpoint for BART summarization"""
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
            
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400

        # Validate text
        is_valid, result = validate_text(text)
        if not is_valid:
            return jsonify({'error': result}), 400
        
        text = result

        # Use the improved summarization function for detailed summary
        summary_result, error = generate_summary(text, "detailed")
        
        if error:
            return jsonify({'error': error}), 500
            
        return jsonify({
            'summary': summary_result['summary'],
            'model_used': summary_result['model_used'],
            'processing_time': summary_result['processing_time'],
            'compression_ratio': summary_result['compression_ratio']
        })

    except Exception as e:
        logger.error(f"Error in summarize_bart endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route("/api/news/summarize", methods=["POST"])
def summarize_news_article():
    """Summarize a news article from URL or content"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        article_url = data.get("url", "")
        article_content = data.get("content", "")
        summary_type = data.get("summary_type", "detailed").lower()
        
        # Validate summary type
        if summary_type not in ["oneliner", "detailed"]:
            return jsonify({"error": "Invalid summary type. Use 'oneliner' or 'detailed'"}), 400
        
        # If URL is provided, try to fetch content (placeholder for now)
        if article_url and not article_content:
            # In a real implementation, you'd scrape the article content from the URL
            # For now, return an error asking for direct content
            return jsonify({
                "error": "URL scraping not implemented. Please provide article content directly.",
                "suggestion": "Use the 'content' field to provide the article text directly."
            }), 400
        
        if not article_content:
            return jsonify({"error": "No article content provided"}), 400
        
        # Validate content
        is_valid, result = validate_text(article_content)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        article_content = result
        
        # Generate summary
        summary_result, error = generate_summary(article_content, summary_type)
        
        if error:
            return jsonify({"error": error}), 500
        
        # Add sentiment analysis to the summary
        sentiment_result, sentiment_error = analyze_sentiment(summary_result['summary'])
        
        response_data = summary_result.copy()
        
        if sentiment_result and not sentiment_error:
            response_data['sentiment_analysis'] = {
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence']
            }
        elif sentiment_error:
            response_data['sentiment_analysis'] = {
                'error': sentiment_error
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in news summarization endpoint: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/api/models/load", methods=["POST"])
def load_models():
    """Manually trigger model loading"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        model_type = data.get("model_type", "all").lower()
        
        valid_types = ["all", "detailed", "oneliner", "sentiment"]
        if model_type not in valid_types:
            return jsonify({"error": f"Invalid model type. Valid options: {', '.join(valid_types)}"}), 400
        
        results = {}
        
        if model_type in ["all", "detailed"]:
            success = model_manager.load_detailed_summarizer_model()
            results['detailed_summarizer'] = "loaded" if success else "failed"
        
        if model_type in ["all", "oneliner"]:
            success = model_manager.load_oneliner_summarizer_model()
            results['oneliner_summarizer'] = "loaded" if success else "failed"
        
        if model_type in ["all", "sentiment"]:
            success = model_manager.load_sentiment_model()
            results['sentiment_analyzer'] = "loaded" if success else "failed"
        
        return jsonify({
            "message": f"Model loading triggered for: {model_type}",
            "results": results,
            "current_status": {
                "detailed_loaded": model_manager.detailed_loaded,
                "oneliner_loaded": model_manager.oneliner_loaded,
                "sentiment_loaded": model_manager.sentiment_loaded
            }
        })
        
    except Exception as e:
        logger.error(f"Error in model loading endpoint: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear various caches"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        cache_type = data.get("cache_type", "all").lower()
        
        valid_types = ["all", "news", "summary"]
        if cache_type not in valid_types:
            return jsonify({"error": f"Invalid cache type. Valid options: {', '.join(valid_types)}"}), 400
        
        cleared = []
        
        if cache_type in ["all", "news"]:
            news_cache.clear()
            cleared.append("news_cache")
        
        if cache_type in ["all", "summary"]:
            cached_summarize.cache_clear()
            cleared.append("summary_cache")
        
        return jsonify({
            "message": f"Cache cleared: {', '.join(cleared)}",
            "cache_type": cache_type,
            "cleared_caches": cleared
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route("/api/batch/summarize", methods=["POST"])
def batch_summarize():
    """Batch summarization endpoint for multiple texts"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        texts = data.get("texts", [])
        summary_type = data.get("summary_type", "detailed").lower()
        max_items = min(len(texts), 10)  # Limit to 10 items max
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Texts must be provided as a list"}), 400
        
        if len(texts) == 0:
            return jsonify({"error": "No texts provided"}), 400
        
        if summary_type not in ["oneliner", "detailed"]:
            return jsonify({"error": "Invalid summary type. Use 'oneliner' or 'detailed'"}), 400
        
        results = []
        errors = []
        
        for i, text in enumerate(texts[:max_items]):
            try:
                # Validate text
                is_valid, validated_text = validate_text(text)
                if not is_valid:
                    errors.append({
                        "index": i,
                        "error": validated_text,
                        "original_text_preview": text[:100] + "..." if len(text) > 100 else text
                    })
                    continue
                
                # Generate summary
                summary_result, error = generate_summary(validated_text, summary_type)
                
                if error:
                    errors.append({
                        "index": i,
                        "error": error,
                        "original_text_preview": text[:100] + "..." if len(text) > 100 else text
                    })
                    continue
                
                results.append({
                    "index": i,
                    "summary": summary_result['summary'],
                    "original_length": summary_result['original_length'],
                    "summary_length": summary_result['summary_length'],
                    "compression_ratio": summary_result['compression_ratio'],
                    "model_used": summary_result['model_used']
                })
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": f"Processing error: {str(e)}",
                    "original_text_preview": text[:100] + "..." if len(text) > 100 else text
                })
        
        return jsonify({
            "processed": len(results),
            "errors": len(errors),
            "total_submitted": min(len(texts), max_items),
            "max_items_limit": max_items,
            "summary_type": summary_type,
            "results": results,
            "errors": errors if errors else None
        })
        
    except Exception as e:
        logger.error(f"Error in batch summarize endpoint: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found", "available_endpoints": [
        "/", "/summarizer", "/summarize", "/analyze_sentiment", "/health", "/stats",
        "/api/news", "/api/news/search", "/api/news/categories", "/api/news/summarize",
        "/api/models/load", "/api/cache/clear", "/api/batch/summarize",
        "/summarize_pegasus", "/summarize_bart"
    ]}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def payload_too_large(error):
    """Handle payload too large errors"""
    return jsonify({"error": "Payload too large. Please reduce the size of your request."}), 413

@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit errors"""
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

def initialize_models():
    """Initialize models in background"""
    logger.info("Starting model loading in background...")
    
    # Load detailed summarizer model
    success = model_manager.load_detailed_summarizer_model()
    if not success:
        logger.warning("Failed to load detailed summarizer model")
    
    # Load one-liner summarizer model
    success = model_manager.load_oneliner_summarizer_model()
    if not success:
        logger.warning("Failed to load one-liner summarizer model")
    
    # Load sentiment model
    success = model_manager.load_sentiment_model()
    if not success:
        logger.warning("Failed to load sentiment model")
    
    logger.info("Model loading process completed!")

def initialize_app():
    """Initialize the application"""
    logger.info("Starting Enhanced News Analysis App...")
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    logger.info("App initialization complete!")

# Add some configuration for production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload

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