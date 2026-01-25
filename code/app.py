"""
Flask REST API for Fake News Detection System
Exposes endpoints to call detectors from HTML/JS frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from text_detector import KeywordFakeNewsDetector, LocalLLMDetector, HybridDetector

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend

# Initialize detectors
keyword_detector = KeywordFakeNewsDetector()
llm_detector = LocalLLMDetector(model_name="llama3:latest")
hybrid_detector = HybridDetector(use_llm=True, model_name="llama3:latest")


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Fake News Detection API is running',
        'available_endpoints': [
            'POST /api/keyword-detect',
            'POST /api/llm-detect',
            'POST /api/hybrid-detect',
            'GET /api/health'
        ]
    }), 200


# ============================================================================
# KEYWORD-BASED DETECTION ENDPOINT
# ============================================================================

@app.route('/api/keyword-detect', methods=['POST'])
def keyword_detect():
    """
    Keyword-based fake news detection
    
    Request JSON:
    {
        "text": "article text here"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'example': {'text': 'Your article content here'}
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'error': 'Text field cannot be empty'
            }), 400
        
        # Run analysis
        result = keyword_detector.analyze(text)
        
        return jsonify({
            'success': True,
            'method': 'keyword-based',
            'result': result
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# LLM-BASED DETECTION ENDPOINT
# ============================================================================

@app.route('/api/llm-detect', methods=['POST'])
def llm_detect():
    """
    Local LLM-based fake news detection (requires Ollama)
    
    Request JSON:
    {
        "text": "article text here",
        "model": "llama3:latest"  (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'example': {'text': 'Your article content here'}
            }), 400
        
        text = data['text'].strip()
        model = data.get('model', 'llama3:latest')
        
        if not text:
            return jsonify({
                'error': 'Text field cannot be empty'
            }), 400
        
        # Use specified model
        detector = LocalLLMDetector(model_name=model)
        
        # Run analysis
        result = detector.analyze(text)
        
        return jsonify({
            'success': True,
            'method': 'llm-based',
            'model': model,
            'result': result
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# HYBRID DETECTION ENDPOINT
# ============================================================================

@app.route('/api/hybrid-detect', methods=['POST'])
def hybrid_detect():
    """
    Hybrid detection (keyword + LLM)
    
    Request JSON:
    {
        "text": "article text here",
        "use_llm": true,
        "model": "llama3:latest"  (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'example': {'text': 'Your article content here'}
            }), 400
        
        text = data['text'].strip()
        use_llm = data.get('use_llm', True)
        model = data.get('model', 'llama3:latest')
        
        if not text:
            return jsonify({
                'error': 'Text field cannot be empty'
            }), 400
        
        # Create detector with specified options
        detector = HybridDetector(use_llm=use_llm, model_name=model)
        
        # Run analysis
        result = detector.analyze(text)
        
        return jsonify({
            'success': True,
            'method': 'hybrid',
            'use_llm': use_llm,
            'model': model if use_llm else None,
            'result': result
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'POST /api/keyword-detect',
            'POST /api/llm-detect',
            'POST /api/hybrid-detect',
            'GET /api/health'
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Fake News Detection API Server")
    print("=" * 70)
    print("\nAPI Endpoints:")
    print("  ✓ POST /api/keyword-detect  - Fast keyword-based detection")
    print("  ✓ POST /api/llm-detect      - LLM-based detection (requires Ollama)")
    print("  ✓ POST /api/hybrid-detect   - Hybrid detection (keyword + LLM)")
    print("  ✓ GET  /api/health          - Health check")
    print("\nServer running on: http://localhost:5000")
    print("Frontend: http://localhost:5000/static/index.html")
    print("\nDocumentation: http://localhost:5000/api/health")
    print("\nTo stop the server, press Ctrl+C")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
