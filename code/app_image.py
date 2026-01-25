"""
Flask REST API for Image Authenticity Detection
Exposes endpoints to call image detector from HTML/JS frontend
Supports three detection methods: image-analysis, vision-llm, and hybrid
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def import_detectors():
    """Attempt to import detector classes from img_analysis.
    Raises ImportError with a helpful message if unavailable.
    """
    try:
        from img_analysis import ImageAnalysisDetector, VisionLLMDetector, HybridImageDetector
        return ImageAnalysisDetector, VisionLLMDetector, HybridImageDetector
    except Exception as e:
        raise ImportError(f"Failed to import img_analysis detectors: {e}")


app = Flask(__name__)

# Set the default JSON encoder to handle numpy types
app.json.encoder = NumpyEncoder

# Also override the json encoder for older Flask versions
import flask
if hasattr(flask.json, 'provider'):
    # Flask 2.3+
    class CustomJSONProvider(flask.json.provider.DefaultJSONProvider):
        def default(self, o):
            if isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    app.json = CustomJSONProvider(app)
else:
    # Older Flask versions
    app.json_encoder = NumpyEncoder

CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_image_to_base64(image_path):
    """Convert image to base64 for embedding in JSON response"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None


def format_result(result, include_base64=False):
    """Format detector result for JSON response"""
    if 'error' in result:
        return result
    
    # Convert suspicious regions to dict
    if 'suspicious_regions' in result:
        region_count = result['suspicious_regions']
        result['suspicious_regions'] = region_count
    
    return result


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Image Authenticity Detection API is running',
        'available_endpoints': [
            'POST /api/analyze-image',
            'POST /api/analyze-image-comparison',
            'POST /api/batch-analyze',
            'GET /api/annotated/<filename>',
            'GET /api/health'
        ],
        'supported_methods': [
            'image-analysis (Computer vision forensics)',
            'llm (Vision LLM semantic analysis)',
            'hybrid (Combined approach - recommended)'
        ],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024)
    }), 200


# ============================================================================
# BASIC IMAGE ANALYSIS ENDPOINT
# ============================================================================

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    Analyze an uploaded image for manipulation using specified method
    
    Request: multipart/form-data
    - file: Image file (required)
    - method: Detection method (optional, default: 'hybrid')
      - 'image-analysis': Computer vision forensics only (fast)
      - 'llm': Vision LLM semantic analysis only (requires Ollama)
      - 'hybrid': Combined approach (recommended)
    - llm_model: LLM model name (optional, default: 'llava', used if method='llm' or 'hybrid')
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'example': 'Send file in form-data with key "file"'
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File too large. Max: {MAX_FILE_SIZE / (1024 * 1024)}MB'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get parameters
        method = request.form.get('method', 'hybrid').lower()
        llm_model = request.form.get('llm_model', 'llava')
        include_base64 = request.form.get('include_base64', 'false').lower() == 'true'
        
        # Validate method
        if method not in ['image-analysis', 'llm', 'hybrid']:
            return jsonify({
                'success': False,
                'error': f'Invalid method: {method}. Must be: image-analysis, llm, or hybrid'
            }), 400
        # Import detectors (lazy) and run analysis based on method
        try:
            ImageAnalysisDetector, VisionLLMDetector, HybridImageDetector = import_detectors()
        except ImportError as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'note': 'Detector module or dependencies may be missing. Ensure img_analysis.py and its dependencies are available.'
            }), 500

        if method == 'image-analysis':
            detector = ImageAnalysisDetector()
            result = detector.analyze(filepath)

        elif method == 'llm':
            detector = VisionLLMDetector(model_name=llm_model)
            result = detector.analyze(filepath)

        else:  # hybrid
            detector = HybridImageDetector(use_llm=True, llm_model=llm_model)
            result = detector.analyze(filepath)
        
        if 'error' in result and result.get('authenticity_score') is None:
            return jsonify({
                'success': False,
                'error': result['error'],
                'note': 'Make sure Ollama is running for LLM methods'
            }), 500
        
        # Format response
        response = {
            'success': True,
            'method': method,
            'file': filename,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'llm_model': llm_model if method in ['llm', 'hybrid'] else None,
            'result': result,
            'authenticity_score': result.get('authenticity_score'),
            'verdict': result.get('verdict'),
            'red_flags': result.get('red_flags', []),
            'statistics': result.get('statistics', {})
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ADVANCED IMAGE ANALYSIS - DETAILED COMPARISON
# ============================================================================

@app.route('/api/analyze-image-comparison', methods=['POST'])
def analyze_image_comparison():
    """
    Compare all three detection methods on the same image
    Returns results from image-analysis, llm, and hybrid approaches
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get LLM model
        llm_model = request.form.get('llm_model', 'llava')
        
        # Lazy import detectors
        try:
            ImageAnalysisDetector, VisionLLMDetector, HybridImageDetector = import_detectors()
        except ImportError as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'note': 'Detector module or dependencies may be missing.'
            }), 500

        results = {}
        
        # 1. Image Analysis only
        try:
            detector = ImageAnalysisDetector()
            results['image_analysis'] = detector.analyze(filepath)
        except Exception as e:
            results['image_analysis'] = {'error': str(e)}
        
        # 2. Vision LLM only
        try:
            detector = VisionLLMDetector(model_name=llm_model)
            results['vision_llm'] = detector.analyze(filepath)
        except Exception as e:
            results['vision_llm'] = {'error': str(e)}
        
        # 3. Hybrid
        try:
            detector = HybridImageDetector(use_llm=True, llm_model=llm_model)
            results['hybrid'] = detector.analyze(filepath)
        except Exception as e:
            results['hybrid'] = {'error': str(e)}
        
        response = {
            'success': True,
            'method': 'comparison',
            'file': filename,
            'llm_model': llm_model,
            'results': results,
            'recommendation': _get_recommendation(results)
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _get_recommendation(results):
    """Generate recommendation based on comparison"""
    try:
        hybrid = results.get('hybrid', {})
        if 'error' not in hybrid and hybrid.get('authenticity_score'):
            score = hybrid['authenticity_score']
            if score >= 70:
                return f"✅ Image likely AUTHENTIC ({score}%)"
            elif score >= 40:
                return f"⚠️  Image is QUESTIONABLE ({score}%)"
            else:
                return f"❌ Image likely MANIPULATED ({score}%)"
    except:
        pass
    return "Unable to determine"


# ============================================================================
# RETRIEVE ANNOTATED IMAGE
# ============================================================================

@app.route('/api/annotated/<filename>', methods=['GET'])
def get_annotated_image(filename):
    """Download annotated image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        # Verify file exists and is in upload folder
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return jsonify({
                'error': 'File not found'
            }), 404
        
        # Verify it's an annotated image
        if '_annotated' not in filename:
            return jsonify({
                'error': 'Invalid file'
            }), 403
        
        return send_file(filepath, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# ============================================================================
# BATCH ANALYSIS ENDPOINT
# ============================================================================

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple images with specified method
    
    Request: multipart/form-data
    - files: Multiple image files
    - method: Detection method (optional, default: 'hybrid')
    - llm_model: LLM model name if using LLM methods
    """
    try:
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        method = request.form.get('method', 'hybrid').lower()
        llm_model = request.form.get('llm_model', 'llava')
        
        if method not in ['image-analysis', 'llm', 'hybrid']:
            return jsonify({
                'success': False,
                'error': f'Invalid method: {method}'
            }), 400
        
        # Lazy import detectors
        try:
            ImageAnalysisDetector, VisionLLMDetector, HybridImageDetector = import_detectors()
        except ImportError as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'note': 'Detector module or dependencies may be missing.'
            }), 500

        results = []
        
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'file': file.filename,
                    'success': False,
                    'error': 'File type not allowed'
                })
                continue
            
            try:
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze based on method
                if method == 'image-analysis':
                    detector = ImageAnalysisDetector()
                    result = detector.analyze(filepath)
                
                elif method == 'llm':
                    detector = VisionLLMDetector(model_name=llm_model)
                    result = detector.analyze(filepath)
                
                else:  # hybrid
                    detector = HybridImageDetector(use_llm=True, llm_model=llm_model)
                    result = detector.analyze(filepath)
                
                if 'error' in result and result.get('authenticity_score') is None:
                    results.append({
                        'file': filename,
                        'success': False,
                        'error': result['error']
                    })
                else:
                    results.append({
                            'file': filename,
                            'success': True,
                            'authenticity_score': result.get('authenticity_score'),
                            'verdict': result.get('verdict'),
                            'manipulated_probability': result.get('manipulated_probability'),
                            'red_flags': result.get('red_flags', []),
                            'statistics': result.get('statistics', {})
                        })
            
            except Exception as e:
                results.append({
                    'file': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'method': method,
            'llm_model': llm_model if 'llm' in method else None,
            'total_files': len(files),
            'successful': len([r for r in results if r.get('success')]),
            'results': results
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
            'POST /api/analyze-image',
            'POST /api/analyze-image-comparison',
            'POST /api/batch-analyze',
            'GET /api/annotated/<filename>',
            'GET /api/health'
        ]
    }), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum: {MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413


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
    print("Image Authenticity Detection API Server")
    print("=" * 70)
    print("\nAPI Endpoints:")
    print("  ✓ POST /api/analyze-image           - Analyze with specified method")
    print("  ✓ POST /api/analyze-image-comparison- Compare all three methods")
    print("  ✓ POST /api/batch-analyze           - Process multiple images")
    print("  ✓ GET  /api/annotated/<filename>    - Download annotated image")
    print("  ✓ GET  /api/health                  - Health check")
    print("\nDetection Methods:")
    print("  • image-analysis - Computer vision forensics (fast, no dependencies)")
    print("  • llm            - Vision LLM semantic analysis (requires Ollama)")
    print("  • hybrid         - Combined approach (40% image + 60% LLM, recommended)")
    print("\nServer running on: http://localhost:5001")
    print("Documentation: http://localhost:5001/api/health")
    print("\nNote: For LLM methods, ensure Ollama is running locally")
    print("To stop the server, press Ctrl+C")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
