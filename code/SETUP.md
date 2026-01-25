# Setup & Run Guide - Fake News Detection Web UI

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python app.py
```

You should see:
```
======================================================================
Fake News Detection API Server
======================================================================

API Endpoints:
  ✓ POST /api/keyword-detect  - Fast keyword-based detection
  ✓ POST /api/llm-detect      - LLM-based detection (requires Ollama)
  ✓ POST /api/hybrid-detect   - Hybrid detection (keyword + LLM)
  ✓ GET  /api/health          - Health check

Server running on: http://localhost:5000
Frontend: http://localhost:5000/static/index.html

To stop the server, press Ctrl+C
======================================================================
```

### 3. Open the Web UI
- **Direct file access**: Open `index.html` in your browser
- **Via server**: Navigate to `http://localhost:5000` (if serving static files)
- Or open the browser console and use the API directly

## API Endpoints

### 1. Keyword-Based Detection (Fastest)
**No dependencies required**

```bash
curl -X POST http://localhost:5000/api/keyword-detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here..."}'
```

**Response:**
```json
{
  "success": true,
  "method": "keyword-based",
  "result": {
    "fake_probability": 45.5,
    "verdict": "Questionable",
    "analysis": {
      "sensational_keywords": 2,
      "emotional_keywords": 3,
      "absolutist_keywords": 5,
      "credibility_markers": 0,
      "vague_attribution": 1,
      "clickbait_patterns": 0,
      "word_count": 250
    },
    "red_flags": [
      "High sensationalism (2 keywords)",
      "Emotional manipulation detected (3 keywords)"
    ]
  }
}
```

### 2. LLM-Based Detection (Most Accurate)
**Requires: Ollama + LLM Model**

Setup:
```bash
# Install Ollama from https://ollama.ai/
# Pull a model
ollama pull llama3:latest
# Or: ollama pull mistral
# Or: ollama pull phi
```

API Call:
```bash
curl -X POST http://localhost:5000/api/llm-detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your article text here...",
    "model": "llama3:latest"
  }'
```

**Response:**
```json
{
  "success": true,
  "method": "llm-based",
  "model": "llama3:latest",
  "result": {
    "fake_probability": 72,
    "verdict": "Likely Fake/Misleading",
    "reasoning": "Article shows heavy use of sensationalism, vague attribution, and emotional language without credible sources",
    "red_flags": [
      "Unsubstantiated claims",
      "Emotional manipulation tactics",
      "Lack of credible sources"
    ]
  }
}
```

### 3. Hybrid Detection (Best of Both)
**Combines keyword + LLM for optimal results**

```bash
curl -X POST http://localhost:5000/api/hybrid-detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your article text here...",
    "use_llm": true,
    "model": "llama3:latest"
  }'
```

**Response:**
```json
{
  "success": true,
  "method": "hybrid",
  "use_llm": true,
  "model": "llama3:latest",
  "result": {
    "fake_probability": 58.5,
    "verdict": "Questionable",
    "keyword_analysis": { ... },
    "llm_analysis": { ... }
  }
}
```

## Web UI Features

### Detection Methods
1. **Keyword-Based** - Fast analysis (instant)
   - No dependencies needed
   - Transparent, explainable results
   - Good for rapid screening

2. **LLM-Based** - Sophisticated analysis (slower)
   - Requires Ollama installation
   - Semantic understanding
   - More accurate for complex misinformation

3. **Hybrid** - Best of both worlds
   - Combines speed and accuracy
   - Graceful fallback if LLM unavailable
   - Recommended for production use

### UI Elements
- **Article Input**: Paste text to analyze
- **Method Selector**: Choose detection approach
- **Probability Bar**: Visual indicator of fake probability
- **Verdict Badge**: Color-coded result (Green/Yellow/Red)
- **Red Flags List**: Specific issues detected
- **Analysis Breakdown**: Detailed metrics
- **API Status**: Real-time health check

### Keyboard Shortcuts
- `Ctrl + Enter`: Analyze article
- Clear button: Reset form and results

## Integration with Your Own Code

### Using the Python Classes Directly
```python
from text_detector import KeywordFakeNewsDetector, HybridDetector

# Simple keyword detection
detector = KeywordFakeNewsDetector()
result = detector.analyze("Your article text")
print(result['fake_probability'])
print(result['verdict'])
print(result['red_flags'])
```

### Using the REST API from JavaScript
```javascript
async function analyzeArticle(text) {
  const response = await fetch('http://localhost:5000/api/hybrid-detect', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      text: text,
      use_llm: true
    })
  });
  
  const data = await response.json();
  return data.result;
}

// Usage
analyzeArticle("Article text here").then(result => {
  console.log(`Fake Probability: ${result.fake_probability}%`);
  console.log(`Verdict: ${result.verdict}`);
});
```

## Troubleshooting

### "API is offline" or "Connection refused"
**Problem**: Server isn't running
**Solution**:
```bash
python app.py
```

### LLM Detection throws error
**Problem**: Ollama not installed
**Solution**:
```bash
# Install Ollama
# Download from https://ollama.ai/

# Pull a model
ollama pull llama3:latest

# Verify Ollama is running
curl http://localhost:11434/api/models
```

### CORS errors
**Problem**: Frontend can't communicate with API
**Solution**: 
- Make sure Flask-CORS is installed: `pip install flask-cors`
- The `app.py` already has CORS enabled
- Try accessing from the same origin

### Slow LLM Response
**Problem**: LLM analysis takes too long
**Solution**:
- Use a smaller model: `ollama pull phi` (much faster)
- Use keyword-only detection instead
- Increase timeout or reduce text size

## Production Deployment

### Option 1: Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 2: Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY text_detector.py .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

### Option 3: Nginx + Gunicorn
Setup reverse proxy to handle multiple requests and SSL

## File Structure
```
code/
├── text_detector.py       # Core detection classes
├── app.py                 # Flask API server
├── index.html             # Web UI
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Security Considerations

1. **Input Validation**: The API validates text input length
2. **CORS**: Configured to allow cross-origin requests (update as needed)
3. **Rate Limiting**: Consider adding rate limiting for production
4. **Text Limits**: API caps text at 2000 chars for LLM analysis
5. **Model Security**: Ollama runs locally (no external API calls)

## Performance Tips

1. **Keyword Detection**: ~5-50ms per article
2. **LLM Detection**: 2-10 seconds depending on model and hardware
3. **Hybrid Detection**: ~3-50ms (if LLM unavailable, falls back to keyword)

### Optimize for Speed:
- Use `keyword-detect` for real-time screening
- Use smaller LLM models (phi, mistral)
- Cache results for duplicate texts
- Batch process articles if possible

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run server: `python app.py`
3. ✅ Open UI: Open `index.html` in browser
4. ✅ Test with sample articles
5. ✅ Integrate into your application

For advanced usage, see [API Documentation](./SETUP.md) or examine the code comments in `text_detector.py` and `app.py`.
