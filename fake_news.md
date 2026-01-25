# Fake News Detection System - Architecture Documentation

## Overview
A multi-approach text-based fake news detection system that implements three complementary detection methodologies: keyword-based analysis, local LLM-based detection, and a hybrid approach combining both methods.

---

## Architecture Components

### 1. **KeywordFakeNewsDetector** (Approach 1: Simple, Fast, No Dependencies)

#### Purpose
Fast, lightweight detection using pattern matching and keyword analysis without external dependencies.

#### Key Components

**Keyword Dictionary Sets:**
- **Sensational Keywords**: Words indicating exaggeration (shocking, unbelievable, miracle, secret, exposed, conspiracy, etc.)
- **Emotional Keywords**: Language designed to trigger emotional responses (outrage, furious, terrifying, horrifying, etc.)
- **Absolutist Keywords**: Words lacking nuance and context (always, never, everyone, nobody, completely, etc.)
- **Credibility Markers**: Indicators of legitimate sources (according to, research shows, expert, university, peer-reviewed, journal, etc.)
- **Vague Attribution**: Red flags for unsubstantiated claims (some say, sources claim, allegedly, supposedly, etc.)

**Pattern Matching:**
- Clickbait regex patterns detecting common fake news templates:
  - "X reasons/ways/things..."
  - "You won't believe..."
  - "What happened next..."
  - "This one weird trick..."

#### Algorithm Flow
1. Convert text to lowercase for matching
2. Count occurrences of each keyword category
3. Detect clickbait patterns using regex
4. Normalize counts by word count (per 100 words)
5. Calculate weighted red flag score
6. Calculate credibility score
7. Compute final fake probability (0-100 scale)

#### Output
```json
{
  "fake_probability": <0-100>,
  "verdict": "<Likely Credible/Questionable/Likely Fake>",
  "analysis": {
    "sensational_keywords": <count>,
    "emotional_keywords": <count>,
    "absolutist_keywords": <count>,
    "credibility_markers": <count>,
    "vague_attribution": <count>,
    "clickbait_patterns": <count>,
    "word_count": <count>
  },
  "red_flags": ["<flag1>", "<flag2>", ...]
}
```

#### Scoring Formula
- **Red Flag Score** = (sensational × 2.0 + emotional × 1.5 + absolutist × 1.0 + vague × 2.5 + clickbait × 3.0) / normalization_factor
- **Credibility Score** = credibility_count / normalization_factor
- **Fake Probability** = min(100, max(0, (red_flag_score × 10) - (credibility_score × 5)))

#### Verdict Classification
- **< 30**: Likely Credible
- **30-60**: Questionable
- **> 60**: Likely Fake/Misleading

#### Strengths
✓ Fast execution (no dependencies)
✓ Transparent, explainable results
✓ Good at detecting sensationalism and clickbait
✓ Lightweight for real-time analysis

#### Limitations
✗ Limited semantic understanding
✗ May miss sophisticated misinformation
✗ Context-blind keyword matching

---

### 2. **LocalLLMDetector** (Approach 2: Advanced LLM-Based Analysis)

#### Purpose
Leverage a local Large Language Model for sophisticated semantic analysis of content credibility and misinformation patterns.

#### Dependencies
- **Ollama**: Local LLM execution framework
- Supported models: llama2, mistral, phi, llama3, etc.

#### Setup Requirements
```bash
1. Install Ollama: https://ollama.ai/
2. Pull model: ollama pull llama3:latest
3. Install Python client: pip install ollama
```

#### Analysis Methodology
1. Create detailed evaluation prompt instructing the LLM to analyze:
   - Sensationalism and emotional manipulation
   - Source credibility and attribution
   - Absolutist or polarizing language
   - Logical consistency
   - Clickbait indicators

2. Sends article text (capped at 2000 chars) to local LLM with JSON format request
3. Parses JSON response from model

#### Expected Output Format
```json
{
  "fake_probability": <0-100>,
  "verdict": "<Credible/Questionable/Likely Fake>",
  "reasoning": "<brief explanation>",
  "red_flags": ["<flag1>", "<flag2>", ...]
}
```

#### Error Handling
- Gracefully falls back if Ollama is unavailable
- Returns error message with installation instructions
- Assumes neutral score of 50 if LLM returns malformed responses

#### Strengths
✓ Deep semantic understanding
✓ Context-aware analysis
✓ Detects sophisticated misinformation
✓ Provides detailed reasoning
✓ Local execution (privacy-preserving)

#### Limitations
✗ Requires Ollama installation and model download
✗ Slower than keyword-based approach
✗ Dependent on LLM model quality
✗ LLM may produce inconsistent or hallucinated responses

---

### 3. **HybridDetector** (Approach 3: Combined Strategy)

#### Purpose
Maximize accuracy and reliability by combining the speed and explainability of keyword analysis with the sophistication of LLM analysis.

#### Architecture
```
Input Text
    ↓
┌───────────────────────────────────┐
│  KeywordFakeNewsDetector          │  (Fast)
│  • Keyword counting               │
│  • Pattern matching               │
│  • Returns: fake_probability      │
└───────────────────────────────────┘
    ↓ (always runs)
    │
    ├─→ [40% weight]
    │
    ├─ If use_llm=True:
    │   ↓
    │   ┌─────────────────────────────────┐
    │   │  LocalLLMDetector               │  (Sophisticated)
    │   │  • LLM semantic analysis        │
    │   │  • Returns: fake_probability    │
    │   │            verdict, reasoning   │
    │   └─────────────────────────────────┘
    │       ↓ (60% weight)
    │
    └─→ Weighted Average
        ↓
    Final Probability = (Keyword × 0.4) + (LLM × 0.6)
        ↓
    ┌──────────────────────┐
    │  Final Verdict       │
    │  < 30: Credible      │
    │  30-60: Questionable │
    │  > 60: Fake          │
    └──────────────────────┘
```

#### Configuration
- **use_llm** (boolean): Enable/disable LLM analysis
- **model_name** (string): Specify LLM model (default: "llama3:latest")

#### Fallback Strategy
- If LLM analysis fails or unavailable, returns keyword-only result with explanatory note
- Ensures system always provides some level of analysis

#### Output Structure
```json
{
  "fake_probability": <0-100>,
  "verdict": "<Likely Credible/Questionable/Likely Fake>",
  "keyword_analysis": {...},      // Full keyword detector output
  "llm_analysis": {...},           // Full LLM detector output
  "method": "hybrid"
}
```

#### Weighting Rationale
- **Keyword (40%)**: Fast, explainable baseline
- **LLM (60%)**: Prioritizes semantic understanding while maintaining keyword foundation

#### Strengths
✓ Best of both worlds: speed + sophistication
✓ Robust fallback if LLM unavailable
✓ Detailed multi-level analysis
✓ Higher accuracy than individual approaches
✓ Transparent decision-making

#### Limitations
✗ More complex to manage
✗ Slower than keyword-only approach (if LLM enabled)
✗ Inherits limitations of both approaches

---

## Data Flow

### Example: Analyzing an Article

```
Article Text
    ↓
KeywordFakeNewsDetector.analyze()
    ├─ Count sensational/emotional/absolutist keywords
    ├─ Check for vague attribution
    ├─ Match clickbait patterns
    ├─ Normalize by word count
    ├─ Calculate red flag and credibility scores
    └─ Return analysis with verdict and red flags
    ↓
[Optional] LocalLLMDetector.analyze()
    ├─ Create evaluation prompt
    ├─ Send to local LLM (Ollama)
    ├─ Parse JSON response
    └─ Return LLM verdict and reasoning
    ↓
[If Hybrid] Combine Results
    ├─ Weight keyword result (40%)
    ├─ Weight LLM result (60%)
    ├─ Merge all analysis data
    └─ Return final verdict
    ↓
Output (JSON)
```

---

## Usage Example

### Keyword-Based Detection
```python
detector = KeywordFakeNewsDetector()
result = detector.analyze(article_text)
print(f"Fake Probability: {result['fake_probability']}%")
print(f"Verdict: {result['verdict']}")
print(f"Red Flags: {result['red_flags']}")
```

### LLM-Based Detection
```python
llm_detector = LocalLLMDetector(model_name="llama3:latest")
result = llm_detector.analyze(article_text)
print(f"LLM Analysis: {result}")
```

### Hybrid Detection
```python
hybrid = HybridDetector(use_llm=True)
result = hybrid.analyze(article_text)
print(f"Final Probability: {result['fake_probability']}%")
print(f"Verdict: {result['verdict']}")
print(f"Keyword Analysis: {result['keyword_analysis']}")
print(f"LLM Analysis: {result['llm_analysis']}")
```

---

## Key Design Decisions

1. **Three-Tier Approach**: Provides flexibility - use keyword-only for speed, LLM for accuracy, or hybrid for balance
2. **Normalized Scoring**: Word count normalization prevents bias toward longer/shorter texts
3. **Weighted Red Flags**: Different weight for different red flag types based on prevalence in fake news
4. **Graceful Degradation**: System functions without LLM, improving robustness
5. **JSON Output Format**: Structured data enables downstream processing and integration
6. **Local LLM**: Privacy-preserving, no external API calls required

---

## Performance Characteristics

| Approach | Speed | Accuracy | Dependencies | Explainability |
|----------|-------|----------|--------------|-----------------|
| Keyword | Very Fast | Moderate | None | High |
| LLM | Slow | High | Ollama + Model | Medium |
| Hybrid | Medium | High | Ollama (optional) | High |

---

## Recommendations for Use

- **Real-time Analysis**: Use keyword-only detector
- **Critical Assessment**: Use hybrid detector with Ollama
- **Batch Processing**: Use hybrid with LLM for thorough analysis
- **Resource-Constrained Environments**: Use keyword-only detector
- **Privacy-Critical**: Use local LLM detector (Ollama) instead of API-based services

---

## 4. **ImageAuthenticityDetector** (Image-Based Manipulation Detection)

### Purpose
Detect image manipulation, deepfakes, and AI-generated images in news articles using multiple forensic analysis techniques and optional vision LLM analysis.

### Architecture Overview

```
Input Image (jpg, png, etc.)
    ↓
    ├─ Metadata Analysis
    │   └─ EXIF data, editing software, timestamps
    ├─ Error Level Analysis (ELA)
    │   └─ JPEG compression inconsistencies
    ├─ Noise Pattern Analysis
    │   └─ Block-wise noise variance
    ├─ Clone Detection
    │   └─ Block hashing for duplicated regions
    ├─ AI Generation Detection
    │   └─ Heuristic checks for AI artifacts
    └─ [Optional] Vision LLM Analysis
        └─ Semantic understanding via llava/moondream
    ↓
Weighted Score Calculation
    ↓
Output: Authenticity Score + Annotated Image
```

### Core Components

#### 1. **Metadata Analysis**
**Purpose**: Detect EXIF data tampering and editing software usage

**Methods**:
- Extract EXIF tags from image
- Check for editing software (Photoshop, GIMP, etc.)
- Verify timestamp consistency
- Validate camera information presence

**Red Flags**:
- No EXIF data (possibly stripped)
- Inconsistent timestamps (DateTimeOriginal vs DateTimeDigitized)
- Missing camera information
- Presence of editing software tags

**Output**:
```python
{
    'has_metadata': bool,
    'software': str,
    'camera': str,
    'date_taken': str,
    'red_flags': List[str],
    'suspicion_score': 0-100  # Lower = more authentic
}
```

#### 2. **Error Level Analysis (ELA)**
**Purpose**: Detect JPEG compression inconsistencies indicating manipulation

**Algorithm**:
1. Save image as JPEG with known quality (90%)
2. Reload and compare with original
3. Calculate absolute difference between pixels
4. Normalize difference map to 0-255 range
5. Identify regions with top 5% compression errors
6. Flag suspicious high-error regions

**Rationale**: Different JPEG compression levels in different regions indicate they were compressed at different times, suggesting one part was added/edited after the other.

**Output**:
```python
{
    'suspicious_pixel_ratio': float,
    'high_error_regions': int,
    'suspicion_score': 0-100,
    'interpretation': str
}
```

#### 3. **Noise Pattern Analysis**
**Purpose**: Detect inconsistent noise patterns indicating splicing or editing

**Algorithm**:
1. Convert image to grayscale
2. Apply Gaussian blur and subtract (isolate noise)
3. Divide image into 64×64 blocks (50% overlap)
4. Calculate noise variance for each block
5. Identify blocks with >2 std deviations from mean variance
6. Flag regions with unusual noise characteristics

**Rationale**: Spliced regions or edited areas have different noise patterns because they come from different camera sensors or processing pipelines.

**Output**:
```python
{
    'inconsistent_regions': int,
    'mean_noise_variance': float,
    'suspicion_score': 0-100,
    'interpretation': str
}
```

#### 4. **Clone Detection**
**Purpose**: Identify copy-paste cloning within images

**Algorithm**:
1. Resize image for faster processing (max 800px)
2. Convert to grayscale
3. Divide into 32×32 blocks with 50% overlap
4. Compute MD5 hash of each block
5. Find duplicate hashes (with distance verification)
6. Create bounding boxes for cloned regions

**Rationale**: Clone stamping tools are common manipulation methods. Duplicate blocks indicate copy-pasting.

**Output**:
```python
{
    'cloned_regions_found': int,
    'suspicion_score': 0-100,
    'interpretation': str
}
```

#### 5. **AI Generation Detection**
**Purpose**: Identify characteristics of AI-generated images

**Heuristics**:
1. **Color Distribution**: Check for uniform color std (50-70 is AI-typical)
2. **Symmetry**: Detect unnaturally perfect left-right symmetry
3. **Smooth Gradients**: Identify lack of edge details (too smooth)
4. **Aspect Ratio**: Detect standard AI generation ratios (1:1, 3:2, 4:3, etc.)

**Note**: These are heuristics; real AI detection requires specialized neural networks.

**Output**:
```python
{
    'ai_probability': 0-100,
    'red_flags': List[str],
    'suspicion_score': 0-100,
    'note': 'Advanced detection requires neural networks'
}
```

#### 6. **Vision LLM Analysis** (Optional)
**Purpose**: Semantic understanding of image manipulation using vision models

**Dependencies**:
- Ollama with vision model: `llava`, `moondream`, or `bakllava`
- Installation: 
  ```bash
  ollama pull llava
  ollama pull moondream
  ```

**Prompt**: Evaluates:
- Digital manipulation artifacts
- AI-generated characteristics
- Photo editing traces
- Context realism

**Output**:
```python
{
    'authenticity': 'Authentic/Questionable/Likely Fake',
    'confidence': 0-100,
    'reasoning': str,
    'red_flags': List[str],
    'suspicion_score': 0-100
}
```

### Scoring System

#### Weights (without LLM):
- **Metadata**: 15%
- **ELA**: 30% (highest - most reliable)
- **Noise**: 20%
- **Clone**: 25%
- **AI**: 10%

#### Weights (with LLM):
- **Metadata**: 10%
- **ELA**: 20%
- **Noise**: 15%
- **Clone**: 15%
- **AI**: 5%
- **LLM**: 35% (highest - semantic understanding)

#### Formula:
```
Authenticity Score = Σ(Score_i × Weight_i) for all components
Authenticity Range: 0-100 (100 = definitely authentic)
Manipulation Probability = 100 - Authenticity Score
```

#### Verdict Classification:
- **Authenticity ≥ 70%**: "Likely Authentic"
- **Authenticity 40-70%**: "Questionable - Possible Manipulation"
- **Authenticity < 40%**: "Likely Manipulated"

### Data Structures

#### SuspiciousRegion
```python
@dataclass
class SuspiciousRegion:
    x: int                # X coordinate in image
    y: int                # Y coordinate in image
    width: int            # Region width
    height: int           # Region height
    confidence: float     # 0-100 confidence score
    reason: str           # Detection reason (e.g., "ELA: Compression inconsistency")
    color: str            # Annotation color (red/orange/yellow)
```

Each analysis method creates multiple SuspiciousRegion objects, which are later visualized in the annotated image.

### Output Artifacts

#### Main Analysis Result:
```json
{
  "authenticity_score": 0-100,
  "verdict": "Likely Authentic/Questionable/Likely Manipulated",
  "manipulated_probability": 0-100,
  "analyses": {
    "metadata": {...},
    "error_level_analysis": {...},
    "noise_patterns": {...},
    "clone_detection": {...},
    "ai_detection": {...},
    "llm_analysis": {...}  // Optional
  },
  "suspicious_regions": int,
  "red_flags": List[str],
  "annotated_image_path": "path/to/annotated_image.jpg"
}
```

#### Annotated Image:
- Original image with highlighted suspicious regions
- Color-coded confidence levels:
  - **Red (>75% confidence)**: High probability of manipulation
  - **Orange (50-75%)**: Medium confidence
  - **Yellow (<50%)**: Low confidence
- Numbered labels and percentage confidence
- Saved as `{original_name}_annotated.jpg`

### Usage Examples

#### Basic Image Analysis:
```python
from img_analysis import ImageAuthenticityDetector

detector = ImageAuthenticityDetector(use_llm=False)
result = detector.analyze('news_image.jpg')

print(f"Authenticity: {result['authenticity_score']}%")
print(f"Verdict: {result['verdict']}")
print(f"Red Flags: {result['red_flags']}")
```

#### With Vision LLM:
```python
detector = ImageAuthenticityDetector(use_llm=True, llm_model="llava")
result = detector.analyze('news_image.jpg', save_annotated=True)
```

#### Batch Analysis:
```python
import os

image_dir = "news_images/"
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.png')):
        result = detector.analyze(os.path.join(image_dir, image_file))
        print(f"{image_file}: {result['verdict']} ({result['authenticity_score']}%)")
```

### Strengths
✓ Multiple complementary analysis techniques
✓ Detects various manipulation types (splicing, cloning, deepfakes)
✓ Optional vision LLM for semantic understanding
✓ Visual output (annotated images) for transparency
✓ Granular suspicious region detection
✓ Local execution (no cloud dependencies)
✓ Handles EXIF tampering detection

### Limitations
✗ Simple heuristics for AI detection (specialized networks needed)
✗ Clone detection uses basic block hashing (sophisticated cloning may evade)
✗ ELA works better on JPEG; less effective on PNG/BMP
✗ Requires PIL/Pillow and NumPy
✗ LLM analysis adds latency (5-30s per image)
✗ May produce false positives on heavily compressed originals

### Performance Characteristics

| Technique | Speed | Accuracy | Notes |
|-----------|-------|----------|-------|
| Metadata | <10ms | Medium | Fast but only works if EXIF present |
| ELA | 100-500ms | High | Most reliable for JPEG manipulation |
| Noise | 200-1000ms | Medium | Good for splicing detection |
| Clone | 1-5s | Medium | Depends on image resolution |
| AI Detection | <50ms | Low | Heuristics only; not reliable |
| Vision LLM | 5-30s | High | Slowest but most comprehensive |

### Dependencies

**Required**:
```
Pillow>=9.0.0  # Image processing
numpy>=1.20.0  # Numerical operations
scipy>=1.5.0   # Connected component labeling
```

**Optional**:
```
ollama>=0.0.11  # Vision LLM support (for llava, moondream, etc.)
```

### Integration with Text Detection

The image analyzer complements the text detector in a complete fake news detection pipeline:

```
Article
├─ Text Analysis (text_detector.py)
│  └─ Keyword/LLM-based credibility assessment
│
├─ Image Analysis (img_analysis.py)
│  └─ Visual authenticity verification
│
└─ Combined Verdict
   └─ Article is suspicious if EITHER text OR images raise concerns
```

This multi-modal approach catches both textual misinformation and manipulated imagery.
