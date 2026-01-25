"""
Image Authenticity Detection for News Articles
Three approaches: Image Analysis, Vision LLM, and Hybrid

Features:
1. Image Analysis Detection - Computer vision forensic techniques
2. Vision LLM Detection - Semantic analysis via local LLM
3. Hybrid Detection - Combines both approaches
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PIL.ExifTags import TAGS
import io
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SuspiciousRegion:
    """Represents a suspicious area in the image"""
    x: int
    y: int
    width: int
    height: int
    confidence: float  # 0-100
    reason: str
    color: str = "red"


# ============================================================================
# APPROACH 1: IMAGE ANALYSIS DETECTION (Computer Vision Forensics)
# ============================================================================

class ImageAnalysisDetector:
    """Forensic image analysis using computer vision techniques"""

    def __init__(self):
        self.suspicious_regions = []
        self.stats = {}

    def analyze(self, image_path: str) -> Dict:
        """Complete image analysis without LLM"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            print(f"Analyzing image: {image_path}")

            # Run all CV-based analyses
            metadata_stats = self._analyze_metadata(image_path)
            ela_stats = self._error_level_analysis(img)
            noise_stats = self._analyze_noise_patterns(img)
            clone_stats = self._detect_cloning(img)
            ai_stats = self._detect_ai_generation(img)

            # Combine statistics
            authenticity_score = self._calculate_authenticity_score(
                metadata_stats, ela_stats, noise_stats, clone_stats, ai_stats
            )

            verdict = self._determine_verdict(authenticity_score)

            return {
                'authenticity_score': round(authenticity_score, 2),
                'verdict': verdict,
                'manipulated_probability': round(100 - authenticity_score, 2),
                'statistics': {
                    'metadata': metadata_stats,
                    'error_level_analysis': ela_stats,
                    'noise_patterns': noise_stats,
                    'clone_detection': clone_stats,
                    'ai_detection': ai_stats
                },
                'suspicious_regions': len(self.suspicious_regions),
                'red_flags': self._collect_red_flags(),
                'suspicious_regions_list': [asdict(r) for r in self.suspicious_regions]
            }

        except Exception as e:
            return {'error': str(e), 'authenticity_score': None}

    def _analyze_metadata(self, image_path: str) -> Dict:
        """Analyze EXIF metadata"""
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()

            if not exif_data:
                return {
                    'score': 30,
                    'has_metadata': False,
                    'issues': ['No EXIF data - possibly stripped'],
                    'risk_level': 'High'
                }

            metadata = {}
            issues = []

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                metadata[tag_name] = str(value)[:100]

            # Check for editing software
            software = metadata.get('Software', '').lower()
            if any(editor in software for editor in ['photoshop', 'gimp', 'paint', 'edited']):
                issues.append(f"Edited with: {metadata.get('Software')}")

            # Check for inconsistent timestamps
            date_original = metadata.get('DateTimeOriginal')
            date_digitized = metadata.get('DateTimeDigitized')
            if date_original and date_digitized and date_original != date_digitized:
                issues.append("Inconsistent timestamps")

            if 'Make' not in metadata and 'Model' not in metadata:
                issues.append("Camera info missing")

            score = max(0, 100 - len(issues) * 20)

            return {
                'score': score,
                'has_metadata': True,
                'camera': f"{metadata.get('Make', 'Unknown')} {metadata.get('Model', '')}",
                'date': metadata.get('DateTimeOriginal', 'Unknown'),
                'issues': issues,
                'risk_level': 'Low' if score >= 70 else 'Medium' if score >= 40 else 'High'
            }

        except Exception as e:
            return {'score': 50, 'error': str(e), 'risk_level': 'Medium'}

    def _error_level_analysis(self, img: Image.Image) -> Dict:
        """ELA - Detect JPEG compression inconsistencies"""
        try:
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, format='JPEG', quality=90)
            temp_buffer.seek(0)
            compressed = Image.open(temp_buffer)

            original_array = np.array(img, dtype=np.float32)
            compressed_array = np.array(compressed, dtype=np.float32)
            diff = np.abs(original_array - compressed_array)
            diff_normalized = (diff / (diff.max() + 1e-5) * 255).astype(np.uint8)

            threshold = np.percentile(diff_normalized, 95)
            high_error_mask = diff_normalized > threshold
            suspicious_ratio = np.sum(high_error_mask) / high_error_mask.size

            self._find_suspicious_regions_from_mask(
                high_error_mask[:, :, 0] if len(high_error_mask.shape) == 3 else high_error_mask,
                "ELA: Compression anomaly", 60
            )

            suspicion = min(suspicious_ratio * 500, 100)
            score = 100 - suspicion

            return {
                'score': round(score, 2),
                'suspicious_ratio': round(suspicious_ratio, 4),
                'high_error_regions': len([r for r in self.suspicious_regions if 'ELA' in r.reason]),
                'interpretation': 'Consistent' if score > 70 else 'Some editing' if score > 40 else 'Heavily manipulated',
                'risk_level': 'Low' if score > 70 else 'Medium' if score > 40 else 'High'
            }

        except Exception as e:
            return {'score': 50, 'error': str(e), 'risk_level': 'Medium'}

    def _analyze_noise_patterns(self, img: Image.Image) -> Dict:
        """Analyze noise consistency"""
        try:
            gray = img.convert('L')
            gray_array = np.array(gray, dtype=np.float32)
            blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
            blurred_array = np.array(blurred, dtype=np.float32)
            noise = gray_array - blurred_array

            block_size = 64
            height, width = noise.shape
            noise_variances = []

            for y in range(0, height - block_size, block_size):
                for x in range(0, width - block_size, block_size):
                    block = noise[y:y + block_size, x:x + block_size]
                    variance = np.var(block)
                    noise_variances.append((x, y, variance))

            if not noise_variances:
                return {'score': 100, 'inconsistent_regions': 0, 'risk_level': 'Low'}

            variances = [v[2] for v in noise_variances]
            mean_var = np.mean(variances)
            std_var = np.std(variances)

            inconsistent_count = 0
            for x, y, var in noise_variances:
                if abs(var - mean_var) > 2 * std_var:
                    self.suspicious_regions.append(SuspiciousRegion(
                        x=x, y=y, width=block_size, height=block_size,
                        confidence=min(abs(var - mean_var) / (std_var + 1e-5) * 30, 100),
                        reason="Noise: Pattern inconsistency", color="yellow"
                    ))
                    inconsistent_count += 1

            score = max(0, 100 - inconsistent_count * 5)

            return {
                'score': round(score, 2),
                'inconsistent_regions': inconsistent_count,
                'mean_variance': round(mean_var, 2),
                'interpretation': 'Natural' if score > 70 else 'Suspicious',
                'risk_level': 'Low' if score > 70 else 'Medium' if score > 40 else 'High'
            }

        except Exception as e:
            return {'score': 50, 'error': str(e), 'risk_level': 'Medium'}

    def _detect_cloning(self, img: Image.Image) -> Dict:
        """Detect copy-paste cloning"""
        try:
            max_size = 800
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img_small = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_small = img

            gray = img_small.convert('L')
            gray_array = np.array(gray)

            block_size = 32
            height, width = gray_array.shape
            block_hashes = {}
            clone_count = 0

            for y in range(0, height - block_size, block_size // 2):
                for x in range(0, width - block_size, block_size // 2):
                    block = gray_array[y:y + block_size, x:x + block_size]
                    block_hash = hashlib.md5(block.tobytes()).hexdigest()

                    if block_hash in block_hashes:
                        orig_x, orig_y = block_hashes[block_hash]
                        distance = np.sqrt((x - orig_x) ** 2 + (y - orig_y) ** 2)

                        if distance > block_size:
                            if max(img.size) > max_size:
                                scale_x = img.size[0] / img_small.size[0]
                                scale_y = img.size[1] / img_small.size[1]
                                x_orig = int(x * scale_x)
                                y_orig = int(y * scale_y)
                                size_orig = int(block_size * scale_x)
                            else:
                                x_orig, y_orig, size_orig = x, y, block_size

                            self.suspicious_regions.append(SuspiciousRegion(
                                x=x_orig, y=y_orig, width=size_orig, height=size_orig,
                                confidence=75, reason="Clone: Duplicated region", color="orange"
                            ))
                            clone_count += 1
                    else:
                        block_hashes[block_hash] = (x, y)

            score = max(0, 100 - clone_count * 25)

            return {
                'score': round(score, 2),
                'cloned_regions': clone_count,
                'interpretation': 'No cloning' if clone_count == 0 else f'{clone_count} suspicious clone(s)',
                'risk_level': 'Low' if clone_count == 0 else 'High'
            }

        except Exception as e:
            return {'score': 50, 'error': str(e), 'risk_level': 'Medium'}

    def _llm_analysis(self, image_path: str) -> Dict:
        """
        Use local vision LLM to analyze image for manipulation
        Requires Ollama with a vision model (llava, moondream, bakllava)
        """
        if not self.llm_available:
            return {
                'error': 'LLM not available',
                'suspicion_score': 0
            }

        try:
            prompt = """Analyze this image carefully for signs of manipulation or fakeness.

Look for:
1. Digital manipulation artifacts (unnatural edges, inconsistent lighting, clone stamping)
2. AI-generated characteristics (too perfect, unnatural details, impossible physics)
3. Photo editing traces (color mismatches, perspective issues, shadow inconsistencies)
4. Context clues (does this look like a real photograph or generated/edited?)

Respond in this format:
AUTHENTICITY: [Authentic/Questionable/Likely Fake]
CONFIDENCE: [0-100]%
REASONING: [Brief explanation]
RED FLAGS: [List specific issues found, or "None" if authentic]
"""

            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )

            analysis_text = response['message']['content']

            # Parse the response
            authenticity = "Unknown"
            confidence = 50
            reasoning = ""
            red_flags = []

            lines = analysis_text.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'authenticity:' in line_lower:
                    if 'authentic' in line_lower and 'likely fake' not in line_lower:
                        authenticity = "Authentic"
                    elif 'questionable' in line_lower:
                        authenticity = "Questionable"
                    elif 'fake' in line_lower or 'manipulated' in line_lower:
                        authenticity = "Likely Fake"

                elif 'confidence:' in line_lower:
                    # Extract percentage
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        confidence = int(match.group(1))

                elif 'reasoning:' in line_lower:
                    reasoning = line.split(':', 1)[1].strip()

                elif 'red flags:' in line_lower:
                    flags_text = line.split(':', 1)[1].strip()
                    if flags_text.lower() != 'none':
                        red_flags = [f.strip() for f in flags_text.split(',')]

            # Convert to suspicion score
            if authenticity == "Authentic":
                suspicion_score = 100 - confidence
            elif authenticity == "Questionable":
                suspicion_score = 50
            else:  # Likely Fake
                suspicion_score = confidence

            return {
                'authenticity': authenticity,
                'confidence': confidence,
                'reasoning': reasoning,
                'red_flags': red_flags,
                'suspicion_score': suspicion_score,
                'raw_response': analysis_text
            }

        except Exception as e:
            return {
                'error': str(e),
                'suspicion_score': 0
            }

    def _detect_ai_generation(self, img: Image.Image) -> Dict:
        """Detect AI-generated image heuristics"""
        try:
            img_array = np.array(img)
            red_flags = []
            suspicion = 0

            # Check 1: Color distribution
            color_std = np.std(img_array, axis=(0, 1))
            if np.all(color_std > 50) and np.all(color_std < 70):
                red_flags.append("Uniform color distribution")
                suspicion += 20

            # Check 2: Symmetry
            left_half = img_array[:, :img_array.shape[1] // 2]
            right_half = np.fliplr(img_array[:, img_array.shape[1] // 2:])
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))

            if symmetry_diff < 10:
                red_flags.append("Unnatural symmetry")
                suspicion += 30

            # Check 3: Edge density
            edges = img.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges.convert('L'))
            edge_density = np.mean(edges_array > 20)

            if edge_density < 0.1:
                red_flags.append("Overly smooth gradients")
                suspicion += 25

            # Check 4: Aspect ratio
            width, height = img.size
            ratio = width / height
            common_ai_ratios = [1.0, 1.5, 0.75, 1.333, 0.667]

            if any(abs(ratio - r) < 0.01 for r in common_ai_ratios):
                red_flags.append("Standard AI aspect ratio")
                suspicion += 15

            score = max(0, 100 - min(suspicion, 100))

            return {
                'score': round(score, 2),
                'ai_probability': round(suspicion, 2),
                'red_flags': red_flags,
                'interpretation': 'Likely authentic' if score > 70 else 'Possibly AI-generated',
                'risk_level': 'Low' if score > 70 else 'Medium' if score > 40 else 'High'
            }

        except Exception as e:
            return {'score': 50, 'error': str(e), 'risk_level': 'Medium'}

    def _find_suspicious_regions_from_mask(self, mask: np.ndarray, reason: str, confidence: float):
        """Convert binary mask to regions"""
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask)

            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) < 100:
                    continue

                rows, cols = np.where(component)
                if len(rows) == 0:
                    continue

                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()

                self.suspicious_regions.append(SuspiciousRegion(
                    x=int(x_min), y=int(y_min),
                    width=int(x_max - x_min), height=int(y_max - y_min),
                    confidence=min(confidence, 100), reason=reason
                ))
        except ImportError:
            pass

    def _calculate_authenticity_score(self, metadata, ela, noise, clone, ai) -> float:
        """Weighted average of all CV-based analyses"""
        scores = {
            'metadata': metadata.get('score', 50),
            'ela': ela.get('score', 50),
            'noise': noise.get('score', 50),
            'clone': clone.get('score', 50),
            'ai': ai.get('score', 50)
        }

        weights = {
            'metadata': 0.15,
            'ela': 0.30,
            'noise': 0.20,
            'clone': 0.25,
            'ai': 0.10
        }

        total = sum(scores[k] * weights[k] for k in weights)
        return max(0, min(100, total))

    def _determine_verdict(self, score: float) -> str:
        """Determine verdict from authenticity score"""
        if score >= 70:
            return "Likely Authentic"
        elif score >= 40:
            return "Questionable - Possible Manipulation"
        else:
            return "Likely Manipulated"

    def _collect_red_flags(self) -> List[str]:
        """Collect all red flags"""
        flags = []
        reasons = set(region.reason for region in self.suspicious_regions)
        flags.extend(reasons)

        if len(self.suspicious_regions) > 5:
            flags.append(f"Multiple suspicious regions ({len(self.suspicious_regions)})")

        return list(flags)


# ============================================================================
# APPROACH 2: VISION LLM DETECTION (Semantic Analysis)
# ============================================================================

class VisionLLMDetector:
    """Vision LLM-based image authenticity analysis"""

    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name
        try:
            import ollama
            self.client = ollama
            self.available = True
        except ImportError:
            print("Warning: ollama not installed. Run: pip install ollama")
            self.available = False

    def analyze(self, image_path: str) -> Dict:
        """Analyze image using vision LLM"""
        if not self.available:
            return {
                'error': 'Ollama not available. Install: pip install ollama',
                'authenticity_score': None
            }

        prompt = """Analyze this image carefully for signs of manipulation or fakeness.

Evaluate:
1. Digital manipulation artifacts (unnatural edges, inconsistent lighting)
2. AI-generated characteristics (unrealistic details, impossible physics)
3. Photo editing traces (color mismatches, perspective issues)
4. Overall authenticity assessment

Respond ONLY with:
AUTHENTICITY: <Authentic/Questionable/Manipulated>
CONFIDENCE: <0-100>
REASONING: <brief explanation>
RED_FLAGS: <comma-separated list or "None">
"""

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )

            analysis_text = response['message']['content']
            return self._parse_response(analysis_text)

        except Exception as e:
            return {
                'error': str(e),
                'authenticity_score': None
            }

    def _parse_response(self, text: str) -> Dict:
        """Parse LLM response"""
        authenticity = "Authentic"
        confidence = 50
        reasoning = ""
        red_flags = []

        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'authenticity:' in line_lower:
                if 'manipulated' in line_lower:
                    authenticity = "Manipulated"
                elif 'questionable' in line_lower:
                    authenticity = "Questionable"
                else:
                    authenticity = "Authentic"

            elif 'confidence:' in line_lower:
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    confidence = int(match.group(1))

            elif 'reasoning:' in line_lower:
                reasoning = line.split(':', 1)[1].strip()

            elif 'red_flags:' in line_lower:
                flags_text = line.split(':', 1)[1].strip()
                if flags_text.lower() != 'none':
                    red_flags = [f.strip() for f in flags_text.split(',')]

        # Calculate score
        if authenticity == "Authentic":
            score = 100 - (100 - confidence)
        elif authenticity == "Questionable":
            score = 50
        else:
            score = confidence

        return {
            'authenticity_score': round(score, 2),
            'verdict': f"{authenticity} (Confidence: {confidence}%)",
            'confidence': confidence,
            'reasoning': reasoning,
            'red_flags': red_flags,
            'statistics': {
                'llm_authenticity': authenticity,
                'llm_confidence': confidence,
                'risk_level': 'Low' if confidence > 70 else 'Medium' if confidence > 40 else 'High'
            }
        }


# ============================================================================
# APPROACH 3: HYBRID DETECTION (Combines Both)
# ============================================================================

class HybridImageDetector:
    """Combines image analysis and vision LLM for best results"""

    def __init__(self, use_llm: bool = True, llm_model: str = "llava"):
        self.image_detector = ImageAnalysisDetector()
        self.use_llm = use_llm
        if use_llm:
            self.llm_detector = VisionLLMDetector(llm_model)

    def analyze(self, image_path: str) -> Dict:
        """Hybrid analysis combining both approaches"""
        # Always run image analysis
        image_result = self.image_detector.analyze(image_path)

        if not self.use_llm:
            return {
                'authenticity_score': image_result.get('authenticity_score'),
                'verdict': image_result.get('verdict'),
                'manipulated_probability': image_result.get('manipulated_probability'),
                'method': 'image-analysis',
                'image_analysis': image_result,
                'llm_analysis': None,
                'red_flags': image_result.get('red_flags', []),
                'suspicious_regions': image_result.get('suspicious_regions', 0)
            }

        # Run LLM analysis
        llm_result = self.llm_detector.analyze(image_path)

        if 'error' in llm_result:
            # Fallback to image analysis only
            image_result['note'] = 'LLM unavailable, using image analysis only'
            return image_result

        # Combine results (40% image, 60% LLM)
        image_score = image_result.get('authenticity_score', 50)
        llm_score = llm_result.get('authenticity_score', 50)

        combined_score = (image_score * 0.4) + (llm_score * 0.6)

        return {
            'authenticity_score': round(combined_score, 2),
            'verdict': self._determine_verdict(combined_score),
            'manipulated_probability': round(100 - combined_score, 2),
            'method': 'hybrid',
            'image_analysis': image_result,
            'llm_analysis': llm_result,
            'scores': {
                'image_analysis_score': round(image_score, 2),
                'llm_score': round(llm_score, 2),
                'combined_score': round(combined_score, 2),
                'image_weight': '40%',
                'llm_weight': '60%'
            },
            'red_flags': list(set(
                image_result.get('red_flags', []) + 
                llm_result.get('red_flags', [])
            )),
            'suspicious_regions': image_result.get('suspicious_regions', 0)
        }

    def _determine_verdict(self, score: float) -> str:
        """Determine verdict from combined score"""
        if score >= 70:
            return "Likely Authentic"
        elif score >= 40:
            return "Questionable - Possible Manipulation"
        else:
            return "Likely Manipulated"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Demo usage"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        IMAGE AUTHENTICITY DETECTOR - THREE APPROACHES            ║
    ║                                                                  ║
    ║  1. Image Analysis Detection   - Computer vision forensics      ║
    ║  2. Vision LLM Detection       - Semantic analysis              ║
    ║  3. Hybrid Detection           - Combined approach              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    print("\nUSAGE EXAMPLES:\n")

    print("1. Image Analysis Only:")
    detector = ImageAnalysisDetector()
    result = detector.analyze("D:\\PES\\projects\\fake_news_detection\\images\\fake_img_3.jpg")
    print(result)
    print("")

    print("2. Vision LLM Only:")
    detector = VisionLLMDetector(model_name='llava')
    result = detector.analyze("D:\\PES\\projects\\fake_news_detection\\images\\fake_img_3.jpg")
    print(result)
    print("")
    print("3. Hybrid (Recommended):")
    detector = HybridImageDetector(use_llm=True)
    result = detector.analyze("D:\\PES\\projects\\fake_news_detection\\images\\fake_img_3.jpg")
    print(result)
    print("   print(f\"Score: {result['authenticity_score']}%\")")
    print("   print(f\"Verdict: {result['verdict']}\")")


if __name__ == "__main__":
    main()