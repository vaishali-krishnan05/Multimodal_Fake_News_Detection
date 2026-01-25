"""
Image Authenticity Detection for News Articles
Detects manipulation, highlights suspicious areas, and estimates authenticity

Features:
1. Error Level Analysis (ELA) - Detects JPEG compression inconsistencies
2. Metadata Analysis - Checks EXIF data for editing signs
3. Clone Detection - Finds copy-pasted regions
4. AI-Generated Image Detection - Detects GAN/Diffusion artifacts
5. Noise Pattern Analysis - Finds inconsistent noise
6. Visual Highlighting - Shows suspicious areas
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from PIL.ExifTags import TAGS
import io
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
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


class ImageAuthenticityDetector:
    """Main detector class for image authenticity analysis"""

    def __init__(self, use_llm: bool = True, llm_model: str = "llava"):
        self.suspicious_regions = []
        self.analysis_results = {}
        self.use_llm = use_llm
        self.llm_model = llm_model

        if use_llm:
            try:
                import ollama
                self.ollama_client = ollama
                self.llm_available = True
            except ImportError:
                print("Warning: ollama not installed. Run: pip install ollama")
                self.llm_available = False
        else:
            self.llm_available = False

    def analyze(self, image_path: str, save_annotated: bool = True) -> Dict:
        """
        Complete analysis of an image

        Args:
            image_path: Path to image file
            save_annotated: Whether to save annotated image

        Returns:
            Dictionary with analysis results
        """
        try:
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            print(f"Analyzing image: {image_path}")
            print(f"Size: {img.size}, Format: {img.format}")

            # Run all analyses
            metadata_result = self._analyze_metadata(image_path)
            ela_result = self._error_level_analysis(img)
            noise_result = self._analyze_noise_patterns(img)
            clone_result = self._detect_cloning(img)
            ai_result = self._detect_ai_generation(img)

            # LLM analysis (if enabled)
            llm_result = None
            if self.use_llm and self.llm_available:
                llm_result = self._llm_analysis(image_path)

            # Calculate overall authenticity score
            authenticity_score = self._calculate_authenticity_score(
                metadata_result,
                ela_result,
                noise_result,
                clone_result,
                ai_result,
                llm_result
            )

            # Generate verdict
            verdict = self._determine_verdict(authenticity_score)

            # Create annotated image
            annotated_img = None
            if save_annotated:
                annotated_img = self._create_annotated_image(img, image_path)

            return {
                'authenticity_score': round(authenticity_score, 2),
                'verdict': verdict,
                'manipulated_probability': round(100 - authenticity_score, 2),
                'analyses': {
                    'metadata': metadata_result,
                    'error_level_analysis': ela_result,
                    'noise_patterns': noise_result,
                    'clone_detection': clone_result,
                    'ai_detection': ai_result,
                    'llm_analysis': llm_result
                },
                'suspicious_regions': len(self.suspicious_regions),
                'red_flags': self._collect_red_flags(),
                'annotated_image_path': annotated_img
            }

        except Exception as e:
            return {
                'error': str(e),
                'authenticity_score': None
            }

    def _analyze_metadata(self, image_path: str) -> Dict:
        """Analyze EXIF metadata for manipulation signs"""
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()

            if not exif_data:
                return {
                    'has_metadata': False,
                    'red_flags': ['No EXIF data (possibly stripped)'],
                    'suspicion_score': 40
                }

            # Extract relevant EXIF tags
            metadata = {}
            red_flags = []

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                metadata[tag_name] = str(value)

            # Check for editing software
            software = metadata.get('Software', '').lower()
            if any(editor in software for editor in ['photoshop', 'gimp', 'paint', 'edited']):
                red_flags.append(f"Edited with: {metadata.get('Software')}")

            # Check for inconsistent timestamps
            date_original = metadata.get('DateTimeOriginal')
            date_digitized = metadata.get('DateTimeDigitized')
            if date_original and date_digitized and date_original != date_digitized:
                red_flags.append("Inconsistent timestamps detected")

            # Check if metadata seems fabricated
            if 'Make' not in metadata and 'Model' not in metadata:
                red_flags.append("Camera information missing")

            suspicion_score = min(len(red_flags) * 20, 80)

            return {
                'has_metadata': True,
                'software': metadata.get('Software', 'Unknown'),
                'camera': f"{metadata.get('Make', '')} {metadata.get('Model', '')}".strip(),
                'date_taken': metadata.get('DateTimeOriginal', 'Unknown'),
                'red_flags': red_flags,
                'suspicion_score': suspicion_score
            }

        except Exception as e:
            return {
                'has_metadata': False,
                'error': str(e),
                'suspicion_score': 30
            }

    def _error_level_analysis(self, img: Image.Image) -> Dict:
        """
        Error Level Analysis (ELA) - Detects JPEG compression inconsistencies
        Areas with different compression levels indicate manipulation
        """
        try:
            # Save image with known quality
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, format='JPEG', quality=90)
            temp_buffer.seek(0)

            # Reload the compressed image
            compressed = Image.open(temp_buffer)

            # Calculate difference between original and recompressed
            original_array = np.array(img, dtype=np.float32)
            compressed_array = np.array(compressed, dtype=np.float32)

            # Compute absolute difference
            diff = np.abs(original_array - compressed_array)

            # Normalize to 0-255 range
            diff_normalized = (diff / diff.max() * 255).astype(np.uint8)

            # Find high-error regions (potential manipulation)
            threshold = np.percentile(diff_normalized, 95)  # Top 5% of errors
            high_error_mask = diff_normalized > threshold

            # Count suspicious pixels
            suspicious_pixel_ratio = np.sum(high_error_mask) / high_error_mask.size

            # Find bounding boxes of suspicious regions
            self._find_suspicious_regions_from_mask(
                high_error_mask[:, :, 0] if len(high_error_mask.shape) == 3 else high_error_mask,
                "ELA: Compression inconsistency",
                min_confidence=60
            )

            suspicion_score = min(suspicious_pixel_ratio * 500, 100)

            return {
                'suspicious_pixel_ratio': round(suspicious_pixel_ratio, 4),
                'high_error_regions': len([r for r in self.suspicious_regions if 'ELA' in r.reason]),
                'suspicion_score': round(suspicion_score, 2),
                'interpretation': self._interpret_ela_score(suspicion_score)
            }

        except Exception as e:
            return {
                'error': str(e),
                'suspicion_score': 0
            }

    def _analyze_noise_patterns(self, img: Image.Image) -> Dict:
        """
        Analyze noise patterns - manipulated areas have different noise
        """
        try:
            # Convert to grayscale for noise analysis
            gray = img.convert('L')
            gray_array = np.array(gray, dtype=np.float32)

            # Apply Gaussian blur and subtract to get noise
            blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
            blurred_array = np.array(blurred, dtype=np.float32)
            noise = gray_array - blurred_array

            # Divide image into blocks and analyze noise variance
            block_size = 64
            height, width = noise.shape
            noise_variances = []

            for y in range(0, height - block_size, block_size):
                for x in range(0, width - block_size, block_size):
                    block = noise[y:y + block_size, x:x + block_size]
                    variance = np.var(block)
                    noise_variances.append((x, y, variance))

            if not noise_variances:
                return {'suspicion_score': 0, 'inconsistent_regions': 0}

            # Find blocks with unusual variance (too high or too low)
            variances = [v[2] for v in noise_variances]
            mean_var = np.mean(variances)
            std_var = np.std(variances)

            inconsistent_count = 0
            for x, y, var in noise_variances:
                # Flag if variance is more than 2 std devs from mean
                if abs(var - mean_var) > 2 * std_var:
                    self.suspicious_regions.append(SuspiciousRegion(
                        x=x, y=y, width=block_size, height=block_size,
                        confidence=min(abs(var - mean_var) / std_var * 30, 100),
                        reason="Noise: Inconsistent noise pattern",
                        color="yellow"
                    ))
                    inconsistent_count += 1

            suspicion_score = min(inconsistent_count * 5, 70)

            return {
                'inconsistent_regions': inconsistent_count,
                'mean_noise_variance': round(mean_var, 2),
                'suspicion_score': round(suspicion_score, 2),
                'interpretation': 'Natural' if suspicion_score < 30 else 'Suspicious'
            }

        except Exception as e:
            return {
                'error': str(e),
                'suspicion_score': 0
            }

    def _detect_cloning(self, img: Image.Image) -> Dict:
        """
        Detect copy-paste cloning (simplified version)
        Uses block-based matching to find duplicated regions
        """
        try:
            # Resize for faster processing
            max_size = 800
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img_small = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_small = img

            gray = img_small.convert('L')
            gray_array = np.array(gray)

            # Divide into blocks and compute hashes
            block_size = 32
            height, width = gray_array.shape
            block_hashes = {}

            for y in range(0, height - block_size, block_size // 2):  # 50% overlap
                for x in range(0, width - block_size, block_size // 2):
                    block = gray_array[y:y + block_size, x:x + block_size]

                    # Simple hash based on block statistics
                    block_hash = hashlib.md5(block.tobytes()).hexdigest()

                    if block_hash in block_hashes:
                        # Found duplicate block!
                        orig_x, orig_y = block_hashes[block_hash]

                        # Make sure it's not just overlapping blocks
                        distance = np.sqrt((x - orig_x) ** 2 + (y - orig_y) ** 2)
                        if distance > block_size:
                            # Scale back to original image coordinates
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
                                confidence=75,
                                reason="Clone: Duplicated region detected",
                                color="orange"
                            ))
                    else:
                        block_hashes[block_hash] = (x, y)

            clone_count = len([r for r in self.suspicious_regions if 'Clone' in r.reason])
            suspicion_score = min(clone_count * 25, 100)

            return {
                'cloned_regions_found': clone_count,
                'suspicion_score': round(suspicion_score, 2),
                'interpretation': 'No cloning detected' if clone_count == 0 else f'{clone_count} potential clone(s)'
            }

        except Exception as e:
            return {
                'error': str(e),
                'suspicion_score': 0
            }

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
        """
        Detect AI-generated images (simplified heuristics)
        Real detection would use specialized neural networks
        """
        try:
            # Convert to numpy array
            img_array = np.array(img)

            red_flags = []
            suspicion_score = 0

            # Check 1: Unusual color distribution (AI images often have characteristic palettes)
            color_std = np.std(img_array, axis=(0, 1))
            if np.all(color_std > 50) and np.all(color_std < 70):
                red_flags.append("Uniform color distribution (AI characteristic)")
                suspicion_score += 20

            # Check 2: Perfect symmetry (some AI generators create too-perfect symmetry)
            left_half = img_array[:, :img_array.shape[1] // 2]
            right_half = np.fliplr(img_array[:, img_array.shape[1] // 2:])

            # Resize if different sizes
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]

            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            if symmetry_diff < 10:
                red_flags.append("Unusual symmetry detected")
                suspicion_score += 30

            # Check 3: Smooth gradients (AI often creates unnaturally smooth transitions)
            edges = img.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges.convert('L'))
            edge_density = np.mean(edges_array > 20)

            if edge_density < 0.1:  # Very few edges = too smooth
                red_flags.append("Unnaturally smooth gradients")
                suspicion_score += 25

            # Check 4: Aspect ratio (AI images often use standard ratios)
            width, height = img.size
            ratio = width / height
            common_ai_ratios = [1.0, 1.5, 0.75, 1.333, 0.667]  # 1:1, 3:2, 2:3, 4:3, 3:4
            if any(abs(ratio - r) < 0.01 for r in common_ai_ratios):
                red_flags.append("Standard AI generation aspect ratio")
                suspicion_score += 15

            return {
                'ai_probability': round(min(suspicion_score, 100), 2),
                'red_flags': red_flags,
                'suspicion_score': round(min(suspicion_score, 100), 2),
                'note': 'Advanced AI detection requires specialized neural networks'
            }

        except Exception as e:
            return {
                'error': str(e),
                'suspicion_score': 0
            }

    def _find_suspicious_regions_from_mask(self, mask: np.ndarray, reason: str, min_confidence: float = 50):
        """Convert binary mask to bounding boxes"""
        from scipy import ndimage

        try:
            # Label connected components
            labeled, num_features = ndimage.label(mask)

            # Find bounding boxes for each component
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) < 100:  # Skip tiny regions
                    continue

                rows, cols = np.where(component)
                if len(rows) == 0:
                    continue

                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()

                # Calculate confidence based on region size
                area = (y_max - y_min) * (x_max - x_min)
                confidence = min(min_confidence + (area / 1000), 100)

                self.suspicious_regions.append(SuspiciousRegion(
                    x=int(x_min), y=int(y_min),
                    width=int(x_max - x_min), height=int(y_max - y_min),
                    confidence=confidence,
                    reason=reason
                ))
        except ImportError:
            # scipy not available, use simple thresholding
            pass

    def _calculate_authenticity_score(self, metadata, ela, noise, clone, ai, llm=None) -> float:
        """
        Calculate overall authenticity score (0-100)
        100 = definitely authentic, 0 = definitely fake
        """
        if llm and llm.get('suspicion_score') is not None and 'error' not in llm:
            # If LLM is available, give it more weight
            weights = {
                'metadata': 0.10,
                'ela': 0.20,
                'noise': 0.15,
                'clone': 0.15,
                'ai': 0.05,
                'llm': 0.35  # LLM gets highest weight
            }

            scores = {
                'metadata': 100 - metadata.get('suspicion_score', 0),
                'ela': 100 - ela.get('suspicion_score', 0),
                'noise': 100 - noise.get('suspicion_score', 0),
                'clone': 100 - clone.get('suspicion_score', 0),
                'ai': 100 - ai.get('suspicion_score', 0),
                'llm': 100 - llm.get('suspicion_score', 0)
            }
        else:
            # Original weights without LLM
            weights = {
                'metadata': 0.15,
                'ela': 0.30,
                'noise': 0.20,
                'clone': 0.25,
                'ai': 0.10
            }

            scores = {
                'metadata': 100 - metadata.get('suspicion_score', 0),
                'ela': 100 - ela.get('suspicion_score', 0),
                'noise': 100 - noise.get('suspicion_score', 0),
                'clone': 100 - clone.get('suspicion_score', 0),
                'ai': 100 - ai.get('suspicion_score', 0)
            }

        # Calculate weighted average
        total_score = sum(scores[k] * weights[k] for k in weights)

        return max(0, min(100, total_score))

    def _determine_verdict(self, authenticity_score: float) -> str:
        """Determine verdict based on authenticity score"""
        if authenticity_score >= 70:
            return "Likely Authentic"
        elif authenticity_score >= 40:
            return "Questionable - Possible Manipulation"
        else:
            return "Likely Manipulated"

    def _collect_red_flags(self) -> List[str]:
        """Collect all red flags from analyses"""
        flags = []

        # Get unique reasons from suspicious regions
        reasons = set(region.reason for region in self.suspicious_regions)
        flags.extend(reasons)

        if len(self.suspicious_regions) > 5:
            flags.append(f"Multiple suspicious regions detected ({len(self.suspicious_regions)})")

        return flags

    def _interpret_ela_score(self, score: float) -> str:
        """Interpret ELA suspicion score"""
        if score < 20:
            return "Consistent compression - likely authentic"
        elif score < 50:
            return "Some compression variations - minor editing possible"
        elif score < 75:
            return "Significant compression inconsistencies - likely edited"
        else:
            return "Severe compression anomalies - heavily manipulated"

    def _create_annotated_image(self, img: Image.Image, original_path: str) -> str:
        """Create annotated image with highlighted suspicious regions"""
        # Create a copy to annotate
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated, 'RGBA')

        # Draw all suspicious regions
        for i, region in enumerate(self.suspicious_regions):
            # Color based on confidence
            if region.confidence > 75:
                color = (255, 0, 0, 100)  # Red - high confidence
                outline = (255, 0, 0, 255)
            elif region.confidence > 50:
                color = (255, 165, 0, 80)  # Orange - medium
                outline = (255, 165, 0, 255)
            else:
                color = (255, 255, 0, 60)  # Yellow - low confidence
                outline = (255, 255, 0, 255)

            # Draw rectangle
            draw.rectangle(
                [region.x, region.y, region.x + region.width, region.y + region.height],
                fill=color,
                outline=outline,
                width=3
            )

            # Add label
            label = f"{i + 1}: {region.confidence:.0f}%"
            draw.text((region.x + 5, region.y + 5), label, fill=(255, 255, 255))

        # Save annotated image
        output_path = original_path.rsplit('.', 1)[0] + '_annotated.jpg'
        annotated.save(output_path, quality=95)
        print(f"Annotated image saved: {output_path}")

        return output_path


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def analyze_image(image_path: str, use_llm: bool = True):
    """Analyze a single image and display results"""
    print("=" * 70)
    print("IMAGE AUTHENTICITY ANALYSIS")
    if use_llm:
        print("(Using Vision LLM for enhanced analysis)")
    print("=" * 70)

    detector = ImageAuthenticityDetector(use_llm=use_llm)
    result = detector.analyze(image_path, save_annotated=True)

    if 'error' in result:
        print(f"\nError: {result['error']}")
        return

    print(f"\n{'VERDICT':-^70}")
    print(f"Authenticity Score: {result['authenticity_score']}/100")
    print(f"Manipulation Probability: {result['manipulated_probability']}%")
    print(f"Verdict: {result['verdict']}")

    print(f"\n{'ANALYSIS BREAKDOWN':-^70}")

    # Metadata
    meta = result['analyses']['metadata']
    print(f"\n1. METADATA ANALYSIS (Suspicion: {meta.get('suspicion_score', 0)}%)")
    if meta.get('has_metadata'):
        print(f"   Camera: {meta.get('camera', 'Unknown')}")
        print(f"   Software: {meta.get('software', 'Unknown')}")
        print(f"   Date: {meta.get('date_taken', 'Unknown')}")
    if meta.get('red_flags'):
        print(f"   ⚠ Red Flags: {', '.join(meta['red_flags'])}")

    # ELA
    ela = result['analyses']['error_level_analysis']
    print(f"\n2. ERROR LEVEL ANALYSIS (Suspicion: {ela.get('suspicion_score', 0)}%)")
    print(f"   {ela.get('interpretation', 'N/A')}")
    print(f"   High-error regions: {ela.get('high_error_regions', 0)}")

    # Noise
    noise = result['analyses']['noise_patterns']
    print(f"\n3. NOISE PATTERN ANALYSIS (Suspicion: {noise.get('suspicion_score', 0)}%)")
    print(f"   {noise.get('interpretation', 'N/A')}")
    print(f"   Inconsistent regions: {noise.get('inconsistent_regions', 0)}")

    # Clone
    clone = result['analyses']['clone_detection']
    print(f"\n4. CLONE DETECTION (Suspicion: {clone.get('suspicion_score', 0)}%)")
    print(f"   {clone.get('interpretation', 'N/A')}")

    # AI
    ai = result['analyses']['ai_detection']
    print(f"\n5. AI GENERATION DETECTION (Suspicion: {ai.get('suspicion_score', 0)}%)")
    print(f"   AI Probability: {ai.get('ai_probability', 0)}%")
    if ai.get('red_flags'):
        for flag in ai['red_flags']:
            print(f"   ⚠ {flag}")

    # LLM (if available)
    if result['analyses'].get('llm_analysis'):
        llm = result['analyses']['llm_analysis']
        if 'error' not in llm:
            print(f"\n6. VISION LLM ANALYSIS (Suspicion: {llm.get('suspicion_score', 0)}%)")
            print(f"   Verdict: {llm.get('authenticity', 'Unknown')}")
            print(f"   Confidence: {llm.get('confidence', 0)}%")
            print(f"   Reasoning: {llm.get('reasoning', 'N/A')}")
            if llm.get('red_flags'):
                print(f"   LLM Red Flags:")
                for flag in llm['red_flags']:
                    print(f"   ⚠ {flag}")

    print(f"\n{'SUSPICIOUS REGIONS':-^70}")
    print(f"Total regions flagged: {result['suspicious_regions']}")

    if result.get('red_flags'):
        print(f"\n{'RED FLAGS SUMMARY':-^70}")
        for i, flag in enumerate(result['red_flags'], 1):
            print(f"{i}. {flag}")

    if result.get('annotated_image_path'):
        print(f"\n{'OUTPUT':-^70}")
        print(f"Annotated image saved: {result['annotated_image_path']}")
        print("Check the image to see highlighted suspicious regions!")

    print("=" * 70)


def main():
    """Demo with sample usage"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          IMAGE AUTHENTICITY DETECTOR FOR NEWS ARTICLES          ║
    ║                                                                  ║
    ║  Detects:                                                        ║
    ║  • JPEG compression manipulation (ELA)                           ║
    ║  • Cloned/copy-pasted regions                                   ║
    ║  • Noise pattern inconsistencies                                ║
    ║  • AI-generated images                                          ║
    ║  • Metadata tampering                                           ║
    ║                                                                  ║
    ║  Output: Annotated image with highlighted suspicious areas      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Example usage
    print("\nUSAGE:")
    print("------")
    print("from image_detector import ImageAuthenticityDetector")
    print("")
    print("detector = ImageAuthenticityDetector()")
    print("result = detector.analyze('news_image.jpg')")
    print("")
    print("# Access results")
    print("print(f\"Authenticity: {result['authenticity_score']}%\")")
    print("print(f\"Verdict: {result['verdict']}\")")
    print("")
    print("# Or use the helper function:")
    print("analyze_image('news_image.jpg')")
    print("\n" + "=" * 70)

    # Uncomment to test with an actual image:
    #analyze_image("D:\\PES\\projects\\fake_img.jpg")
    #analyze_image("D:\\PES\\projects\\real_img.jpg")
    analyze_image("D:\\PES\\projects\\fake_img_2.png")


if __name__ == "__main__":
    main()