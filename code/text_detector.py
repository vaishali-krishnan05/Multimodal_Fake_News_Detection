"""
Text-based Fake News Detection System
Supports: Keyword-based, ML-based, and Local LLM-based detection
"""

import re
import json
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np

# ============================================================================
# APPROACH 1: KEYWORD-BASED DETECTION (Simple, Fast, No Dependencies)
# ============================================================================

class KeywordFakeNewsDetector:
    """Simple keyword-based fake news detection"""

    def __init__(self):
        # Sensationalism indicators
        self.sensational_keywords = {
            'shocking', 'unbelievable', 'amazing', 'incredible', 'you won\'t believe',
            'scientists baffled', 'doctors hate', 'secret', 'exposed', 'revealed',
            'conspiracy', 'they don\'t want you to know', 'breaking', 'bombshell',
            'miracle', 'guaranteed', 'instant', 'revolutionary'
        }

        # Emotional manipulation words
        self.emotional_keywords = {
            'outrage', 'furious', 'shocking', 'terrifying', 'devastating',
            'horrifying', 'disgusting', 'pathetic', 'ridiculous', 'insane',
            'crazy', 'unbelievable', 'catastrophic', 'apocalyptic'
        }

        # Absolutist language (lacks nuance)
        self.absolutist_keywords = {
            'always', 'never', 'everyone', 'nobody', 'all', 'none',
            'completely', 'totally', 'absolutely', 'definitely', 'certainly',
            'undoubtedly', 'obviously', 'clearly'
        }

        # Credibility markers (positive indicators)
        self.credibility_keywords = {
            'according to', 'research shows', 'study found', 'data indicates',
            'expert', 'professor', 'dr.', 'university', 'peer-reviewed',
            'published', 'journal', 'analysis', 'evidence'
        }

        # Vague attribution (red flag)
        self.vague_attribution = {
            'some say', 'many people', 'sources say', 'insiders claim',
            'rumors suggest', 'allegedly', 'supposedly', 'it is believed',
            'people are saying'
        }

        # Clickbait patterns
        self.clickbait_patterns = [
            r'\d+\s+(?:reasons|ways|things|tricks|secrets)',  # "10 reasons why..."
            r'you won\'t believe',
            r'what happened next',
            r'number \d+ will (?:shock|amaze)',
            r'this one (?:weird|simple) trick',
        ]

    def analyze(self, text: str) -> Dict:
        """Analyze text and return fake news probability"""
        text_lower = text.lower()

        # Count keyword occurrences
        sensational_count = sum(1 for word in self.sensational_keywords if word in text_lower)
        emotional_count = sum(1 for word in self.emotional_keywords if word in text_lower)
        absolutist_count = sum(1 for word in self.absolutist_keywords if word in text_lower)
        credibility_count = sum(1 for word in self.credibility_keywords if word in text_lower)
        vague_count = sum(1 for word in self.vague_attribution if word in text_lower)

        # Check clickbait patterns
        clickbait_count = sum(1 for pattern in self.clickbait_patterns
                              if re.search(pattern, text_lower))

        # Calculate scores
        word_count = len(text.split())

        # Normalize by word count (per 100 words)
        norm_factor = max(word_count / 100, 1)

        red_flag_score = (
                                 (sensational_count * 2.0) +
                                 (emotional_count * 1.5) +
                                 (absolutist_count * 1.0) +
                                 (vague_count * 2.5) +
                                 (clickbait_count * 3.0)
                         ) / norm_factor

        credibility_score = credibility_count / norm_factor

        # Calculate fake probability (0-100)
        fake_probability = min(100, max(0,
                                        (red_flag_score * 10) - (credibility_score * 5)
                                        ))

        # Determine verdict
        if fake_probability < 30:
            verdict = "Likely Credible"
        elif fake_probability < 60:
            verdict = "Questionable"
        else:
            verdict = "Likely Fake/Misleading"

        return {
            'fake_probability': round(fake_probability, 2),
            'verdict': verdict,
            'analysis': {
                'sensational_keywords': sensational_count,
                'emotional_keywords': emotional_count,
                'absolutist_keywords': absolutist_count,
                'credibility_markers': credibility_count,
                'vague_attribution': vague_count,
                'clickbait_patterns': clickbait_count,
                'word_count': word_count
            },
            'red_flags': self._get_red_flags(
                sensational_count, emotional_count, absolutist_count,
                vague_count, clickbait_count, credibility_count
            )
        }

    def _get_red_flags(self, sens, emot, absol, vague, click, cred) -> List[str]:
        """Generate list of detected red flags"""
        flags = []
        if sens > 2:
            flags.append(f"High sensationalism ({sens} keywords)")
        if emot > 3:
            flags.append(f"Emotional manipulation detected ({emot} keywords)")
        if absol > 5:
            flags.append(f"Absolutist language ({absol} keywords)")
        if vague > 1:
            flags.append(f"Vague attribution ({vague} instances)")
        if click > 0:
            flags.append(f"Clickbait patterns detected ({click})")
        if cred == 0:
            flags.append("No credibility markers found")
        return flags


# ============================================================================
# APPROACH 2: LOCAL LLM-BASED DETECTION (Using Ollama)
# ============================================================================

class LocalLLMDetector:
    """
    Use a local LLM (via Ollama) for sophisticated analysis

    Setup:
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull llama2 (or mistral, phi, etc.)
    3. pip install ollama
    """

    def __init__(self, model_name: str = "llama3:latest"):
        self.model_name = model_name
        try:
            import ollama
            self.client = ollama
            self.available = True
        except ImportError:
            print("Warning: ollama not installed. Run: pip install ollama")
            self.available = False

    def analyze(self, text: str) -> Dict:
        """Analyze text using local LLM"""
        if not self.available:
            return {
                'error': 'Ollama not available. Install with: pip install ollama',
                'fake_probability': None
            }

        prompt = f"""Analyze this article for signs of fake news or misinformation.

Article:
{text[:2000]}  # Limit to avoid token limits

Please evaluate:
1. Sensationalism and emotional manipulation
2. Source credibility and attribution
3. Use of absolutist or polarizing language
4. Logical consistency
5. Clickbait indicators

Respond in JSON format:
{{
    "fake_probability": <0-100>,
    "verdict": "<Credible/Questionable/Likely Fake>",
    "reasoning": "<brief explanation>",
    "red_flags": ["<flag1>", "<flag2>"]
}}"""

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                format='json'  # Request JSON output
            )

            # Parse response
            result = json.loads(response['response'])
            return result

        except Exception as e:
            return {
                'error': str(e),
                'fake_probability': None
            }


# ============================================================================
# APPROACH 3: HYBRID DETECTOR (Combines keyword + LLM)
# ============================================================================

class HybridDetector:
    """Combines keyword analysis with LLM for best results"""

    def __init__(self, use_llm: bool = True, model_name: str = "llama3:latest"):
        self.keyword_detector = KeywordFakeNewsDetector()
        self.use_llm = use_llm
        if use_llm:
            self.llm_detector = LocalLLMDetector(model_name)

    def analyze(self, text: str) -> Dict:
        """Run hybrid analysis"""
        # Always run keyword analysis (fast)
        keyword_result = self.keyword_detector.analyze(text)

        if not self.use_llm:
            return keyword_result

        # Run LLM analysis if available
        llm_result = self.llm_detector.analyze(text)

        if 'error' in llm_result:
            # Fall back to keyword-only
            keyword_result['note'] = 'LLM analysis unavailable, using keyword-based only'
            return keyword_result

        # Combine results (weighted average)
        combined_probability = (
            #llm can return malformed answers. If we dont get fake probability tag in llm-json response, assume a neutral score of 50
                keyword_result['fake_probability'] * 0.4 +
                llm_result.get('fake_probability', 50) * 0.6
        )

        return {
            'fake_probability': round(combined_probability, 2),
            'verdict': self._determine_verdict(combined_probability),
            'keyword_analysis': keyword_result,
            'llm_analysis': llm_result,
            'method': 'hybrid'
        }

    def _determine_verdict(self, prob: float) -> str:
        if prob < 30:
            return "Likely Credible"
        elif prob < 60:
            return "Questionable"
        else:
            return "Likely Fake/Misleading"


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    # Example fake news article
    fake_article = """
    SHOCKING: You Won't Believe What Scientists Just Discovered!

    Sources say that this one weird trick will absolutely change your life forever.
    Experts are FURIOUS and they don't want you to know this secret!

    Everyone is talking about this revolutionary breakthrough that will completely
    transform everything we know. This is unbelievable and totally insane!
    Nobody can deny this shocking truth anymore.
    """

    # Example credible article
    credible_article = """
    New Study Shows Moderate Exercise Benefits Cognitive Function

    According to research published in the Journal of Neuroscience, regular 
    moderate exercise may improve cognitive function in older adults. The study,
    conducted by researchers at Stanford University, analyzed data from 500 
    participants over 2 years.

    Dr. Jane Smith, lead researcher, noted that "the data indicates a correlation
    between consistent physical activity and improved memory performance, though
    further research is needed to establish causation."
    """

    print("="*70)
    print("APPROACH 1: Keyword-Based Detection")
    print("="*70)

    detector = KeywordFakeNewsDetector()

    print("\n--- Analyzing Fake Article ---")
    result = detector.analyze(fake_article)
    print(f"Fake Probability: {result['fake_probability']}%")
    print(f"Verdict: {result['verdict']}")
    print(f"Red Flags: {result['red_flags']}")
    print(f"Analysis: {json.dumps(result['analysis'], indent=2)}")

    print("\n--- Analyzing Credible Article ---")
    result = detector.analyze(credible_article)
    print(f"Fake Probability: {result['fake_probability']}%")
    print(f"Verdict: {result['verdict']}")
    print(f"Red Flags: {result['red_flags']}")

    print("\n" + "="*70)
    print("APPROACH 2: Local LLM Detection (requires Ollama)")
    print("="*70)

    llm_detector = LocalLLMDetector(model_name="llama3:latest")
    if llm_detector.available:
        print("\n--- Analyzing with LLM ---")
        result = llm_detector.analyze(fake_article)
        print(json.dumps(result, indent=2))
        result = llm_detector.analyze(fake_article)
        print(json.dumps(result, indent=2))
    else:
        print("Ollama not available. Install: pip install ollama")
        print("Then run: ollama pull llama3:latest")

    print("\n" + "="*70)
    print("APPROACH 3: Hybrid Detection")
    print("="*70)

    hybrid = HybridDetector(use_llm=True)  # Set to True if Ollama is installed
    result = hybrid.analyze(fake_article)
    print(json.dumps(result, indent=2))
    result = hybrid.analyze(credible_article)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()