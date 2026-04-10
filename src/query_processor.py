"""
Query processing module for the professor matching system.

This module provides:
1. QueryValidator - Validates queries and filters irrelevant ones
2. KeywordExtractor - Extracts and weights keywords from queries
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


class QueryStatus(Enum):
    """Status of query validation."""
    VALID = "valid"
    INVALID = "invalid"
    WEAK_RELEVANCE = "weak_relevance"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass
class QueryValidationResult:
    """Result of query validation."""
    status: QueryStatus
    message: str
    suggestions: List[str]
    confidence: float


@dataclass
class ExtractedKeywords:
    """Extracted keywords with weights."""
    keywords: List[Tuple[str, float]]  # (keyword, weight)
    filtered_query: str  # Query reconstructed from important keywords
    original_query: str


class KeywordExtractor:
    """
    Context-aware keyword extractor using spaCy NLP.
    
    Key design principles:
    1. Only filter true stopwords (function words like for, and, the)
    2. Use context-sensitive weighting - a word's importance depends on 
       what other words are in the query
    3. Domain specificity affects relative importance, not absolute filtering
    """
    
    # Pure function words to always filter (language mechanics, not content)
    STOPWORDS = {
        # Articles and determiners
        'a', 'an', 'the', 'this', 'that', 'these', 'those',
        # Prepositions
        'for', 'with', 'from', 'via', 'through', 'towards', 'toward', 'towards',
        'into', 'onto', 'upon', 'over', 'under', 'between', 'among', 'about',
        'after', 'before', 'during', 'without', 'within', 'behind', 'beyond',
        # Conjunctions
        'and', 'or', 'but', 'nor', 'yet', 'so', 'both', 'either', 'neither',
        # Auxiliary/linking
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'must', 'shall',
        # Pronouns
        'it', 'its', 'they', 'their', 'them', 'we', 'our', 'us', 'you', 'your',
        'he', 'she', 'his', 'her', 'i', 'my', 'me',
        # Other function words
        'to', 'of', 'by', 'as', 'at', 'on', 'in', 'if', 'then', 'than', 'when',
        'what', 'which', 'who', 'whom', 'how', 'why', 'where', 'there', 'here',
        'some', 'any', 'all', 'each', 'every', 'no', 'not', 'only', 'also',
        'very', 'just', 'more', 'most', 'less', 'least', 'other', 'another',
        'such', 'same', 'different', 'like', 'unlike',
    }
    
    # High-specificity domain terms (extracted from publications/projects)
    # These indicate strong domain focus when present
    HIGH_SPECIFICITY_TERMS = {
        # AI/ML specific
        'neural', 'cnn', 'rnn', 'lstm', 'gpt', 'llm', 'transformer', 'bert',
        'diffusion', 'generative', 'adversarial', 'gan', 'vae', 'autoencoder',
        'federated', 'reinforcement', 'backpropagation', 'gradient', 'softmax',
        'attention', 'embedding', 'tokenization', 'fine-tuning', 'pretrained',
        'contrastive', 'self-supervised', 'semi-supervised', 'unsupervised',
        # Hardware specific
        'fpga', 'asic', 'vlsi', 'cmos', 'sram', 'dram', 'nand', 'nor',
        'photonic', 'optoelectronic', 'plasmonic', 'metamaterial',
        'perovskite', 'graphene', 'nanotube', 'nanowire', 'quantum-dot',
        'ferroelectric', 'piezoelectric', 'thermoelectric', 'magnetoresistive',
        'microfluidic', 'microelectromechanical', 'nanoelectromechanical',
        # Communication specific  
        'mimo', 'ofdm', 'cdma', 'tdma', 'fdma', 'beamforming', 'precoding',
        '5g', '6g', 'lte', 'wifi', 'bluetooth', 'zigbee', 'lorawan',
        'lidar', 'radar', 'sonar', 'ultrasonic', 'terahertz', 'millimeter-wave',
        # Computing specific
        'cuda', 'opencl', 'tensorrt', 'onnx', 'kubernetes', 'docker',
        'hpc', 'mpi', 'openmp', 'simd', 'risc-v', 'arm', 'x86',
        # Domain specific - expanded
        'crispr', 'proteomics', 'genomics', 'metabolomics', 'transcriptomics',
        'electrochemical', 'spectroscopy', 'chromatography', 'rheology',
        'geotechnical', 'geophysical', 'seismology', 'hydrology', 'oceanography',
        'aerodynamics', 'thermodynamics', 'kinematics', 'tribology',
        'biomechanics', 'bioinformatics', 'microbiome', 'immunotherapy',
        # Environment specific
        'underwater', 'subsea', 'offshore', 'subsurface', 'subterranean',
        'atmospheric', 'tropospheric', 'stratospheric', 'ionospheric',
        # Specific methods/algorithms
        'slam', 'kalman', 'bayesian', 'markov', 'monte-carlo', 'ransac',
        'convex', 'stochastic', 'heuristic', 'metaheuristic', 'evolutionary',
        'pid', 'mpc', 'lqr', 'svm', 'knn', 'pca', 'svd', 'fft', 'dft',
        # Specific applications
        'exoskeleton', 'prosthetics', 'haptic', 'holographic', 'stereoscopic',
    }
    
    # Words to filter out even as individual tokens
    FILTER_WORDS = {
        'self', 'based', 'using', 'used', 'use', 'uses', 'novel', 'new',
        'approach', 'method', 'technique', 'way', 'manner', 'type', 'kind',
        'study', 'research', 'work', 'paper', 'project', 'development',
        'improved', 'enhanced', 'advanced', 'efficient', 'effective',
        'high', 'low', 'fast', 'slow', 'large', 'small', 'big', 'tiny',
        'good', 'better', 'best', 'bad', 'worse', 'worst', 'various', 'multiple',
    }
    
    # Medium-specificity domain terms (common in engineering but meaningful)
    MEDIUM_SPECIFICITY_TERMS = {
        # AI/ML
        'learning', 'neural', 'deep', 'machine', 'ai', 'artificial', 'intelligence',
        'classification', 'regression', 'clustering', 'prediction', 'inference',
        'training', 'validation', 'testing', 'overfitting', 'regularization',
        # Common technical
        'network', 'networks', 'algorithm', 'optimization', 'simulation',
        'detection', 'recognition', 'segmentation', 'tracking', 'estimation',
        'control', 'feedback', 'adaptive', 'robust', 'real-time',
        # Hardware
        'circuit', 'circuits', 'chip', 'processor', 'accelerator', 'gpu', 'cpu',
        'memory', 'cache', 'bandwidth', 'latency', 'throughput',
        'sensor', 'sensors', 'actuator', 'transducer', 'mems',
        # Communication
        'wireless', 'antenna', 'signal', 'channel', 'modulation', 'encoding',
        'protocol', 'packet', 'routing', 'congestion', 'qos',
        # Computing
        'distributed', 'parallel', 'cloud', 'edge', 'fog', 'serverless',
        'computing', 'computer', 'quantum', 'cryptography', 'encryption',
        # Robotics
        'robot', 'robotic', 'robotics', 'autonomous', 'drone', 'uav', 'ugv',
        'manipulation', 'navigation', 'localization', 'perception', 'planning',
        # Energy/Materials
        'energy', 'power', 'battery', 'solar', 'thermal', 'renewable',
        'material', 'materials', 'nano', 'polymer', 'composite', 'alloy',
        'carbon', 'graphene', 'ceramic', 'semiconductor', 'superconductor',
        # Civil/Environmental
        'water', 'wastewater', 'membrane', 'filtration', 'treatment',
        'soil', 'structural', 'seismic', 'geotechnical', 'construction',
        'sustainable', 'urban', 'transportation', 'traffic', 'infrastructure',
        # Biomedical
        'biomedical', 'medical', 'imaging', 'diagnostic', 'therapeutic',
        'wearable', 'implant', 'prosthetic', 'biosensor', 'drug',
        # General technical
        'data', 'database', 'visualization', 'interface', 'api',
        'modeling', 'analysis', 'processing', 'synthesis', 'characterization',
        'flow', 'heat', 'transfer', 'diffusion', 'reaction', 'kinetics',
        # Additional from corpus
        'federated', 'graph', 'knowledge', 'multimodal', 'heterogeneous',
        'sparse', 'dense', 'dynamic', 'static', 'hybrid', 'integrated',
        'video', 'image', 'audio', 'speech', 'language', 'text', 'visual',
        'spatial', 'temporal', 'spectral', 'acoustic', 'optical',
    }
    
    # Words that are context-dependent (can be important or not depending on context)
    CONTEXT_DEPENDENT = {
        # These are meaningful when combined with domain terms
        'system', 'systems', 'model', 'models', 'framework', 'frameworks',
        'design', 'architecture', 'platform', 'application', 'applications',
        'method', 'methods', 'approach', 'technique', 'techniques',
        'solution', 'solutions', 'tool', 'tools', 'technology', 'technologies',
        'development', 'implementation', 'integration', 'deployment',
        'performance', 'efficiency', 'accuracy', 'reliability', 'scalability',
        'hardware', 'software', 'middleware', 'firmware',
        'layer', 'stack', 'component', 'module', 'service', 'services',
        # Modifiers that can be meaningful in context
        'large', 'small', 'high', 'low', 'fast', 'slow', 'efficient', 'effective',
        'novel', 'advanced', 'modern', 'intelligent', 'smart', 'automated',
        'enhanced', 'improved', 'optimized', 'scalable', 'flexible', 'portable',
    }
    
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """Initialize with spaCy model."""
        import spacy
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", nlp_model], check=True)
            self.nlp = spacy.load(nlp_model)
    
    def extract(self, query: str) -> ExtractedKeywords:
        """
        Extract keywords with context-aware weighting.
        
        Strategy:
        1. Parse query and identify all content words
        2. Calculate domain specificity for each word
        3. Compute query-level context (how domain-specific is the overall query)
        4. Adjust individual word weights based on context
        """
        doc = self.nlp(query.lower())
        
        # Step 1: Extract all candidate tokens/phrases
        candidates = self._extract_candidates(doc)
        
        # Step 2: Calculate context - what's the domain specificity of this query?
        context_score = self._compute_query_context(candidates)
        
        # Step 3: Score each candidate with context awareness
        scored_keywords = []
        for candidate, base_info in candidates.items():
            final_score = self._context_aware_score(candidate, base_info, context_score, candidates)
            if final_score >= 0.3 and len(candidate) > 2:
                scored_keywords.append((candidate, final_score))
        
        # Sort by score
        scored_keywords.sort(key=lambda x: -x[1])
        
        # Build filtered query from top meaningful keywords
        filtered_query = self._build_filtered_query(scored_keywords, query)
        
        return ExtractedKeywords(
            keywords=scored_keywords,
            filtered_query=filtered_query,
            original_query=query
        )
    
    def _extract_candidates(self, doc) -> Dict[str, Dict]:
        """Extract candidate keywords and phrases with metadata."""
        candidates = {}
        
        # Extract noun chunks as phrases
        for chunk in doc.noun_chunks:
            # Filter stopwords and filter words from chunk
            tokens = [t for t in chunk if t.text.lower() not in self.STOPWORDS 
                     and t.text.lower() not in self.FILTER_WORDS
                     and not t.is_punct and not t.is_space]
            if tokens:
                phrase = " ".join(t.text for t in tokens)
                if len(phrase) > 2:
                    candidates[phrase] = {
                        'type': 'phrase',
                        'tokens': tokens,
                        'root': chunk.root.text.lower(),
                        'length': len(tokens),
                    }
        
        # Extract individual tokens
        for token in doc:
            word = token.text.lower()
            if (word not in self.STOPWORDS 
                and word not in self.FILTER_WORDS
                and not token.is_punct 
                and not token.is_space and len(word) > 2):
                if word not in candidates:
                    candidates[word] = {
                        'type': 'word',
                        'pos': token.pos_,
                        'dep': token.dep_,
                        'head': token.head.text.lower(),
                    }
        
        return candidates
    
    def _compute_query_context(self, candidates: Dict[str, Dict]) -> Dict:
        """
        Compute the overall domain context of the query.
        
        Returns context info including:
        - has_high_specificity: whether query has highly specific terms
        - high_spec_count: number of high specificity terms
        - medium_spec_count: number of medium specificity terms
        - domain_focus: estimated domain focus level (0-1)
        """
        high_spec_count = 0
        medium_spec_count = 0
        context_dependent_count = 0
        
        for candidate in candidates:
            words = candidate.lower().split()
            for word in words:
                if word in self.HIGH_SPECIFICITY_TERMS:
                    high_spec_count += 1
                elif word in self.MEDIUM_SPECIFICITY_TERMS:
                    medium_spec_count += 1
                elif word in self.CONTEXT_DEPENDENT:
                    context_dependent_count += 1
        
        # Compute domain focus score
        total_meaningful = high_spec_count + medium_spec_count + context_dependent_count
        if total_meaningful == 0:
            domain_focus = 0.3  # Low focus
        else:
            domain_focus = min(1.0, (high_spec_count * 0.4 + medium_spec_count * 0.2) / total_meaningful + 0.3)
        
        return {
            'has_high_specificity': high_spec_count > 0,
            'high_spec_count': high_spec_count,
            'medium_spec_count': medium_spec_count,
            'context_dependent_count': context_dependent_count,
            'domain_focus': domain_focus,
        }
    
    def _context_aware_score(self, candidate: str, info: Dict, 
                            context: Dict, all_candidates: Dict) -> float:
        """
        Score a candidate keyword with context awareness.
        
        Key insight: A word's importance depends on what else is in the query.
        - If query has high-specificity terms, generic words are less important
        - If query only has generic words, they all matter equally
        """
        words = candidate.lower().split()
        
        # Base score from word's own domain specificity
        high_spec_in_candidate = sum(1 for w in words if w in self.HIGH_SPECIFICITY_TERMS)
        medium_spec_in_candidate = sum(1 for w in words if w in self.MEDIUM_SPECIFICITY_TERMS)
        context_dep_in_candidate = sum(1 for w in words if w in self.CONTEXT_DEPENDENT)
        
        # Calculate intrinsic score
        if high_spec_in_candidate > 0:
            base_score = 0.85 + 0.05 * min(high_spec_in_candidate, 3)
        elif medium_spec_in_candidate > 0:
            base_score = 0.65 + 0.05 * min(medium_spec_in_candidate, 3)
        elif context_dep_in_candidate > 0:
            base_score = 0.5
        else:
            # Unknown word - could be domain-specific jargon or generic
            # Check if it's a noun (likely meaningful) or other POS
            if info.get('type') == 'word' and info.get('pos') in {'NOUN', 'PROPN'}:
                base_score = 0.55
            else:
                base_score = 0.4
        
        # Context adjustment: if query has high-specificity terms,
        # reduce weight of context-dependent and unknown words
        if context['has_high_specificity'] and context_dep_in_candidate > 0 and high_spec_in_candidate == 0:
            # This candidate is context-dependent and query has specific terms
            # Reduce its weight based on how specific the query is
            reduction = min(0.3, context['high_spec_count'] * 0.1)
            base_score -= reduction
        
        # Phrase bonus: multi-word phrases are more specific
        if info.get('type') == 'phrase' and info.get('length', 1) >= 2:
            # More bonus for phrases with domain terms
            if high_spec_in_candidate > 0 or medium_spec_in_candidate > 0:
                base_score += 0.15
            else:
                base_score += 0.08
        
        # Adjacency bonus: if this word modifies a high-specificity term
        if info.get('type') == 'word':
            head = info.get('head', '')
            if head in self.HIGH_SPECIFICITY_TERMS or head in self.MEDIUM_SPECIFICITY_TERMS:
                base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _build_filtered_query(self, scored_keywords: List[Tuple[str, float]], 
                             original_query: str) -> str:
        """Build a filtered query from top keywords."""
        if not scored_keywords:
            return original_query
        
        # Take keywords with score >= 0.5, up to 5
        top_keywords = [kw for kw, score in scored_keywords[:7] if score >= 0.5]
        
        if not top_keywords:
            top_keywords = [kw for kw, _ in scored_keywords[:3]]
        
        if not top_keywords:
            return original_query
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in top_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return " ".join(unique_keywords)


class QueryValidator:
    """
    Validates queries for relevance to engineering/research domains.
    
    Uses embedding similarity and heuristics to determine if a query
    is valid, irrelevant, or needs clarification.
    """
    
    # Common conversational/greeting patterns (definitely not research queries)
    CONVERSATIONAL_PATTERNS = {
        # Greetings
        'hello', 'hi', 'hey', 'greetings', 'howdy', 'hola', 'bonjour',
        'good morning', 'good afternoon', 'good evening', 'good night',
        'how are you', 'how do you do', "what's up", 'whats up', 'sup',
        # Chinese/other languages
        '你好', '您好', '早上好', '下午好', '晚上好', '再见', 'goodbye', 'bye',
        'こんにちは', 'こんばんは', 'おはよう', 'さようなら',
        # Weather
        'weather', 'rain', 'sunny', 'cloudy', 'temperature', 'forecast', 'climate today',
        'what is the weather', "what's the weather", 'how is the weather',
        # Time/Date
        'what time', 'what date', 'what day', 'today', 'tomorrow', 'yesterday',
        'current time', 'current date',
        # Conversational
        'thank you', 'thanks', 'please', 'sorry', 'excuse me', 'pardon',
        'yes', 'no', 'maybe', 'okay', 'ok', 'sure', 'fine', 'great', 'nice',
        # Questions about the system
        'who are you', 'what are you', 'what can you do', 'help me',
        'how does this work', 'what is this',
        # Jokes/casual
        'tell me a joke', 'sing a song', 'play a game', 'fun', 'funny',
        # Food/personal
        'hungry', 'food', 'eat', 'drink', 'tired', 'sleep', 'bored',
        # Generic non-technical
        'test', 'testing', 'just testing', 'ignore', 'nothing', 'random',
    }
    
    # Words that indicate completely off-topic queries (non-technical contexts)
    OFF_TOPIC_INDICATORS = {
        # Daily life / personal
        'weather', 'forecast', 'recipe', 'cooking', 'food', 'restaurant',
        'movie', 'film', 'music', 'song', 'singer', 'actor', 'celebrity',
        'sports', 'game', 'football', 'basketball', 'soccer', 'tennis',
        'travel', 'vacation', 'hotel', 'flight', 'booking', 'reservation',
        'shopping', 'price', 'discount', 'sale', 'buy', 'shop',
        'friend', 'family', 'relationship', 'dating', 'love', 'marriage',
        'news', 'politics', 'election', 'president', 'government',
        'joke', 'funny', 'laugh', 'humor', 'meme',
        # Non-research questions
        'meaning of life', 'who invented', 'when was', 'where is', 'why do',
    }
    
    # Engineering/Science domain keywords for relevance checking
    ENGINEERING_DOMAINS = {
        # Computer Science
        'computer', 'computing', 'software', 'hardware', 'algorithm', 'data', 
        'machine learning', 'artificial intelligence', 'ai', 'ml', 'deep learning',
        'neural network', 'programming', 'database', 'network', 'security',
        'cryptography', 'visualization', 'graphics', 'vision', 'nlp', 'robotics',
        # Electrical Engineering
        'electrical', 'electronic', 'circuit', 'signal', 'communication',
        'wireless', 'antenna', 'rf', 'microwave', 'photonics', 'semiconductor',
        'vlsi', 'asic', 'fpga', 'embedded', 'sensor', 'iot', 'power',
        # Mechanical Engineering
        'mechanical', 'thermal', 'fluid', 'dynamics', 'aerodynamics',
        'materials', 'manufacturing', 'cad', 'robotics', 'control', 'automation',
        'mems', 'nanotechnology', 'biomechanics',
        # Civil Engineering
        'civil', 'structural', 'geotechnical', 'environmental', 'water',
        'transportation', 'construction', 'sustainable', 'infrastructure',
        # Biomedical
        'biomedical', 'bioengineering', 'medical device', 'biosensor',
        'imaging', 'tissue', 'prosthetics', 'drug delivery',
        # Others
        'chemistry', 'physics', 'mathematics', 'optimization', 'simulation',
        'quantum', 'energy', 'renewable', 'battery', 'solar', 'climate',
    }
    
    # Non-engineering domains that may need redirect
    NON_ENGINEERING_HINTS = {
        'banking', 'finance', 'investment', 'stock', 'trading', 'accounting',
        'marketing', 'sales', 'advertising', 'business', 'management', 'hr',
        'campaign', 'branding', 'commerce', 'retail', 'consumer',
        'psychology', 'mental', 'disorder', 'therapy', 'counseling', 'psychiatric',
        'law', 'legal', 'litigation', 'contract', 'policy', 'legislation',
        'art', 'music', 'painting', 'sculpture', 'literature', 'poetry', 'drama',
        'history', 'philosophy', 'sociology', 'anthropology', 'political', 'economics',
        'religion', 'theology', 'sports', 'athletics', 'fitness', 'yoga',
        'cooking', 'culinary', 'fashion', 'beauty', 'entertainment', 'cinema',
        'journalism', 'media', 'editorial', 'publishing',
    }
    
    def __init__(
        self, 
        embedder=None,
        domain_embeddings: Optional[np.ndarray] = None,
        domain_texts: Optional[List[str]] = None,
        similarity_threshold: float = 0.25,
        weak_threshold: float = 0.35,
    ):
        """
        Initialize validator.
        
        Args:
            embedder: TextEmbedder for computing similarities
            domain_embeddings: Pre-computed embeddings of domain texts
            domain_texts: Domain text corpus (research interests, etc.)
            similarity_threshold: Below this = INVALID
            weak_threshold: Below this = WEAK_RELEVANCE
        """
        self.embedder = embedder
        self.domain_embeddings = domain_embeddings
        self.domain_texts = domain_texts or []
        self.similarity_threshold = similarity_threshold
        self.weak_threshold = weak_threshold
        
        # Compile domain keywords for quick lookup
        self._domain_words = set()
        for domain in self.ENGINEERING_DOMAINS:
            self._domain_words.update(domain.lower().split())
        
        self._non_eng_words = set()
        for hint in self.NON_ENGINEERING_HINTS:
            self._non_eng_words.update(hint.lower().split())
    
    def validate(self, query: str) -> QueryValidationResult:
        """
        Validate a query and return status with suggestions.
        """
        # Step 1: Basic format validation
        format_result = self._check_format(query)
        if format_result:
            return format_result
        
        # Step 2: Check for obviously non-engineering queries
        non_eng_result = self._check_non_engineering(query)
        if non_eng_result:
            return non_eng_result
        
        # Step 3: Check domain relevance using embeddings
        if self.embedder is not None and self.domain_embeddings is not None:
            relevance_result = self._check_domain_relevance(query)
            if relevance_result:
                return relevance_result
        
        # Step 4: Check if query has enough specificity
        specificity_result = self._check_specificity(query)
        if specificity_result:
            return specificity_result
        
        # Query passes all checks
        return QueryValidationResult(
            status=QueryStatus.VALID,
            message="Query is valid and relevant.",
            suggestions=[],
            confidence=0.9
        )
    
    def _check_format(self, query: str) -> Optional[QueryValidationResult]:
        """Check basic format requirements and detect off-topic queries."""
        query = query.strip()
        query_lower = query.lower()
        
        # Empty or whitespace only
        if not query:
            return QueryValidationResult(
                status=QueryStatus.INVALID,
                message="Please enter a valid query describing your research needs.",
                suggestions=[
                    "Try describing the technology or research area you're interested in",
                    "Example: 'machine learning for medical imaging'"
                ],
                confidence=1.0
            )
        
        # Pure numbers or very short meaningless strings
        if re.match(r'^[\d\s\W]+$', query):
            return QueryValidationResult(
                status=QueryStatus.INVALID,
                message="The query appears to be invalid (numbers or symbols only).",
                suggestions=[
                    "Please enter a descriptive query about your research needs",
                    "Example: 'wireless sensor networks for IoT'"
                ],
                confidence=1.0
            )
        
        # Check for gibberish/nonsense strings
        words = query_lower.split()
        if self._is_gibberish(words):
            return QueryValidationResult(
                status=QueryStatus.INVALID,
                message="Your query doesn't appear to be a valid research query.",
                suggestions=[
                    "Please enter a descriptive query about technology or research",
                    "Example: 'machine learning for medical imaging'"
                ],
                confidence=0.9
            )
        
        # Check for conversational/greeting patterns (exact or partial match)
        for pattern in self.CONVERSATIONAL_PATTERNS:
            # Exact match or query starts/ends with pattern
            if (query_lower == pattern or 
                query_lower.startswith(pattern + ' ') or
                query_lower.endswith(' ' + pattern) or
                ' ' + pattern + ' ' in ' ' + query_lower + ' '):
                return QueryValidationResult(
                    status=QueryStatus.INVALID,
                    message="This appears to be a casual conversation rather than a research query.",
                    suggestions=[
                        "This system helps match industry partners with professors based on research interests",
                        "Please enter a technical query, e.g., 'deep learning for autonomous vehicles'"
                    ],
                    confidence=0.95
                )
        
        # Check if query is mostly conversational words
        words = set(re.findall(r'\b\w+\b', query_lower))
        conversational_words = {'is', 'the', 'what', 'how', 'why', 'when', 'where', 'who',
                                'do', 'does', 'did', 'can', 'could', 'would', 'should',
                                'a', 'an', 'it', 'i', 'you', 'me', 'my', 'your', 'we',
                                'like', 'today', 'now', 'here', 'there', 'this', 'that'}
        non_conversational = words - conversational_words - {w for w in words if len(w) <= 2}
        
        # If after removing conversational words, nothing meaningful remains
        if len(non_conversational) == 0 and len(words) > 0:
            return QueryValidationResult(
                status=QueryStatus.INVALID,
                message="Your query doesn't contain any technical or research-related terms.",
                suggestions=[
                    "Please describe the technology or research area you're interested in",
                    "Example: 'machine learning', 'wireless communication', 'biomedical sensors'"
                ],
                confidence=0.9
            )
        
        # Check for off-topic indicators (non-technical queries) 
        off_topic_matches = words & self.OFF_TOPIC_INDICATORS
        if len(off_topic_matches) >= 1:
            # Check if there are any engineering terms to balance it out
            eng_terms_in_query = words & self._domain_words
            if len(eng_terms_in_query) == 0:
                return QueryValidationResult(
                    status=QueryStatus.INVALID,
                    message=f"Your query about '{', '.join(off_topic_matches)}' is not related to engineering research.",
                    suggestions=[
                        "This system is designed for engineering and technology research queries",
                        "Please enter a query about technical topics such as AI, robotics, materials science, etc."
                    ],
                    confidence=0.85
                )
        
        # Too short to be meaningful (single word after filtering)
        meaningful_words = [w for w in query.split() if len(w) > 1 and w.lower() not in conversational_words]
        if len(meaningful_words) < 1:
            return QueryValidationResult(
                status=QueryStatus.INVALID,
                message="Your query is too brief or doesn't contain research-related terms.",
                suggestions=[
                    "Add more context about your specific technical needs",
                    "Include the application domain or research area"
                ],
                confidence=0.85
            )
        
        return None
    
    def _check_non_engineering(self, query: str) -> Optional[QueryValidationResult]:
        """Check if query is clearly non-engineering."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Count non-engineering vs engineering matches
        non_eng_matches = query_words & self._non_eng_words
        eng_matches = query_words & self._domain_words
        
        # Exclude generic words that might overlap (like 'optimization')
        generic_overlap = {'optimization', 'analysis', 'system', 'systems', 'design', 'development'}
        actual_eng_matches = eng_matches - generic_overlap
        
        # Clear non-engineering query (2+ non-eng words, no real engineering words)
        if len(non_eng_matches) >= 2 and len(actual_eng_matches) == 0:
            matched_domains = ", ".join(list(non_eng_matches)[:3])
            return QueryValidationResult(
                status=QueryStatus.WEAK_RELEVANCE,
                message=f"Your query seems to be about '{matched_domains}', which may not align well with our engineering-focused expertise.",
                suggestions=[
                    "Our database focuses on engineering and technology research",
                    "Consider queries related to: AI/ML, robotics, materials, circuits, etc.",
                    "If your query has a technical aspect, try emphasizing that"
                ],
                confidence=0.75
            )
        
        # Predominantly non-engineering (more non-eng than actual eng)
        if len(non_eng_matches) >= 1 and len(non_eng_matches) > len(actual_eng_matches):
            matched_domains = ", ".join(list(non_eng_matches)[:3])
            return QueryValidationResult(
                status=QueryStatus.WEAK_RELEVANCE,
                message=f"Your query about '{matched_domains}' may not align well with our engineering expertise.",
                suggestions=[
                    "Our database focuses on engineering and technology research",
                    "Consider emphasizing the technical aspects of your query"
                ],
                confidence=0.65
            )
        
        # Single strong non-engineering indicator with no engineering context
        if len(non_eng_matches) >= 1 and len(actual_eng_matches) == 0:
            return QueryValidationResult(
                status=QueryStatus.WEAK_RELEVANCE,
                message="This query may have limited relevance to engineering research.",
                suggestions=[
                    "Try adding technical or engineering aspects to your query",
                    "Our experts specialize in engineering and technology fields"
                ],
                confidence=0.6
            )
        
        return None
    
    def _check_domain_relevance(self, query: str) -> Optional[QueryValidationResult]:
        """Check semantic relevance to research domains using embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.domain_embeddings)[0]
        
        max_similarity = float(np.max(similarities))
        mean_similarity = float(np.mean(similarities))
        
        # Find top matching domains for suggestions
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_domains = [self.domain_texts[i][:100] for i in top_indices if i < len(self.domain_texts)]
        
        if max_similarity < self.similarity_threshold:
            return QueryValidationResult(
                status=QueryStatus.INVALID,
                message="Your query doesn't appear to match any research areas in our database.",
                suggestions=[
                    "Try a query related to engineering or technology research",
                    f"Some areas we cover: {', '.join(d.split(';')[0] for d in top_domains[:2]) if top_domains else 'AI, robotics, materials, circuits'}"
                ],
                confidence=0.85
            )
        
        if max_similarity < self.weak_threshold:
            return QueryValidationResult(
                status=QueryStatus.WEAK_RELEVANCE,
                message="Your query has weak relevance to our research areas.",
                suggestions=[
                    "Consider being more specific about the technical aspects",
                    f"Related areas: {', '.join(d.split(';')[0] for d in top_domains[:2])}" if top_domains else ""
                ],
                confidence=0.7
            )
        
        return None
    
    def _is_gibberish(self, words: List[str]) -> bool:
        """
        Detect if the query is gibberish/meaningless.
        
        Gibberish indicators:
        - Very short words only (all <= 3 chars)
        - Repetitive patterns (all same length short words)
        - No recognizable English words
        - Single long random string
        """
        if not words:
            return True
        
        # Common English words to check against (very basic vocab)
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it',
            'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this',
            'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
            'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come',
            'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
            'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were',
            # Technical words we allow
            'data', 'system', 'design', 'model', 'network', 'learning', 'machine',
            'deep', 'neural', 'sensor', 'control', 'robot', 'image', 'signal',
            'power', 'energy', 'material', 'analysis', 'algorithm', 'optimization',
        }
        
        # Single word gibberish check
        if len(words) == 1:
            word = words[0]
            # Check for keyboard smash patterns (consecutive keyboard rows)
            keyboard_rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
            for row in keyboard_rows:
                # If word consists mostly of consecutive chars from same row
                row_chars = sum(1 for c in word if c in row)
                if len(word) >= 5 and row_chars / len(word) >= 0.7:
                    return True
            
            # Long random string with unusual vowel ratio
            if len(word) > 5:
                vowels = sum(1 for c in word if c in 'aeiou')
                # Too few vowels or single word not in vocabulary
                if vowels == 0 or (vowels / len(word) < 0.15 and word not in common_words and word not in self._domain_words):
                    return True
                # Check if it's a real word by checking consonant clusters
                consonants_in_row = 0
                max_consonants = 0
                for c in word:
                    if c not in 'aeiou':
                        consonants_in_row += 1
                        max_consonants = max(max_consonants, consonants_in_row)
                    else:
                        consonants_in_row = 0
                # More than 4 consonants in a row is unusual for English
                if max_consonants >= 5 and word not in common_words and word not in self._domain_words:
                    return True
                    
            # Short single word not in common vocabulary
            if len(word) <= 4 and word not in common_words and word not in self._domain_words:
                return True
        
        # All very short words pattern (like "aaa bbb ccc")
        if all(len(w) <= 3 for w in words) and len(words) >= 2:
            # Check if any are recognizable
            recognized = sum(1 for w in words if w in common_words or w in self._domain_words)
            if recognized == 0:
                return True
        
        # Check if all words are similar length and none recognizable (gibberish pattern)
        if len(words) >= 2:
            lengths = [len(w) for w in words]
            # All same short length (like "xyz abc def")
            if len(set(lengths)) == 1 and lengths[0] <= 4:
                recognized = sum(1 for w in words if w in common_words or w in self._domain_words)
                if recognized == 0:
                    return True
        
        return False
    
    def _check_specificity(self, query: str) -> Optional[QueryValidationResult]:
        """Check if query is specific enough."""
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        
        # Very generic queries (all words are context-dependent)
        generic_only = all(
            w in KeywordExtractor.CONTEXT_DEPENDENT 
            or w in KeywordExtractor.FILTER_WORDS
            or w in KeywordExtractor.STOPWORDS
            for w in query_words
        )
        
        if generic_only and len(query_words) <= 3:
            return QueryValidationResult(
                status=QueryStatus.NEEDS_CLARIFICATION,
                message="Your query is too generic. Please be more specific.",
                suggestions=[
                    "Add specific technologies or methods (e.g., 'deep learning', 'MEMS')",
                    "Mention the application domain (e.g., 'healthcare', 'automotive')"
                ],
                confidence=0.7
            )
        
        return None


class EnhancedQueryProcessor:
    """
    Combines query validation and keyword extraction for enhanced matching.
    """
    
    def __init__(
        self,
        embedder=None,
        domain_embeddings: Optional[np.ndarray] = None,
        domain_texts: Optional[List[str]] = None,
        similarity_threshold: float = 0.25,
        weak_threshold: float = 0.35,
    ):
        self.keyword_extractor = KeywordExtractor()
        self.validator = QueryValidator(
            embedder=embedder,
            domain_embeddings=domain_embeddings,
            domain_texts=domain_texts,
            similarity_threshold=similarity_threshold,
            weak_threshold=weak_threshold,
        )
    
    def process(self, query: str) -> Tuple[QueryValidationResult, Optional[ExtractedKeywords]]:
        """
        Process a query: validate and extract keywords.
        
        Returns:
            Tuple of (validation_result, extracted_keywords or None if invalid)
        """
        # First validate
        validation = self.validator.validate(query)
        
        # Only extract keywords if query is at least somewhat valid
        if validation.status in {QueryStatus.VALID, QueryStatus.WEAK_RELEVANCE, QueryStatus.NEEDS_CLARIFICATION}:
            keywords = self.keyword_extractor.extract(query)
            return validation, keywords
        
        return validation, None
    
    def get_enhanced_query(self, query: str) -> Tuple[str, QueryValidationResult, Optional[ExtractedKeywords]]:
        """
        Get an enhanced query for matching.
        
        If keywords are extracted, returns a weighted combination.
        """
        validation, keywords = self.process(query)
        
        if keywords and validation.status != QueryStatus.INVALID:
            # Build enhanced query from keywords
            enhanced = keywords.filtered_query
            if enhanced and enhanced != query:
                return enhanced, validation, keywords
        
        return query, validation, keywords
