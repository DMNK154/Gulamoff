# app/cross_reference.py
"""
Cross-Reference Engine for GPT-GU
Discovers complementary and oppositional glyph relationships
"""
from __future__ import annotations
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from pathlib import Path

# Import existing lexicon
from app.guardrails import LEX_DICT2GLYPH, PAIR_TO_TEXT

class GlyphCrossReference:
    """
    Analyzes glyph relationships from scroll dataset.
    Finds complementary glyphs, co-occurrences, and transformation patterns.
    """

    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "glyph_dataset_jsonl"

        self.dataset_path = dataset_path

        # Inverse lexicon: glyph -> name
        self.glyph_to_name = {v: k for k, v in LEX_DICT2GLYPH.items()}
        self.name_to_glyph = LEX_DICT2GLYPH.copy()

        # Relationship data structures
        self.co_occurrences: Dict[str, Counter] = defaultdict(Counter)
        self.formula_patterns: Dict[str, List[str]] = defaultdict(list)
        self.transformations: Dict[str, Set[str]] = defaultdict(set)
        self.oppositions: Dict[str, Set[str]] = defaultdict(set)

        # Context storage for each glyph
        self.glyph_contexts: Dict[str, List[str]] = defaultdict(list)

        # Load oppositions from existing PAIR_TO_TEXT
        self._load_oppositions()

        # Parse dataset
        self._parse_dataset()

    def _load_oppositions(self):
        """Load oppositional pairs from existing PAIR_TO_TEXT mapping."""
        for (glyph_a, glyph_b), (name_a, name_b) in PAIR_TO_TEXT.items():
            self.oppositions[glyph_a].add(glyph_b)
            self.oppositions[glyph_b].add(glyph_a)

    def _parse_dataset(self):
        """Parse the JSONL dataset to extract glyph relationships."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        self._process_entry(entry)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON on line {line_num}")
        except FileNotFoundError:
            print(f"Dataset not found at {self.dataset_path}")

    def _process_entry(self, entry: dict):
        """Process a single dataset entry to extract relationships."""
        task = entry.get("tag", "")

        if task == "SCROLL_FORMULA":
            # Extract glyph sequence and meaning
            prompt = entry.get("input", "")
            output = entry.get("output", "")

            # Extract glyph sequence from PROMPT: line
            glyph_match = re.search(r'PROMPT:\s*(.+?)\s*\n', prompt)
            if glyph_match:
                glyph_seq = glyph_match.group(1).strip()
                self._analyze_formula(glyph_seq, output)

        elif task == "SCROLL_SUMMARY":
            # Extract context from summaries
            prompt = entry.get("input", "")
            summary = entry.get("output", "")

            # Find glyphs in the prompt
            glyph_match = re.search(r'PROMPT:\s*(.+?)-', prompt)
            if glyph_match:
                glyph_seq = glyph_match.group(1).strip()
                glyphs = self._extract_glyphs(glyph_seq)

                # Store context for each glyph
                for glyph in glyphs:
                    self.glyph_contexts[glyph].append(summary[:200])

    def _analyze_formula(self, glyph_seq: str, meaning: str):
        """Analyze a glyph formula to find co-occurrence and transformation patterns."""
        # Extract individual glyphs
        glyphs = self._extract_glyphs(glyph_seq)

        # Record co-occurrences (all pairs in this formula)
        for i, glyph_a in enumerate(glyphs):
            for glyph_b in glyphs[i+1:]:
                # Mutual co-occurrence
                self.co_occurrences[glyph_a][glyph_b] += 1
                self.co_occurrences[glyph_b][glyph_a] += 1

            # Store the full formula pattern
            self.formula_patterns[glyph_a].append(glyph_seq)

        # Detect transformations (A → B patterns)
        if '→' in glyph_seq or '⇒' in glyph_seq:
            parts = re.split(r'[→⇒]', glyph_seq)
            if len(parts) >= 2:
                left_glyphs = self._extract_glyphs(parts[0])
                right_glyphs = self._extract_glyphs(parts[1])

                for left in left_glyphs:
                    for right in right_glyphs:
                        self.transformations[left].add(right)

    def _extract_glyphs(self, text: str) -> List[str]:
        """Extract individual glyphs from a text sequence."""
        # Split on common separators but preserve individual glyphs
        # Remove operators and separators
        cleaned = re.sub(r'[→⇒+:\-\s/]', ' ', text)

        glyphs = []
        for char in cleaned:
            if char.strip() and char in self.glyph_to_name:
                glyphs.append(char)

        # If no glyphs found, try character by character on original
        if not glyphs:
            for char in text:
                if char in self.glyph_to_name:
                    glyphs.append(char)

        return glyphs

    def get_complementary_glyphs(self, glyph_or_name: str, top_n: int = 5) -> List[Tuple[str, str, int]]:
        """
        Find glyphs that complement the given glyph.
        Returns: [(glyph, name, co_occurrence_count), ...]
        """
        # Convert name to glyph if needed
        glyph = glyph_or_name if glyph_or_name in self.glyph_to_name else self.name_to_glyph.get(glyph_or_name)

        if not glyph or glyph not in self.co_occurrences:
            return []

        # Get co-occurrences, sorted by frequency
        complementary = []
        for other_glyph, count in self.co_occurrences[glyph].most_common(top_n * 2):
            # Skip if this is an opposition (we want complements, not opposites)
            if other_glyph in self.oppositions.get(glyph, set()):
                continue

            name = self.glyph_to_name.get(other_glyph, "Unknown")
            complementary.append((other_glyph, name, count))

            if len(complementary) >= top_n:
                break

        return complementary

    def get_oppositions(self, glyph_or_name: str) -> List[Tuple[str, str]]:
        """
        Get oppositional glyphs (from CNN training data).
        Returns: [(glyph, name), ...]
        """
        # Convert name to glyph if needed
        glyph = glyph_or_name if glyph_or_name in self.glyph_to_name else self.name_to_glyph.get(glyph_or_name)

        if not glyph or glyph not in self.oppositions:
            return []

        return [(opp, self.glyph_to_name.get(opp, "Unknown"))
                for opp in self.oppositions[glyph]]

    def get_transformations(self, glyph_or_name: str) -> List[Tuple[str, str]]:
        """
        Get glyphs that this one transforms into (A → B patterns).
        Returns: [(glyph, name), ...]
        """
        # Convert name to glyph if needed
        glyph = glyph_or_name if glyph_or_name in self.glyph_to_name else self.name_to_glyph.get(glyph_or_name)

        if not glyph or glyph not in self.transformations:
            return []

        return [(trans, self.glyph_to_name.get(trans, "Unknown"))
                for trans in self.transformations[glyph]]

    def get_formula_examples(self, glyph_or_name: str, max_examples: int = 3) -> List[str]:
        """Get example formulas containing this glyph."""
        # Convert name to glyph if needed
        glyph = glyph_or_name if glyph_or_name in self.glyph_to_name else self.name_to_glyph.get(glyph_or_name)

        if not glyph or glyph not in self.formula_patterns:
            return []

        return self.formula_patterns[glyph][:max_examples]

    def get_context(self, glyph_or_name: str, max_contexts: int = 2) -> List[str]:
        """Get scroll contexts mentioning this glyph."""
        # Convert name to glyph if needed
        glyph = glyph_or_name if glyph_or_name in self.glyph_to_name else self.name_to_glyph.get(glyph_or_name)

        if not glyph or glyph not in self.glyph_contexts:
            return []

        return self.glyph_contexts[glyph][:max_contexts]

    def analyze(self, glyph_or_name: str) -> Dict:
        """
        Complete analysis of a glyph's relationships.
        Returns all relationship types in one structured response.
        """
        # Convert to glyph and get name
        if glyph_or_name in self.glyph_to_name:
            glyph = glyph_or_name
            name = self.glyph_to_name[glyph]
        else:
            name = glyph_or_name
            glyph = self.name_to_glyph.get(glyph_or_name, glyph_or_name)

        return {
            "glyph": glyph,
            "name": name,
            "complementary": [
                {"glyph": g, "name": n, "frequency": f}
                for g, n, f in self.get_complementary_glyphs(glyph, top_n=5)
            ],
            "oppositions": [
                {"glyph": g, "name": n}
                for g, n in self.get_oppositions(glyph)
            ],
            "transformations": [
                {"glyph": g, "name": n}
                for g, n in self.get_transformations(glyph)
            ],
            "formula_examples": self.get_formula_examples(glyph, max_examples=3),
            "contexts": self.get_context(glyph, max_contexts=2)
        }

    def suggest_combinations(self, glyphs: List[str]) -> Dict:
        """
        Given multiple glyphs, suggest what other glyphs might complete the formula.
        """
        # Convert names to glyphs if needed
        glyph_list = []
        for item in glyphs:
            if item in self.glyph_to_name:
                glyph_list.append(item)
            elif item in self.name_to_glyph:
                glyph_list.append(self.name_to_glyph[item])

        # Count co-occurrences across all input glyphs
        suggestions = Counter()
        for glyph in glyph_list:
            if glyph in self.co_occurrences:
                for other, count in self.co_occurrences[glyph].items():
                    if other not in glyph_list:  # Don't suggest glyphs already in input
                        suggestions[other] += count

        # Return top suggestions
        top_suggestions = []
        for glyph, score in suggestions.most_common(5):
            name = self.glyph_to_name.get(glyph, "Unknown")
            top_suggestions.append({"glyph": glyph, "name": name, "score": score})

        return {
            "input_glyphs": glyph_list,
            "suggestions": top_suggestions
        }


# Singleton instance
_cross_ref_engine: GlyphCrossReference = None

def get_cross_reference_engine() -> GlyphCrossReference:
    """Get or create the singleton cross-reference engine."""
    global _cross_ref_engine
    if _cross_ref_engine is None:
        _cross_ref_engine = GlyphCrossReference()
    return _cross_ref_engine


# Convenience functions for use in UI
def analyze_glyph(glyph_or_name: str) -> Dict:
    """Analyze a glyph's relationships."""
    engine = get_cross_reference_engine()
    return engine.analyze(glyph_or_name)


def find_complementary(glyph_or_name: str, top_n: int = 5) -> List[Tuple[str, str, int]]:
    """Find complementary glyphs."""
    engine = get_cross_reference_engine()
    return engine.get_complementary_glyphs(glyph_or_name, top_n=top_n)


def find_oppositions(glyph_or_name: str) -> List[Tuple[str, str]]:
    """Find oppositional glyphs."""
    engine = get_cross_reference_engine()
    return engine.get_oppositions(glyph_or_name)


__all__ = [
    'GlyphCrossReference',
    'get_cross_reference_engine',
    'analyze_glyph',
    'find_complementary',
    'find_oppositions',
]
