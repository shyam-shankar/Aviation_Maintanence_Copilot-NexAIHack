"""
Aviation Text Normalizer for Maintenance Logbooks
Expands abbreviations using MaintNet lexicon + standard aviation maintenance shorthand.
"""

import os
import re
import csv
from typing import Dict, List, Tuple

# Additional standard aviation maintenance abbreviations frequently seen in aircraft logbooks
COMMON_MAINT_ABBREVIATIONS = {
    "r&r": "removed and replaced",
    "r & r": "removed and replaced",
    "rem": "removed",
    "repl": "replaced",
    "rep": "repair",
    "a/c": "aircraft",
    "ac": "aircraft",
    "l/h": "left hand",
    "lh": "left hand",
    "r/h": "right hand",
    "rh": "right hand",
    "inop": "inoperative",
    "o/h": "overhauled",
    "oh": "overhaul",
    "prop": "propeller",
    "rpm": "revolutions per minute",
    "psi": "pounds per square inch",
    "p/n": "part number",
    "pn": "part number",
    "s/n": "serial number",
    "sn": "serial number",
    "nav": "navigation",
    "comm": "communication",
    "vac": "vacuum",
    "elec": "electrical",
    "tach": "tachometer",
    "adj": "adjusted",
    "inst": "installed",
    "serv": "serviced",
    "svc": "serviced",
    "sec": "secured",
    "tight": "tightened",
    "fwd": "forward",
    "aft": "after",
    "cyl": "cylinder",
    "cyls": "cylinders",
    "eng": "engine",
    "engs": "engines",
    "alt": "alternator",
    "carb": "carburetor",
    "batt": "battery",
    "comp": "compressor",
    "exh": "exhaust",
    "egt": "exhaust gas temperature",
    "cht": "cylinder head temperature",
    "mixt": "mixture",
    "ovrhd": "overhead",
    "temp": "temperature",
    "press": "pressure",
    "leak": "leaking",
    "leaks": "leaking",
    "leaking": "leaking",
    "wo": "without",
    "w/": "with",
    "w/o": "without",
}

class AviationNormalizer:
    def __init__(self, abbreviation_csv_path: str = None):
        self.abbr_map: Dict[str, str] = {}
        # Load custom/common dictionary first
        for k, v in COMMON_MAINT_ABBREVIATIONS.items():
            self.abbr_map[k.lower()] = v.lower()

        # Load MaintNet abbreviations if provided
        if abbreviation_csv_path and os.path.exists(abbreviation_csv_path):
            self._load_maintnet_abbreviations(abbreviation_csv_path)

        # Precompile regex for token expansion
        # Words, hyphenated terms, and symbols like & or /
        self.token_pattern = re.compile(r"[a-zA-Z0-9]+(?:[/'&][a-zA-Z0-9]+)*|#\d+")

    def _load_maintnet_abbreviations(self, path: str):
        with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                abbr = row.get("Abbreviated", "").strip().lower()
                desc = row.get("Standard_Description", "").strip().lower()
                # Skip overly ambiguous single-letter abbreviations like 'q' -> 'quit'
                if abbr and desc and len(abbr) > 1:
                    self.abbr_map[abbr] = desc

    def expand_token(self, token: str) -> str:
        clean_token = token.lower().strip()
        # Direct lookup
        if clean_token in self.abbr_map:
            return self.abbr_map[clean_token]
        # Handle cylinder notations like #2 -> cylinder 2, #2 & 4 -> cylinder 2 and 4
        if clean_token.startswith("#") and clean_token[1:].isdigit():
            return f"cylinder {clean_token[1:]}"
        return clean_token

    def normalize(self, text: str) -> str:
        """
        Normalizes a raw logbook text:
        - Expands multi-word and compound abbreviations (e.g., R&R, A/C, L/H)
        - Expands single-token abbreviations (e.g., CYL -> cylinder)
        - Standardizes cylinder numbers (#2 -> cylinder 2)
        - Produces clean tokens
        """
        if not text or not isinstance(text, str):
            return ""

        s = text.lower()
        # Phrase/compound replacements
        compound_replacements = [
            (r"\br\s*&\s*r\b", "removed and replaced"),
            (r"\br\s*/\s*r\b", "removed and replaced"),
            (r"\ba\s*/\s*c\b", "aircraft"),
            (r"\bl\s*/\s*h\b", "left hand"),
            (r"\br\s*/\s*h\b", "right hand"),
            (r"\bw\s*/\s*o\b", "without"),
            (r"\bw\s*/\b", "with"),
            (r"\bp\s*/\s*n\b", "part number"),
            (r"\bs\s*/\s*n\b", "serial number"),
            (r"#(\d+)", r"cylinder \1"),
            (r"\bpowerloss\b", "power loss"),
            (r"\bchoking\b", "choke"),
            (r"\bloss\s+of\s+power\b", "lose power"),
            (r"\blost\s+power\b", "lose power"),
            (r"\bvibrating\b", "vibration"),
            (r"\bshaking\b", "vibration"),
            (r"\bhesitating\b", "hesitation"),
            (r"\bstalling\b", "stall"),
            (r"\boverheating\b", "overheat"),
            (r"\bbackfiring\b", "backfire"),
        ]
        for pattern, repl in compound_replacements:
            s = re.sub(pattern, repl, s)

        # Tokenize words
        tokens = re.findall(r"[a-z0-9]+", s)
        expanded_tokens = []
        for t in tokens:
            if t in self.abbr_map:
                expanded_tokens.append(self.abbr_map[t])
            else:
                expanded_tokens.append(t)

        normalized = " ".join(expanded_tokens)
        return re.sub(r"\s+", " ", normalized).strip()

    def tokenize(self, text: str) -> List[str]:
        """Returns normalized tokens for BM25"""
        norm_text = self.normalize(text)
        return [t for t in norm_text.split() if len(t) > 1]


if __name__ == "__main__":
    # Quick sanity check
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    abbr_path = os.path.join(BASE_DIR, "dataset", "Aviation_Abbreviation_Dataset.csv")
    norm = AviationNormalizer(abbr_path)
    sample = "#2 & 4 CYL ROCKER COVER GASKETS ARE LEAKING. R&R WITH NEW."
    print("Original:  ", sample)
    print("Normalized:", norm.normalize(sample))
    print("Tokens:    ", norm.tokenize(sample))
