from __future__ import annotations

import re
from typing import Dict, List, Tuple
from mocps.sequences import normalize_dna

CIS_PATTERNS = [
    r'AATAAA', r'ATTAAA', r'AGTAAA', r'TATAAA', r'GGTAAG', r'AGGTAA',
    r'TTTTTTT', r'AAAAAAA', r'CCCCCC', r'GGGGGG', r'ATTTA',
]
CIS_COMPILED = [re.compile(pattern) for pattern in CIS_PATTERNS]

def compute_cis(dna: str) -> int:
    dna = normalize_dna(dna)
    return sum(len(pattern.findall(dna)) for pattern in CIS_COMPILED)

def compute_gc(dna: str) -> float:
    dna = normalize_dna(dna)
    return (dna.count('G') + dna.count('C')) / len(dna) * 100 if dna else 0.0

def compute_mfe_or_none(dna: str):
    try:
        import RNA
    except ImportError:
        return None
    _struct, mfe = RNA.fold(normalize_dna(dna).replace('T', 'U'))
    return float(mfe)

def compute_csi(dna: str, csi_weights: Dict[str, float]) -> float:
    from CodonTransformer.CodonEvaluation import get_CSI_value
    return float(get_CSI_value(normalize_dna(dna), csi_weights))

def compute_cfd(dna: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]], threshold: float = 0.3) -> float:
    from CodonTransformer.CodonEvaluation import get_cfd
    return float(get_cfd(normalize_dna(dna), codon_frequencies, threshold))
