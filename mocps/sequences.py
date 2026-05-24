from __future__ import annotations

from typing import Dict, Tuple

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S','TAT':'Y','TAC':'Y','TAA':'_','TAG':'_','TGT':'C','TGC':'C','TGA':'_','TGG':'W',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T','AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A','GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def normalize_dna(value: str) -> str:
    return ''.join(ch for ch in value.upper().replace('U', 'T') if ch in 'ACGT')

def normalize_protein(value: str) -> str:
    protein = ''.join(ch for ch in value.upper().replace('*', '_') if not ch.isspace())
    return protein if protein.endswith('_') else protein + '_'

def dna_to_protein(dna: str) -> str:
    dna = normalize_dna(dna)
    return ''.join(CODON_TABLE.get(dna[i:i+3], 'X') for i in range(0, len(dna) - 2, 3))

def get_codon_maps() -> Tuple[Dict[str, str], Dict[str, list[str]]]:
    codon2aa = dict(CODON_TABLE)
    aa2codons: Dict[str, list[str]] = {}
    for codon, aa in codon2aa.items():
        aa2codons.setdefault(aa, []).append(codon)
    return codon2aa, aa2codons

def build_codon_to_amino(codon_frequencies):
    mapping = {}
    for aa, (codons, _freqs) in codon_frequencies.items():
        for codon in codons:
            mapping[codon.upper()] = aa
    return mapping
