from __future__ import annotations

from typing import Dict, List, Tuple
import torch
from CodonTransformer.CodonUtils import ORGANISM2ID
from mocps.metrics import CIS_COMPILED, compute_cfd, compute_cis, compute_csi
from mocps.sequences import get_codon_maps, normalize_dna, normalize_protein

class ICROptimizer:
    def __init__(self, model, tokenizer, organism: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]], csi_weights: Dict[str, float], device: torch.device, w_csi: float = 2.0, w_cfd: float = 1.0, w_cis: float = 2.0):
        self.model = model
        self.tokenizer = tokenizer
        self.organism = organism
        self.organism_id = ORGANISM2ID.get(organism, 0)
        self.codon_freq = codon_frequencies
        self.csi_weights = csi_weights
        self.device = device
        self.w_csi = w_csi
        self.w_cfd = w_cfd
        self.w_cis = w_cis
        self.norm = {'CSI': {'mean': 0.937, 'std': 0.014}, 'CFD': {'mean': 0.065, 'std': 0.101}, 'CIS': {'mean': 0.5, 'std': 1.0}}
        self.codon2aa, self.aa2codons = get_codon_maps()
        self.rare_codons = self._build_rare_codons()

    def _build_rare_codons(self, threshold: float = 0.3) -> set[str]:
        rare = set()
        for aa, (codons, freqs) in self.codon_freq.items():
            if not freqs:
                continue
            max_freq = max(freqs)
            if max_freq == 0:
                continue
            for codon, freq in zip(codons, freqs):
                if freq / max_freq < threshold:
                    rare.add(codon.upper())
        return rare

    def _compute_reward(self, dna: str) -> tuple[float, dict]:
        dna = normalize_dna(dna)
        csi = compute_csi(dna, self.csi_weights)
        cfd = compute_cfd(dna, self.codon_freq) / 100.0
        cis = compute_cis(dna)
        csi_z = (csi - self.norm['CSI']['mean']) / self.norm['CSI']['std']
        cis_z = -(cis - self.norm['CIS']['mean']) / self.norm['CIS']['std']
        cfd_target = max(cfd, 0.02)
        cfd_z = -(cfd_target - self.norm['CFD']['mean']) / self.norm['CFD']['std']
        reward = self.w_csi * csi_z + self.w_cfd * cfd_z + self.w_cis * cis_z
        return reward, {'CSI': csi, 'CFD': cfd, 'CIS': cis, 'reward': reward}

    def _get_position_scores(self, dna: str) -> list[float]:
        codons = [dna[i:i+3] for i in range(0, len(dna) - 2, 3)]
        scores = []
        for i, codon in enumerate(codons):
            score = 0.0
            if codon.upper() in self.rare_codons:
                score += 2.0
            start = max(0, i * 3 - 6)
            end = min(len(dna), i * 3 + 9)
            window = dna[start:end].upper()
            score += sum(len(pattern.findall(window)) for pattern in CIS_COMPILED)
            scores.append(score)
        return scores

    def _model_score_codon(self, protein: str, dna: str, pos: int, candidate_codon: str) -> float:
        aa = protein[pos] if pos < len(protein) else '_'
        tokens = []
        for i, amino in enumerate(protein):
            codon = dna[i * 3:(i + 1) * 3] if i * 3 + 3 <= len(dna) else None
            if i == pos:
                tokens.append(f'{amino.lower()}_unk')
            elif codon and len(codon) == 3:
                tokens.append(f'{amino.lower()}_{codon.lower()}')
            else:
                tokens.append(f'{amino.lower()}_unk')
        enc = self.tokenizer(' '.join(tokens), return_tensors='pt', padding=True, truncation=True, max_length=2048)
        input_ids = enc['input_ids'].to(self.device)
        attn_mask = enc['attention_mask'].to(self.device)
        type_ids = torch.full_like(input_ids, self.organism_id).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=type_ids).logits[0, pos + 1]
        target_tok = f'{aa.lower()}_{candidate_codon.lower()}'
        tok_id = self.tokenizer.get_vocab().get(target_tok)
        if tok_id is None:
            return -999.0
        return torch.log_softmax(logits, dim=-1)[tok_id].item()

    def optimize(self, protein: str, initial_dna: str, n_rounds: int = 2, top_k_frac: float = 0.15) -> tuple[str, list[dict]]:
        protein = normalize_protein(protein)
        dna = normalize_dna(initial_dna)
        current_reward, current_metrics = self._compute_reward(dna)
        history = [{'round': 0, **current_metrics}]
        n_codons = len(dna) // 3
        k = max(1, int(n_codons * top_k_frac))
        for round_idx in range(1, n_rounds + 1):
            pos_scores = self._get_position_scores(dna)
            top_positions = sorted(range(len(pos_scores)), key=lambda i: pos_scores[i], reverse=True)[:k]
            improved = False
            for pos in top_positions:
                if pos >= len(protein) - 1:
                    continue
                aa = protein[pos]
                current_codon = dna[pos * 3:(pos + 1) * 3]
                synonymous = self.aa2codons.get(aa, [])
                if len(synonymous) <= 1:
                    continue
                best_codon = current_codon
                best_score = -float('inf')
                for candidate in synonymous:
                    new_dna = dna[:pos * 3] + candidate + dna[(pos + 1) * 3:]
                    new_reward, _ = self._compute_reward(new_dna)
                    model_lp = self._model_score_codon(protein, dna, pos, candidate)
                    combined = new_reward + 0.1 * model_lp
                    if combined > best_score:
                        best_score = combined
                        best_codon = candidate
                if best_codon != current_codon:
                    new_dna = dna[:pos * 3] + best_codon + dna[(pos + 1) * 3:]
                    new_reward, _ = self._compute_reward(new_dna)
                    if new_reward >= current_reward - 0.1:
                        dna = new_dna
                        current_reward = new_reward
                        improved = True
            current_reward, current_metrics = self._compute_reward(dna)
            history.append({'round': round_idx, **current_metrics})
            if not improved:
                break
        return dna, history
