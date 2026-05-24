import statistics
from typing import Dict, List, Tuple

from mocps.metrics import (
    build_cousin_ref,
    compute_cfd,
    compute_cis,
    compute_cousin,
    compute_csi,
    update_norm_stats,
    z_score,
)


class RewardCalculator:
    def __init__(
        self,
        natural_dnas: List[str],
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
        w_csi: float = 1.0,
        w_cfd: float = 1.0,
        w_cousin: float = 1.0,
        w_cis: float = 1.0,
    ):
        print("[RewardCalc] 正在构建 CSI weights...")
        from CodonTransformer.CodonEvaluation import get_CSI_weights

        self.csi_weights = get_CSI_weights(natural_dnas)
        print("[RewardCalc] 正在构建 COUSIN ref_freq...")
        self.ref_freq = build_cousin_ref(natural_dnas, codon_frequencies)
        self.codon_frequencies = codon_frequencies
        self.w_csi = w_csi
        self.w_cfd = w_cfd
        self.w_cousin = w_cousin
        self.w_cis = w_cis
        print("[RewardCalc] 初始化完成。")
        print(f"  权重: CSI={w_csi}, CFD={w_cfd}, COUSIN={w_cousin}, CIS={w_cis}")

    def compute(self, dna: str) -> Tuple[float, Dict[str, float]]:
        csi = compute_csi(dna, self.csi_weights)
        cfd = compute_cfd(dna, self.codon_frequencies)
        cousin = compute_cousin(dna, self.ref_freq, self.codon_frequencies)
        cis = compute_cis(dna)

        csi_z = z_score(csi, "CSI")
        cfd_z = -z_score(cfd, "CFD")
        cousin_z = z_score(cousin, "COUSIN")
        cis_z = -z_score(cis, "CIS")
        reward = self.w_csi * csi_z + self.w_cfd * cfd_z + self.w_cis * cis_z

        return reward, {
            "CSI": csi,
            "CFD": cfd,
            "COUSIN": cousin,
            "CIS": cis,
            "CSI_z": csi_z,
            "CFD_z": cfd_z,
            "COUSIN_z": cousin_z,
            "CIS_z": cis_z,
        }

    def compute_batch(self, dnas: List[str]) -> Tuple[List[float], List[Dict[str, float]]]:
        rewards, metrics_list = [], []
        for dna in dnas:
            if not dna or len(dna) < 3:
                rewards.append(float("nan"))
                metrics_list.append(
                    {"CSI": float("nan"), "CFD": float("nan"), "COUSIN": float("nan"), "CIS": 0}
                )
                continue
            reward, metrics = self.compute(dna)
            rewards.append(reward)
            metrics_list.append(metrics)
        return rewards, metrics_list

    def calibrate_norm_stats(self, dnas: List[str], n_samples: int = 500) -> None:
        dnas = dnas[:n_samples]
        print(f"[RewardCalc] 正在校准归一化统计数据（{len(dnas)} 条序列）...")

        cis_vals = [compute_cis(dna) for dna in dnas]
        cousin_vals = [
            compute_cousin(dna, self.ref_freq, self.codon_frequencies) for dna in dnas
        ]
        cousin_vals = [value for value in cousin_vals if not (value != value)]

        if cis_vals:
            update_norm_stats(
                "CIS",
                mean=statistics.mean(cis_vals),
                std=statistics.stdev(cis_vals) if len(cis_vals) > 1 else 1.0,
            )
        if cousin_vals:
            update_norm_stats(
                "COUSIN",
                mean=statistics.mean(cousin_vals),
                std=statistics.stdev(cousin_vals) if len(cousin_vals) > 1 else 1.0,
            )
        print("[RewardCalc] 校准完成。")
