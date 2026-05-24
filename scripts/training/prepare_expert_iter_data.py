"""
prepare_expert_iter_data.py
将 homo_sapiens_icr_2000.csv 转换为训练用的 .jsonl 格式
用法：python prepare_expert_iter_data.py
"""

import json
import pandas as pd
from pathlib import Path
from CodonTransformer.CodonUtils import ORGANISM2ID

# ── 配置（不需要修改）──────────────────────────────────────────────
ICR_CSV   = "artifacts/results/homo_sapiens_icr_2000.csv"
OUT_JSONL = "data/finetune_json/expert_iter_homo_sapiens.jsonl"
ORGANISM  = "Homo sapiens"
# ─────────────────────────────────────────────────────────────────

def dna_to_codon_tokens(protein: str, dna: str) -> str:
    """
    把蛋白质+DNA转成CodonTransformer的token格式
    例如: M_ATG A_GCC L_CTG ... *_TAA
    """
    protein = str(protein).strip().upper().replace("*", "_")
    if not protein.endswith("_"):
        protein += "_"
    dna = str(dna).strip().upper().replace(" ", "").replace("\n", "")

    codons  = [dna[i:i+3] for i in range(0, len(dna), 3)]
    aa_list = list(protein)

    if len(codons) != len(aa_list):
        raise ValueError(f"长度不匹配: aa={len(aa_list)}, codons={len(codons)}")

    return " ".join(f"{aa}_{codon}" for aa, codon in zip(aa_list, codons))


def main():
    print(f"读取: {ICR_CSV}")
    df = pd.read_csv(ICR_CSV)
    print(f"  共 {len(df)} 条")

    organism_id = ORGANISM2ID[ORGANISM]
    print(f"  {ORGANISM} → organism_id = {organism_id}")

    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)

    success, skip = 0, 0
    with open(OUT_JSONL, "w") as f:
        for idx, row in df.iterrows():
            try:
                protein = str(row["protein"]).strip()
                dna     = str(row["icr_dna"]).strip()

                if not protein or not dna or protein == "nan" or dna == "nan":
                    skip += 1
                    continue

                codons_str = dna_to_codon_tokens(protein, dna)
                f.write(json.dumps({"codons": codons_str, "organism": organism_id}) + "\n")
                success += 1

                if success % 500 == 0:
                    print(f"  已处理 {success} 条...")

            except Exception as e:
                skip += 1
                if skip <= 3:
                    print(f"  ⚠️ 第{idx}条跳过: {e}")

    print(f"\n✅ 完成! 成功={success}, 跳过={skip}")
    print(f"  输出文件: {OUT_JSONL}")


if __name__ == "__main__":
    main()
