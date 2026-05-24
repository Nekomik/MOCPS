# Scripts

本目录集中放置 MOCPS 项目的训练、评估和辅助脚本，避免项目根目录过于混乱。

## 目录说明

```text
scripts/training/      # 预训练、微调、专家迭代和 RL 训练相关脚本
scripts/evaluation/    # benchmark、消融实验、时间统计和商业工具复算脚本
scripts/species/       # 单物种数据准备、推理和训练脚本
scripts/utils/         # 数据准备、预测和奖励计算等通用工具
```

## 运行方式

建议始终在项目根目录运行脚本，例如：

```bash
python scripts/training/finetune.py --help
python scripts/evaluation/run_benchmark.py --help
```

多数脚本依赖本地 `data/`、`artifacts/` 和模型 checkpoint。由于这些文件通常较大，默认不提交到 GitHub。
