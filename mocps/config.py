from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_DIR = Path(os.environ.get('MOCPS_BASE_MODEL_DIR', PROJECT_ROOT / 'CodonTransformer_base'))
ARTIFACTS_DIR = Path(os.environ.get('MOCPS_ARTIFACTS_DIR', PROJECT_ROOT / 'artifacts'))
DATA_DIR = Path(os.environ.get('MOCPS_DATA_DIR', PROJECT_ROOT / 'data'))
CHECKPOINTS_DIR = ARTIFACTS_DIR / 'checkpoints'
RESULTS_DIR = ARTIFACTS_DIR / 'results'

DEFAULT_ORGANISM = 'Homo sapiens'
DEFAULT_HUMAN_CHECKPOINT = Path(
    os.environ.get('MOCPS_HUMAN_CHECKPOINT', CHECKPOINTS_DIR / 'homo_sapiens' / 'finetune.ckpt')
)
DEFAULT_HUMAN_EXPERT_CHECKPOINT = Path(
    os.environ.get('MOCPS_EXPERT_CHECKPOINT', CHECKPOINTS_DIR / 'homo_sapiens_expert_iter' / 'expert_iter.ckpt')
)
DEFAULT_HUMAN_EVAL_CSV = Path(
    os.environ.get(
        'MOCPS_HUMAN_EVAL_CSV',
        RESULTS_DIR / 'all_eval_with_my_finetune' / 'homo_sapiens_all_eval_with_my_finetune.csv',
    )
)
