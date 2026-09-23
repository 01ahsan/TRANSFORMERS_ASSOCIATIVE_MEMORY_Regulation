import os
import re
import gc
import sys
import json
import math
import time
import glob
import shutil
import random
import pickle
import hashlib
import argparse
import itertools
import subprocess
import warnings
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--architectures', type=str, default='')
    return parser.parse_args()
if __name__ == '__main__':
    arguments = parse_arguments()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

def artifact_reference(path: str) -> str:
    root = os.path.abspath(os.environ.get('HEAD_GEOMETRY_ROOT', "./outputs"))
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, root)
    except ValueError:
        return os.path.basename(absolute)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return os.path.basename(absolute)
    return os.path.join('outputs', relative).replace(os.sep, '/')

def default_base_dir() -> str:
    return os.environ.get('AGNEWS_CONFIRMATION_DIR', os.path.join(os.environ.get('HEAD_GEOMETRY_ROOT', "./outputs"), 'agnews', 'confirmation'))
BASE_DIR = default_base_dir()
CACHE_DIR = os.environ.get('AGNEWS_CONFIRMATION_CACHE_DIR', os.path.join(BASE_DIR, 'cache'))
RESULT_DIR = os.environ.get('AGNEWS_CONFIRMATION_RESULT_DIR', os.path.join(BASE_DIR, 'results'))
CHECKPOINT_DIR = os.environ.get('AGNEWS_CONFIRMATION_CHECKPOINT_DIR', os.path.join(BASE_DIR, 'run_checkpoints'))
FINAL_DIR = os.environ.get('AGNEWS_CONFIRMATION_FINAL_DIR', os.path.join(BASE_DIR, 'final'))

def initialize_output_directories():
    for path in [BASE_DIR, CACHE_DIR, RESULT_DIR, CHECKPOINT_DIR, FINAL_DIR]:
        os.makedirs(path, exist_ok=True)
PHASE = 'agnews_external_confirmation_F6'
SEEDS = list(range(300, 420))
F_VALUES = [6]
EVALUATE_TEST = True
N_PAIRED_SEEDS = len(SEEDS)
ALLOW_INTERIM_INFERENCE = False
DATASET_NAME = 'fancyzhx/ag_news'
DATASET_CONFIG = None
TEXT_FIELD = 'text'
TEXT_ENCODER_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
D_INPUT = 384
MAX_TRAIN_SENTENCES = 50000
MAX_VALIDATION_SENTENCES = 5000
MAX_TEST_SENTENCES = 5000
MIN_WORDS = 8
MAX_WORDS = 80
WORD_DROP_PROBABILITY = 0.1
MIN_QUERY_WORDS = 6
AUGMENTATION_SEED = 20260712
VALIDATION_HASH_MODULUS = 10
VALIDATION_HASH_BUCKET = 0
METADATA_CSV = os.environ.get('AGNEWS_CONFIRMATION_METADATA_CSV', os.path.join(CACHE_DIR, 'agnews_unique_value_metadata.csv'))
PREPARED_DATA_FILE = os.environ.get('AGNEWS_CONFIRMATION_PREPARED_DATA_FILE', os.path.join(CACHE_DIR, 'agnews_key_query_minilm.pt'))
N_VALUE_CLASSES = 128
NORMALIZED_BINDING_REGIME_THRESHOLD = 5.0 / 12.0
SELECTED_F = 6
RAW_EQUIVALENT_VALIDATION_THRESHOLD = 1.0 / SELECTED_F + NORMALIZED_BINDING_REGIME_THRESHOLD * (1.0 - 1.0 / SELECTED_F)
REGIME_SUCCESS_THRESHOLD = RAW_EQUIVALENT_VALIDATION_THRESHOLD
HIGH_TEST_SUCCESS_THRESHOLD = 0.8
SUCCESS_THRESHOLD = HIGH_TEST_SUCCESS_THRESHOLD
GLOBAL_CHANCE_ACCURACY = 1.0 / N_VALUE_CLASSES
PLANNING_WIDE_ONLY_PROBABILITY = 0.3
PLANNING_NARROW_ONLY_PROBABILITY = 0.1
PLANNING_ALPHA = 0.05
D_MODEL = 512
N_LAYERS = 6
ARCHITECTURES = [{'architecture_id': 'H8_D32', 'n_heads': 8, 'head_dim': 32}, {'architecture_id': 'H8_D64', 'n_heads': 8, 'head_dim': 64}]
TARGET_INNER_DIM = 16 * 64
BASE_D_FF = 4 * D_MODEL
for architecture in ARCHITECTURES:
    architecture['inner_dim'] = architecture['n_heads'] * architecture['head_dim']
    architecture['d_ff'] = BASE_D_FF + 2 * (TARGET_INNER_DIM - architecture['inner_dim'])
ARCHITECTURE_BY_ID = {architecture['architecture_id']: architecture for architecture in ARCHITECTURES}
GPU_ASSIGNMENTS = {0: ['H8_D32'], 1: ['H8_D64']}
BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
VALIDATION_BATCHES = 8
TEST_BATCHES = 8
MAX_STEPS = 20000
WARMUP_STEPS = 500
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
LOG_EVERY = 1000
CHECKPOINT_EVERY = 1000
DIAGNOSTIC_EVERY = 1000
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_ENABLED = False
VALIDATION_EPISODE_SEED_BASE = 310000
TEST_EPISODE_SEED_BASE = 910000

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def capture_rng_state() -> dict:
    payload = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        payload['torch_cuda'] = torch.cuda.get_rng_state()
    return payload

def _as_cpu_byte_tensor(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        tensor = state.detach().to(device='cpu', dtype=torch.uint8)
    else:
        tensor = torch.as_tensor(state, dtype=torch.uint8, device='cpu')
    return tensor.contiguous()

def restore_rng_state(payload: dict | None) -> None:
    if not payload:
        return
    random.setstate(payload['python'])
    np.random.set_state(payload['numpy'])
    torch.set_rng_state(_as_cpu_byte_tensor(payload['torch_cpu']))
    if torch.cuda.is_available() and 'torch_cuda' in payload:
        torch.cuda.set_rng_state(_as_cpu_byte_tensor(payload['torch_cuda']))

def atomic_pickle_save(payload, path: str) -> None:
    temporary = path + '.tmp'
    with open(temporary, 'wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)

def atomic_torch_save(payload, path: str) -> None:
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)

def clean_text(text: str) -> str:
    return re.sub('\\s+', ' ', str(text).strip())

def is_valid_sentence(text: str) -> bool:
    if not text or text.startswith('='):
        return False
    word_count = len(text.split())
    return MIN_WORDS <= word_count <= MAX_WORDS

def stable_text_seed(text: str, split_name: str) -> int:
    message = f'{AUGMENTATION_SEED}|{split_name}|{text}'.encode('utf-8')
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=False)

def make_query_view(text: str, split_name: str) -> str:
    words = text.split()
    if len(words) <= MIN_QUERY_WORDS:
        return text
    rng = np.random.default_rng(stable_text_seed(text, split_name))
    keep_mask = rng.random(len(words)) >= WORD_DROP_PROBABILITY
    if int(keep_mask.sum()) < MIN_QUERY_WORDS:
        priority = rng.permutation(len(words))[:MIN_QUERY_WORDS]
        keep_mask[:] = False
        keep_mask[priority] = True
    query = ' '.join((word for word, keep in zip(words, keep_mask) if keep))
    return query if query.strip() else text

def deterministic_partition(text: str) -> str:
    message = f'AGNEWS_SPLIT|{text}'.encode('utf-8')
    digest = hashlib.sha256(message).digest()
    bucket = int.from_bytes(digest[:8], 'little') % VALIDATION_HASH_MODULUS
    return 'validation' if bucket == VALIDATION_HASH_BUCKET else 'train'

def collect_agnews_train_validation(dataset_split, globally_seen_texts: set[str]) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    for record in dataset_split:
        text = clean_text(record[TEXT_FIELD])
        if not is_valid_sentence(text):
            continue
        if text in globally_seen_texts:
            continue
        split_name = deterministic_partition(text)
        if split_name == 'train' and len(train_rows) >= MAX_TRAIN_SENTENCES:
            continue
        if split_name == 'validation' and len(validation_rows) >= MAX_VALIDATION_SENTENCES:
            continue
        globally_seen_texts.add(text)
        row = {'split': split_name, 'key_text': text, 'query_text': make_query_view(text, split_name)}
        if split_name == 'train':
            train_rows.append(row)
        else:
            validation_rows.append(row)
        if len(train_rows) >= MAX_TRAIN_SENTENCES and len(validation_rows) >= MAX_VALIDATION_SENTENCES:
            break
    return (train_rows, validation_rows)

def collect_agnews_test(dataset_split, globally_seen_texts: set[str]) -> list[dict]:
    test_rows: list[dict] = []
    for record in dataset_split:
        text = clean_text(record[TEXT_FIELD])
        if not is_valid_sentence(text):
            continue
        if text in globally_seen_texts:
            continue
        globally_seen_texts.add(text)
        test_rows.append({'split': 'test', 'key_text': text, 'query_text': make_query_view(text, 'test')})
        if len(test_rows) >= MAX_TEST_SENTENCES:
            break
    return test_rows

def prepare_data() -> None:
    if os.path.exists(PREPARED_DATA_FILE):
        return
    if DATASET_CONFIG is None:
        dataset = load_dataset(DATASET_NAME)
    else:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    if 'train' not in dataset or 'test' not in dataset:
        raise RuntimeError(f'Expected official train/test splits, found: {list(dataset.keys())}')
    globally_seen_texts: set[str] = set()
    train_rows, validation_rows = collect_agnews_train_validation(dataset['train'], globally_seen_texts)
    test_rows = collect_agnews_test(dataset['test'], globally_seen_texts)
    rows = train_rows + validation_rows + test_rows
    metadata = pd.DataFrame(rows)
    metadata.insert(0, 'sentence_id', np.arange(len(metadata), dtype=np.int64))
    metadata.to_csv(METADATA_CSV, index=False)
    print(f'Train examples:       {len(train_rows):,}', flush=True)
    print(f'Validation examples:  {len(validation_rows):,}', flush=True)
    print(f'Test examples:        {len(test_rows):,}', flush=True)
    required_counts = {'train': MAX_TRAIN_SENTENCES, 'validation': MAX_VALIDATION_SENTENCES, 'test': MAX_TEST_SENTENCES}
    actual_counts = {'train': len(train_rows), 'validation': len(validation_rows), 'test': len(test_rows)}
    for split_name, required in required_counts.items():
        actual = actual_counts[split_name]
        if actual < required:
            raise RuntimeError(f'{split_name} produced only {actual:,} usable examples; the frozen design requires {required:,}. Do not continue with a silently smaller bank.')
    if metadata['key_text'].duplicated().any():
        raise RuntimeError('Exact text duplicates remain in the prepared bank.')
    encoder = SentenceTransformer(TEXT_ENCODER_NAME, device='cuda:0')
    key_texts = metadata['key_text'].tolist()
    query_texts = metadata['query_text'].tolist()
    key_embeddings = encoder.encode(key_texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).detach().cpu().to(torch.float16).contiguous()
    query_embeddings = encoder.encode(query_texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).detach().cpu().to(torch.float16).contiguous()
    if key_embeddings.shape != query_embeddings.shape:
        raise RuntimeError(f'Key/query embedding shape mismatch: {key_embeddings.shape} vs {query_embeddings.shape}')
    if key_embeddings.shape[1] != D_INPUT:
        raise RuntimeError(f'Expected D_INPUT={D_INPUT}, found {key_embeddings.shape[1]}.')
    split_indices = {}
    for split_name in ['train', 'validation', 'test']:
        split_indices[split_name] = torch.tensor(metadata.index[metadata['split'] == split_name].to_numpy(), dtype=torch.long)
    payload = {'key_embeddings': key_embeddings, 'query_embeddings': query_embeddings, 'train_indices': split_indices['train'], 'validation_indices': split_indices['validation'], 'test_indices': split_indices['test'], 'dataset_name': DATASET_NAME, 'dataset_config': DATASET_CONFIG, 'text_field': TEXT_FIELD, 'encoder_name': TEXT_ENCODER_NAME, 'word_drop_probability': WORD_DROP_PROBABILITY, 'validation_partition': {'method': 'sha256_hash_bucket', 'modulus': VALIDATION_HASH_MODULUS, 'bucket': VALIDATION_HASH_BUCKET}, 'metadata_csv': METADATA_CSV}
    atomic_torch_save(payload, PREPARED_DATA_FILE)
    del encoder, key_embeddings, query_embeddings, payload, dataset
    torch.cuda.empty_cache()
    gc.collect()

class FullyActiveMultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.n_heads * self.head_dim
        self.q_projection = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k_projection = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v_projection = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.output_projection = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.last_diagnostics: dict[str, float] = {}

    def forward(self, x: torch.Tensor, collect_diagnostics: bool=False) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        q = self.q_projection(x).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(x).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_projection(x).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).triu(diagonal=1)
        logits = logits.masked_fill(causal_mask, float('-inf'))
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, v)
        if collect_diagnostics:
            with torch.no_grad():
                finite_logits = logits[torch.isfinite(logits)]
                entropy = -(attention.clamp_min(1e-09) * attention.clamp_min(1e-09).log()).sum(dim=-1)
                self.last_diagnostics = {'q_rms': float(q.float().pow(2).mean().sqrt().item()), 'k_rms': float(k.float().pow(2).mean().sqrt().item()), 'v_rms': float(v.float().pow(2).mean().sqrt().item()), 'attention_logit_std': float(finite_logits.float().std().item()) if finite_logits.numel() > 1 else 0.0, 'attention_entropy': float(entropy.float().mean().item()), 'attention_max_probability': float(attention.float().amax(dim=-1).mean().item())}
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.inner_dim)
        return self.output_projection(attended)

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, head_dim: int, d_ff: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = FullyActiveMultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, head_dim=head_dim)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x: torch.Tensor, collect_diagnostics: bool=False) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x), collect_diagnostics=collect_diagnostics)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class RealTextUniqueValueTransformer(nn.Module):

    def __init__(self, architecture: dict):
        super().__init__()
        self.architecture_id = str(architecture['architecture_id'])
        self.n_heads = int(architecture['n_heads'])
        self.head_dim = int(architecture['head_dim'])
        self.inner_dim = int(architecture['inner_dim'])
        self.d_ff = int(architecture['d_ff'])
        self.real_text_projection = nn.Linear(D_INPUT, D_MODEL, bias=False)
        self.value_embeddings = nn.Embedding(N_VALUE_CLASSES, D_MODEL)
        maximum_sequence_length = 2 * max(F_VALUES) + 1
        self.position_embeddings = nn.Embedding(maximum_sequence_length, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=D_MODEL, n_heads=self.n_heads, head_dim=self.head_dim, d_ff=self.d_ff) for _ in range(N_LAYERS)])
        self.final_layer_norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(D_MODEL, N_VALUE_CLASSES, bias=False)
        self.classifier.weight = self.value_embeddings.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, key_embeddings: torch.Tensor, value_ids: torch.Tensor, query_embeddings: torch.Tensor, collect_diagnostics: bool=False) -> torch.Tensor:
        projected_keys = self.real_text_projection(key_embeddings)
        projected_queries = self.real_text_projection(query_embeddings)
        value_vectors = self.value_embeddings(value_ids)
        batch_size, n_associations, _ = projected_keys.shape
        sequence_length = 2 * n_associations + 1
        sequence = torch.empty(batch_size, sequence_length, D_MODEL, dtype=projected_keys.dtype, device=projected_keys.device)
        sequence[:, 0:2 * n_associations:2, :] = projected_keys
        sequence[:, 1:2 * n_associations:2, :] = value_vectors
        sequence[:, -1, :] = projected_queries
        positions = torch.arange(sequence_length, device=sequence.device)
        sequence = sequence + self.position_embeddings(positions)
        for block in self.blocks:
            sequence = block(sequence, collect_diagnostics=collect_diagnostics)
        sequence = self.final_layer_norm(sequence)
        return self.classifier(sequence[:, -1, :])

    def aggregate_attention_diagnostics(self) -> dict[str, float]:
        keys = ['q_rms', 'k_rms', 'v_rms', 'attention_logit_std', 'attention_entropy', 'attention_max_probability']
        output: dict[str, float] = {}
        for key in keys:
            values = [block.attention.last_diagnostics[key] for block in self.blocks if key in block.attention.last_diagnostics]
            output[key] = float(np.mean(values)) if values else float('nan')
        return output

def count_parameters(model: nn.Module) -> int:
    return sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

def verify_parameter_matching() -> int:
    counts = {}
    for architecture in ARCHITECTURES:
        model = RealTextUniqueValueTransformer(architecture)
        counts[architecture['architecture_id']] = count_parameters(model)
        del model
    unique_counts = sorted(set(counts.values()))
    for architecture_id, count in counts.items():
        architecture = ARCHITECTURE_BY_ID[architecture_id]
    if len(unique_counts) != 1:
        raise RuntimeError(f'Parameter matching failed: {counts}')
    return unique_counts[0]

def sha256_file(path: str, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def _source_roots() -> list[str]:
    candidates = [os.environ.get('AGNEWS_CAPACITY_SOURCE_DIR', os.environ.get('AGNEWS_CAPACITY_DIR', os.path.join(os.environ.get('HEAD_GEOMETRY_ROOT', "./outputs"), 'agnews', 'capacity'))), os.environ.get('AGNEWS_TRANSFER_SOURCE_DIR', os.environ.get('AGNEWS_TRANSFER_DIR', os.path.join(os.environ.get('HEAD_GEOMETRY_ROOT', "./outputs"), 'agnews', 'transfer'))), os.environ.get('WIKITEXT_CONFIRMATION_SOURCE_DIR', os.environ.get('WIKITEXT_CONFIRMATION_DIR', os.path.join(os.environ.get('HEAD_GEOMETRY_ROOT', "./outputs"), 'wikitext', 'confirmation'))), os.environ.get('HEAD_GEOMETRY_INPUT_ROOT', os.path.join(os.environ.get('HEAD_GEOMETRY_ROOT', './outputs'), 'inputs'))]
    return list(dict.fromkeys((os.path.abspath(path) for path in candidates if path and os.path.exists(path))))

def find_external_file(filename: str, preferred_tokens: tuple[str, ...]=()) -> str | None:
    matches = []
    current_base = os.path.abspath(BASE_DIR)
    for root in _source_roots():
        direct = os.path.join(root, filename)
        if os.path.isfile(direct):
            matches.append(direct)
        for candidate in glob.glob(os.path.join(root, '**', filename), recursive=True):
            absolute = os.path.abspath(candidate)
            if absolute.startswith(current_base + os.sep):
                continue
            if os.path.isfile(absolute):
                matches.append(absolute)
    if not matches:
        return None
    tokens = tuple((token.lower() for token in preferred_tokens))

    def score(path: str):
        lowered = path.replace('\\', '/').lower()
        return (sum((token in lowered for token in tokens)), -len(Path(path).parts))
    return sorted(set(matches), key=score, reverse=True)[0]

def exact_mcnemar_power(n_pairs: int, p_wide_only: float, p_narrow_only: float, alpha: float=0.05) -> float:
    if p_wide_only < 0 or p_narrow_only < 0:
        raise ValueError('Discordant probabilities must be nonnegative.')
    if p_wide_only + p_narrow_only > 1:
        raise ValueError('Discordant probabilities sum above one.')
    p_concordant = 1.0 - p_wide_only - p_narrow_only
    power = 0.0
    for wide_only in range(n_pairs + 1):
        for narrow_only in range(n_pairs - wide_only + 1):
            concordant = n_pairs - wide_only - narrow_only
            discordant = wide_only + narrow_only
            if discordant == 0:
                p_value = 1.0
            else:
                p_value = float(stats.binomtest(wide_only, n=discordant, p=0.5, alternative='two-sided').pvalue)
            if p_value >= alpha:
                continue
            log_probability = math.lgamma(n_pairs + 1) - math.lgamma(wide_only + 1) - math.lgamma(narrow_only + 1) - math.lgamma(concordant + 1)
            for count, probability in [(wide_only, p_wide_only), (narrow_only, p_narrow_only), (concordant, p_concordant)]:
                if count == 0:
                    continue
                if probability <= 0:
                    log_probability = -float('inf')
                    break
                log_probability += count * math.log(probability)
            if np.isfinite(log_probability):
                power += math.exp(log_probability)
    return float(power)

def write_power_audit() -> pd.DataFrame:
    scenarios = [('mapping_point_estimate', 0.3, 0.1), ('smaller_asymmetric_effect', 0.25, 0.1), ('same_risk_difference_less_overlap', 0.25, 0.05), ('weak_effect', 0.2, 0.1), ('moderate_low_discordance', 0.2, 0.05)]
    sample_sizes = [60, 80, 100, 120, 140]
    rows = []
    for scenario, p10, p01 in scenarios:
        for n_pairs in sample_sizes:
            rows.append({'scenario': scenario, 'n_paired_seeds': n_pairs, 'p_wide_only': p10, 'p_narrow_only': p01, 'risk_difference': p10 - p01, 'discordance_probability': p10 + p01, 'exact_two_sided_mcnemar_power': exact_mcnemar_power(n_pairs, p10, p01, PLANNING_ALPHA)})
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_SAMPLE_SIZE_POWER_AUDIT.csv'), index=False)
    return frame

def verify_selection_lock_and_reuse_cache() -> dict:
    selection_source = find_external_file('AGNEWS_CONFIRMATION_FROZEN_LOAD_SELECTION.json', preferred_tokens=('agnews/capacity', 'capacity'))
    if selection_source is None:
        raise FileNotFoundError('AG News capacity mapping load-selection JSON was not found. Attach the AG News capacity mapping output directory or set AGNEWS_CAPACITY_SOURCE_DIR.')
    with open(selection_source, 'r', encoding='utf-8') as file:
        selection = json.load(file)
    selected_f = int(selection['selected_F_for_agnews_confirmation'])
    selected_threshold = float(selection['normalized_binding_threshold'])
    mapping_seeds = {int(seed) for seed in selection['mapping_seeds']}
    if selected_f != SELECTED_F or F_VALUES != [selected_f]:
        raise RuntimeError(f'Frozen AG News capacity mapping load is F={selected_f}, but AG News confirmation is configured for F_VALUES={F_VALUES}.')
    if not math.isclose(selected_threshold, NORMALIZED_BINDING_REGIME_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError('Frozen normalized-binding threshold does not match AG News confirmation.')
    overlap = sorted(mapping_seeds & set(SEEDS))
    if overlap:
        raise RuntimeError(f'AG News confirmation seeds overlap AG News capacity mapping mapping seeds: {overlap}')
    copied_selection = os.path.join(FINAL_DIR, 'FROZEN_AGNEWS_CAPACITY_LOAD_SELECTION.json')
    shutil.copy2(selection_source, copied_selection)
    if not os.path.exists(PREPARED_DATA_FILE):
        cache_source = find_external_file('agnews_key_query_minilm.pt', preferred_tokens=('agnews/capacity', 'capacity', 'agnews/transfer'))
        if cache_source is None:
            raise FileNotFoundError('AG News embedding cache not found. Attach AG News capacity mapping or AG News transfer outputs, or set AGNEWS_CAPACITY_SOURCE_DIR.')
        shutil.copy2(cache_source, PREPARED_DATA_FILE)
    else:
        cache_source = PREPARED_DATA_FILE
    metadata_source = find_external_file('agnews_unique_value_metadata.csv', preferred_tokens=('agnews/capacity', 'capacity', 'agnews/transfer'))
    if metadata_source is not None and (not os.path.exists(METADATA_CSV)):
        shutil.copy2(metadata_source, METADATA_CSV)
    payload = torch.load(PREPARED_DATA_FILE, map_location='cpu', weights_only=False)
    required = {'key_embeddings', 'query_embeddings', 'train_indices', 'validation_indices', 'test_indices', 'dataset_name', 'encoder_name'}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise RuntimeError(f'Cache is missing required fields: {missing}')
    if payload['dataset_name'] != DATASET_NAME:
        raise RuntimeError(f"Expected dataset {DATASET_NAME}, found {payload['dataset_name']}.")
    if payload['encoder_name'] != TEXT_ENCODER_NAME:
        raise RuntimeError('Encoder mismatch between frozen cache and AG News confirmation.')
    if tuple(payload['key_embeddings'].shape) != (60000, D_INPUT):
        raise RuntimeError(f"Unexpected key-bank shape: {tuple(payload['key_embeddings'].shape)}")
    if payload['key_embeddings'].shape != payload['query_embeddings'].shape:
        raise RuntimeError('Key/query bank shape mismatch.')
    source_audit = {'selection_source': artifact_reference(selection_source), 'selection_sha256': sha256_file(selection_source), 'selected_F': selected_f, 'normalized_binding_threshold': selected_threshold, 'raw_equivalent_validation_threshold': RAW_EQUIVALENT_VALIDATION_THRESHOLD, 'mapping_seeds': sorted(mapping_seeds), 'confirmatory_seeds': SEEDS, 'seed_overlap': overlap, 'cache_source': artifact_reference(cache_source), 'cache_destination': artifact_reference(PREPARED_DATA_FILE), 'cache_sha256': sha256_file(PREPARED_DATA_FILE), 'dataset_name': payload['dataset_name'], 'encoder_name': payload['encoder_name'], 'key_bank_shape': list(payload['key_embeddings'].shape), 'train_size': int(len(payload['train_indices'])), 'validation_size': int(len(payload['validation_indices'])), 'test_size': int(len(payload['test_indices']))}
    with open(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_SOURCE_LOCK_AUDIT.json'), 'w', encoding='utf-8') as file:
        json.dump(source_audit, file, indent=2)
    return source_audit

def write_preregistration(source_audit: dict) -> dict:
    planning_power = exact_mcnemar_power(N_PAIRED_SEEDS, PLANNING_WIDE_ONLY_PROBABILITY, PLANNING_NARROW_ONLY_PROBABILITY, PLANNING_ALPHA)
    preregistration = {'study': 'agnews_confirmation', 'phase': PHASE, 'scientific_question': 'At the independently selected AG News transition load F=6, does H8_D64 increase successful associative-regime entry relative to parameter-matched H8_D32?', 'architectures': ARCHITECTURES, 'selected_F': SELECTED_F, 'selection_lock_sha256': source_audit['selection_sha256'], 'cache_sha256': source_audit['cache_sha256'], 'seeds': SEEDS, 'n_paired_seeds': N_PAIRED_SEEDS, 'seed_overlap_with_mapping': source_audit['seed_overlap'], 'max_steps': MAX_STEPS, 'primary_endpoint': 'maximum normalized validation binding >= 5/12', 'normalized_binding_formula': '(accuracy - 1/F) / (1 - 1/F)', 'normalized_binding_threshold': NORMALIZED_BINDING_REGIME_THRESHOLD, 'raw_equivalent_validation_threshold': RAW_EQUIVALENT_VALIDATION_THRESHOLD, 'primary_alpha': PLANNING_ALPHA, 'primary_test': 'exact two-sided paired McNemar test', 'primary_effect': 'paired risk difference P(success|H8_D64) - P(success|H8_D32)', 'primary_decision_rule': ['paired risk difference > 0', 'exact two-sided McNemar p < 0.05', 'paired bootstrap 95% CI lower bound > 0'], 'bootstrap_replicates': 50000, 'no_interim_inference': True, 'no_optional_stopping': True, 'planning_probabilities': {'wide_only': PLANNING_WIDE_ONLY_PROBABILITY, 'narrow_only': PLANNING_NARROW_ONLY_PROBABILITY}, 'planned_exact_power': planning_power, 'secondary_endpoints': ['held-out test normalized binding >= 5/12', 'maximum normalized validation binding', 'held-out test normalized binding', 'held-out test accuracy', 'maximum logged validation accuracy', 'held-out test accuracy >= 0.80', 'first validation-regime transition step']}
    with open(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_PREREGISTRATION.json'), 'w', encoding='utf-8') as file:
        json.dump(preregistration, file, indent=2)
    return preregistration

def sample_unique_rows(population_size: int, batch_size: int, sample_size: int, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    if sample_size > population_size:
        raise ValueError('sample_size exceeds population_size.')
    samples = torch.randint(0, population_size, (batch_size, sample_size), device=device, generator=generator)
    while True:
        sorted_samples = samples.sort(dim=1).values
        duplicate_rows = (sorted_samples[:, 1:] == sorted_samples[:, :-1]).any(dim=1)
        number_bad = int(duplicate_rows.sum().item())
        if number_bad == 0:
            return samples
        samples[duplicate_rows] = torch.randint(0, population_size, (number_bad, sample_size), device=device, generator=generator)

def generate_episode_batch(key_bank: torch.Tensor, query_bank: torch.Tensor, pool_indices: torch.Tensor, n_associations: int, batch_size: int, device: torch.device, generator: torch.Generator | None=None):
    sampled_pool_positions = sample_unique_rows(pool_indices.numel(), batch_size, n_associations, device, generator)
    sentence_indices = pool_indices[sampled_pool_positions]
    keys = key_bank[sentence_indices]
    query_views = query_bank[sentence_indices]
    value_ids = sample_unique_rows(N_VALUE_CLASSES, batch_size, n_associations, device, generator)
    query_positions = torch.randint(0, n_associations, (batch_size,), device=device, generator=generator)
    rows = torch.arange(batch_size, device=device)
    queries = query_views[rows, query_positions]
    targets = value_ids[rows, query_positions]
    return (keys, value_ids, queries, targets)

def learning_rate_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(MAX_STEPS - WARMUP_STEPS, 1)
    progress = min(max(progress, 0.0), 1.0)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate_fixed_episodes(model: nn.Module, key_bank: torch.Tensor, query_bank: torch.Tensor, pool_indices: torch.Tensor, n_associations: int, batch_size: int, number_of_batches: int, episode_seed: int, device: torch.device, return_predictions: bool=False):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(episode_seed))
    total_correct = 0
    total_examples = 0
    saved_predictions = []
    saved_targets = []
    for _ in range(number_of_batches):
        keys, value_ids, queries, targets = generate_episode_batch(key_bank=key_bank, query_bank=query_bank, pool_indices=pool_indices, n_associations=n_associations, batch_size=batch_size, device=device, generator=generator)
        with torch.autocast('cuda', dtype=torch.float16, enabled=True):
            logits = model(keys, value_ids, queries)
        predictions = logits.argmax(dim=-1)
        total_correct += int((predictions == targets).sum().item())
        total_examples += int(targets.numel())
        if return_predictions:
            saved_predictions.append(predictions.cpu().to(torch.uint8))
            saved_targets.append(targets.cpu().to(torch.uint8))
    accuracy = total_correct / total_examples
    if not return_predictions:
        return accuracy
    return (accuracy, torch.cat(saved_predictions).numpy(), torch.cat(saved_targets).numpy())

@torch.no_grad()
def collect_diagnostics(model: RealTextUniqueValueTransformer, key_bank: torch.Tensor, query_bank: torch.Tensor, validation_indices: torch.Tensor, n_associations: int, device: torch.device, episode_seed: int) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(episode_seed))
    keys, value_ids, queries, _ = generate_episode_batch(key_bank=key_bank, query_bank=query_bank, pool_indices=validation_indices, n_associations=n_associations, batch_size=min(VALIDATION_BATCH_SIZE, 256), device=device, generator=generator)
    with torch.autocast('cuda', dtype=torch.float16, enabled=True):
        model(keys, value_ids, queries, collect_diagnostics=True)
    return model.aggregate_attention_diagnostics()

def present_value_baseline(n_associations: int) -> float:
    return 1.0 / n_associations

def normalized_binding_score(accuracy: float, n_associations: int) -> float:
    baseline = present_value_baseline(n_associations)
    return (accuracy - baseline) / (1.0 - baseline)

def run_checkpoint_path(architecture_id: str, n_associations: int, seed: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f'{PHASE}_{architecture_id}_F{n_associations}_seed{seed}.pt')

def learning_curve_path(architecture_id: str, n_associations: int, seed: int) -> str:
    directory = os.path.join(RESULT_DIR, 'learning_curves')
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f'{PHASE}_{architecture_id}_F{n_associations}_seed{seed}.csv')

def save_run_checkpoint(path: str, model: nn.Module, optimizer, scaler, step: int, architecture: dict, n_associations: int, seed: int, learning_curve_rows: list[dict]) -> None:
    atomic_torch_save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scaler_state': scaler.state_dict(), 'step': int(step), 'architecture': architecture, 'F': int(n_associations), 'seed': int(seed), 'phase': PHASE, 'rng_state': capture_rng_state(), 'learning_curve_rows': learning_curve_rows}, path)

def train_one_run(architecture: dict, n_associations: int, seed: int, key_bank: torch.Tensor, query_bank: torch.Tensor, train_indices: torch.Tensor, validation_indices: torch.Tensor, test_indices: torch.Tensor, device: torch.device) -> dict:
    architecture_id = str(architecture['architecture_id'])
    set_seed(seed)
    model = RealTextUniqueValueTransformer(architecture).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    checkpoint_path = run_checkpoint_path(architecture_id, n_associations, seed)
    curve_path = learning_curve_path(architecture_id, n_associations, seed)
    starting_step = 1
    learning_curve_rows: list[dict] = []
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_architecture = checkpoint['architecture']
        if checkpoint_architecture['architecture_id'] != architecture_id:
            raise RuntimeError('Checkpoint architecture mismatch.')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        starting_step = int(checkpoint['step']) + 1
        learning_curve_rows = list(checkpoint.get('learning_curve_rows', []))
        restore_rng_state(checkpoint.get('rng_state'))
    run_start = time.time()
    final_step = starting_step - 1
    last_loss = float('nan')
    last_gradient_norm = float('nan')
    for step in range(starting_step, MAX_STEPS + 1):
        model.train()
        current_lr = learning_rate_at(step)
        for group in optimizer.param_groups:
            group['lr'] = current_lr
        keys, value_ids, queries, targets = generate_episode_batch(key_bank=key_bank, query_bank=query_bank, pool_indices=train_indices, n_associations=n_associations, batch_size=BATCH_SIZE, device=device, generator=None)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=True):
            logits = model(keys, value_ids, queries)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        last_gradient_norm = float(gradient_norm.item())
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.detach().item())
        final_step = step
        should_log = step % LOG_EVERY == 0 or step == MAX_STEPS
        if should_log:
            validation_accuracy = evaluate_fixed_episodes(model=model, key_bank=key_bank, query_bank=query_bank, pool_indices=validation_indices, n_associations=n_associations, batch_size=VALIDATION_BATCH_SIZE, number_of_batches=VALIDATION_BATCHES, episode_seed=VALIDATION_EPISODE_SEED_BASE + 1000 * n_associations, device=device, return_predictions=False)
            diagnostics = collect_diagnostics(model=model, key_bank=key_bank, query_bank=query_bank, validation_indices=validation_indices, n_associations=n_associations, device=device, episode_seed=VALIDATION_EPISODE_SEED_BASE + 900000 + 1000 * n_associations)
            row = {'architecture_id': architecture_id, 'n_heads': int(architecture['n_heads']), 'head_dim': int(architecture['head_dim']), 'inner_dim': int(architecture['inner_dim']), 'd_ff': int(architecture['d_ff']), 'F': int(n_associations), 'seed': int(seed), 'step': int(step), 'training_loss': last_loss, 'validation_accuracy': float(validation_accuracy), 'validation_binding_score': float(normalized_binding_score(validation_accuracy, n_associations)), 'learning_rate': float(current_lr), 'gradient_norm_before_clipping': last_gradient_norm, 'elapsed_minutes': float((time.time() - run_start) / 60.0), **diagnostics}
            learning_curve_rows.append(row)
            pd.DataFrame(learning_curve_rows).to_csv(curve_path, index=False)
            print(f"architecture={architecture_id} | heads={architecture['n_heads']:>2} | head_dim={architecture['head_dim']:>2} | inner={architecture['inner_dim']:>4} | d_ff={architecture['d_ff']:>4} | F={n_associations} | seed={seed:>2} | step={step:>5}/{MAX_STEPS} | loss={last_loss:.5f} | val_acc={validation_accuracy:.4f} | binding={row['validation_binding_score']:.4f} | grad={last_gradient_norm:.4f} | entropy={diagnostics['attention_entropy']:.4f} | logit_std={diagnostics['attention_logit_std']:.4f} | time={row['elapsed_minutes']:.1f} min", flush=True)
        if step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS:
            save_run_checkpoint(path=checkpoint_path, model=model, optimizer=optimizer, scaler=scaler, step=step, architecture=architecture, n_associations=n_associations, seed=seed, learning_curve_rows=learning_curve_rows)
    validation_accuracy = evaluate_fixed_episodes(model=model, key_bank=key_bank, query_bank=query_bank, pool_indices=validation_indices, n_associations=n_associations, batch_size=VALIDATION_BATCH_SIZE, number_of_batches=VALIDATION_BATCHES, episode_seed=VALIDATION_EPISODE_SEED_BASE + 1000 * n_associations, device=device, return_predictions=False)
    test_accuracy, test_predictions, test_targets = evaluate_fixed_episodes(model=model, key_bank=key_bank, query_bank=query_bank, pool_indices=test_indices, n_associations=n_associations, batch_size=TEST_BATCH_SIZE, number_of_batches=TEST_BATCHES, episode_seed=TEST_EPISODE_SEED_BASE + 1000 * n_associations, device=device, return_predictions=True)
    final_diagnostics = collect_diagnostics(model=model, key_bank=key_bank, query_bank=query_bank, validation_indices=validation_indices, n_associations=n_associations, device=device, episode_seed=VALIDATION_EPISODE_SEED_BASE + 950000 + 1000 * n_associations)
    elapsed_seconds = time.time() - run_start
    maximum_logged_validation_accuracy = float(max((row['validation_accuracy'] for row in learning_curve_rows)))
    maximum_logged_validation_binding = float(normalized_binding_score(maximum_logged_validation_accuracy, n_associations))
    test_binding_score = float(normalized_binding_score(test_accuracy, n_associations))
    result = {'phase': PHASE, 'experiment': 'preregistered_agnews_external_confirmation_F6', 'architecture_id': architecture_id, 'dataset': 'AG News', 'dataset_config': DATASET_CONFIG, 'encoder': TEXT_ENCODER_NAME, 'F': int(n_associations), 'seed': int(seed), 'validation_accuracy': float(validation_accuracy), 'validation_binding_score': float(normalized_binding_score(validation_accuracy, n_associations)), 'test_accuracy': float(test_accuracy), 'test_binding_score': test_binding_score, 'primary_accuracy': float(test_accuracy), 'maximum_logged_validation_accuracy': maximum_logged_validation_accuracy, 'maximum_logged_validation_binding': maximum_logged_validation_binding, 'entered_normalized_binding_regime': bool(maximum_logged_validation_binding >= NORMALIZED_BINDING_REGIME_THRESHOLD), 'test_normalized_binding_regime': bool(test_binding_score >= NORMALIZED_BINDING_REGIME_THRESHOLD), 'high_test_success_080': bool(test_accuracy >= HIGH_TEST_SUCCESS_THRESHOLD), 'success': bool(maximum_logged_validation_binding >= NORMALIZED_BINDING_REGIME_THRESHOLD), 'normalized_binding_regime_threshold': float(NORMALIZED_BINDING_REGIME_THRESHOLD), 'raw_equivalent_validation_threshold': float(RAW_EQUIVALENT_VALIDATION_THRESHOLD), 'present_value_baseline': float(present_value_baseline(n_associations)), 'global_chance_accuracy': float(GLOBAL_CHANCE_ACCURACY), 'd_input': D_INPUT, 'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': int(architecture['n_heads']), 'head_dim': int(architecture['head_dim']), 'inner_dim': int(architecture['inner_dim']), 'd_ff': int(architecture['d_ff']), 'parameter_count': int(parameter_count), 'batch_size': BATCH_SIZE, 'max_steps': MAX_STEPS, 'final_step': int(final_step), 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'elapsed_seconds': float(elapsed_seconds), 'final_gradient_norm_before_clipping': float(last_gradient_norm), **{f'final_{key}': value for key, value in final_diagnostics.items()}, 'learning_curve_csv': curve_path, 'test_predictions_uint8': test_predictions, 'test_targets_uint8': test_targets}
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return result

def result_paths(architecture_id: str):
    stem = f'{PHASE}_{architecture_id}_F6'
    return (os.path.join(RESULT_DIR, stem + '.pkl'), os.path.join(RESULT_DIR, stem + '.csv'), os.path.join(RESULT_DIR, stem + '_config.json'))

def export_csv(results: dict, csv_path: str) -> pd.DataFrame:
    rows = []
    for f_value, seeds in sorted(results.items()):
        for seed, result in sorted(seeds.items()):
            rows.append({key: value for key, value in result.items() if key not in {'test_predictions_uint8', 'test_targets_uint8'}})
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    return frame

def run_architecture(architecture: dict, key_bank: torch.Tensor, query_bank: torch.Tensor, train_indices: torch.Tensor, validation_indices: torch.Tensor, test_indices: torch.Tensor, device: torch.device) -> None:
    architecture_id = architecture['architecture_id']
    result_pkl, result_csv, config_json = result_paths(architecture_id)
    temporary_model = RealTextUniqueValueTransformer(architecture)
    parameter_count = count_parameters(temporary_model)
    del temporary_model
    config = {'phase': PHASE, 'scientific_question': 'independent AG News confirmation of fully active head-width effect at the AG News capacity mapping selected transition load', 'F_values': F_VALUES, 'seeds': SEEDS, **architecture, 'parameter_count': parameter_count, 'd_model': D_MODEL, 'n_layers': N_LAYERS, 'max_steps': MAX_STEPS, 'primary_endpoint': 'maximum_logged_validation_binding >= 5/12', 'normalized_regime_success_threshold': NORMALIZED_BINDING_REGIME_THRESHOLD, 'raw_equivalent_validation_threshold': RAW_EQUIVALENT_VALIDATION_THRESHOLD, 'secondary_endpoint': 'held-out test normalized binding >= 5/12', 'high_test_success_threshold': HIGH_TEST_SUCCESS_THRESHOLD}
    with open(config_json, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=2)
    if os.path.exists(result_pkl):
        with open(result_pkl, 'rb') as file:
            results = pickle.load(file)
    else:
        results = {}
        atomic_pickle_save(results, result_pkl)
    for n_associations in F_VALUES:
        results.setdefault(int(n_associations), {})
        for seed in SEEDS:
            if int(seed) in results[int(n_associations)]:
                continue
            print(f'TRAINING | architecture={architecture_id} | F={n_associations} | seed={seed}', flush=True)
            result = train_one_run(architecture=architecture, n_associations=n_associations, seed=seed, key_bank=key_bank, query_bank=query_bank, train_indices=train_indices, validation_indices=validation_indices, test_indices=test_indices, device=device)
            results[int(n_associations)][int(seed)] = result
            atomic_pickle_save(results, result_pkl)
            export_csv(results, result_csv)
            print(f"RESULT | architecture={architecture_id} | heads={architecture['n_heads']} | head_dim={architecture['head_dim']} | F={n_associations} | seed={seed} | val={result['validation_accuracy']:.4f} | test={result['test_accuracy']:.4f} | test_binding={result['test_binding_score']:.4f} | max_val_binding={result['maximum_logged_validation_binding']:.4f} | success={result['success']} | time={result['elapsed_seconds'] / 60:.1f} min", flush=True)
    frame = export_csv(results, result_csv)
    if not frame.empty:
        print(frame.groupby('F').agg(n_runs=('seed', 'count'), mean_test_accuracy=('test_accuracy', 'mean'), mean_test_binding=('test_binding_score', 'mean'), mean_max_validation_binding=('maximum_logged_validation_binding', 'mean'), n_success=('entered_normalized_binding_regime', 'sum'), p_success=('entered_normalized_binding_regime', 'mean')).to_string(), flush=True)

def worker_main(architecture_ids_text: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    architecture_ids = [value.strip() for value in architecture_ids_text.split(',') if value.strip()]
    unknown = [architecture_id for architecture_id in architecture_ids if architecture_id not in ARCHITECTURE_BY_ID]
    if unknown:
        raise ValueError(f'Unknown architecture IDs: {unknown}')
    payload = torch.load(PREPARED_DATA_FILE, map_location='cpu', weights_only=False)
    key_bank = payload['key_embeddings'].to(device).contiguous()
    query_bank = payload['query_embeddings'].to(device).contiguous()
    train_indices = payload['train_indices'].to(device)
    validation_indices = payload['validation_indices'].to(device)
    test_indices = payload['test_indices'].to(device)
    for architecture_id in architecture_ids:
        run_architecture(architecture=ARCHITECTURE_BY_ID[architecture_id], key_bank=key_bank, query_bank=query_bank, train_indices=train_indices, validation_indices=validation_indices, test_indices=test_indices, device=device)
        torch.cuda.empty_cache()
        gc.collect()
PRIMARY_ALPHA = 0.05
BOOTSTRAP_REPLICATES = 50000
ANALYSIS_RANDOM_SEED = 20260720

def exact_mcnemar_test(success_wide: np.ndarray, success_narrow: np.ndarray) -> dict:
    success_wide = np.asarray(success_wide, dtype=bool)
    success_narrow = np.asarray(success_narrow, dtype=bool)
    wide_only = int(np.sum(success_wide & ~success_narrow))
    narrow_only = int(np.sum(~success_wide & success_narrow))
    both = int(np.sum(success_wide & success_narrow))
    neither = int(np.sum(~success_wide & ~success_narrow))
    discordant = wide_only + narrow_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(wide_only, n=discordant, p=0.5, alternative='two-sided').pvalue)
    return {'wide_only_success': wide_only, 'narrow_only_success': narrow_only, 'both_success': both, 'neither_success': neither, 'discordant_pairs': discordant, 'exact_mcnemar_p_value': p_value}

def paired_risk_difference_bootstrap(wide: np.ndarray, narrow: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    wide = np.asarray(wide, dtype=float)
    narrow = np.asarray(narrow, dtype=float)
    paired_differences = wide - narrow
    indices = rng.integers(0, len(paired_differences), size=(BOOTSTRAP_REPLICATES, len(paired_differences)))
    bootstrap = paired_differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return (float(paired_differences.mean()), float(low), float(high))

def exact_sign_flip_p_value(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    observed = abs(float(differences.mean()))
    n = len(differences)
    if n <= 20:
        exceedances = 0
        total = 2 ** n
        for assignment in range(total):
            signs = np.array([1.0 if assignment >> bit & 1 else -1.0 for bit in range(n)], dtype=float)
            statistic = abs(float(np.mean(differences * signs)))
            if statistic >= observed - 1e-15:
                exceedances += 1
        return exceedances / total
    rng = np.random.default_rng(ANALYSIS_RANDOM_SEED)
    signs = rng.choice([-1.0, 1.0], size=(200000, n))
    null_statistics = np.abs((signs * differences).mean(axis=1))
    return float((1 + np.sum(null_statistics >= observed - 1e-15)) / (len(null_statistics) + 1))

def paired_continuous_contrast(paired: pd.DataFrame, outcome: str, rng: np.random.Generator) -> dict:
    wide = paired[f'{outcome}_H8_D64'].to_numpy(dtype=float)
    narrow = paired[f'{outcome}_H8_D32'].to_numpy(dtype=float)
    differences = wide - narrow
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {'outcome': outcome, 'paired_seeds': int(len(differences)), 'mean_H8_D64': float(wide.mean()), 'mean_H8_D32': float(narrow.mean()), 'mean_difference_H8_D64_minus_H8_D32': float(differences.mean()), 'median_difference_H8_D64_minus_H8_D32': float(np.median(differences)), 'bootstrap_ci_low': float(low), 'bootstrap_ci_high': float(high), 'exact_sign_flip_p_value': float(exact_sign_flip_p_value(differences)), 'wins_H8_D64': int(np.sum(differences > 0)), 'ties': int(np.sum(differences == 0)), 'losses_H8_D64': int(np.sum(differences < 0))}

def transition_step_from_curve(curve_path: str, threshold: float=REGIME_SUCCESS_THRESHOLD) -> float:
    if not isinstance(curve_path, str) or not os.path.exists(curve_path):
        return float('nan')
    curve = pd.read_csv(curve_path)
    matching = curve[curve['validation_accuracy'] >= threshold]
    if matching.empty:
        return float('nan')
    return float(matching.iloc[0]['step'])

def clopper_pearson_interval(successes: int, total: int, alpha: float=0.05) -> tuple[float, float]:
    if total <= 0:
        return (float('nan'), float('nan'))
    low = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return (low, high)

def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()

def paired_continuous_contrast_extended(paired: pd.DataFrame, outcome: str, rng: np.random.Generator) -> dict:
    result = paired_continuous_contrast(paired, outcome, rng)
    wide = paired[f'{outcome}_H8_D64'].to_numpy(dtype=float)
    narrow = paired[f'{outcome}_H8_D32'].to_numpy(dtype=float)
    differences = wide - narrow
    sd_difference = float(np.std(differences, ddof=1))
    result['sd_paired_difference'] = sd_difference
    result['paired_standardized_mean_difference_dz'] = float(differences.mean() / sd_difference) if sd_difference > 0 else float('nan')
    return result

def binary_endpoint_summary(paired: pd.DataFrame, column: str, endpoint_name: str, rng: np.random.Generator) -> dict:
    wide = paired[f'{column}_H8_D64'].astype(bool).to_numpy()
    narrow = paired[f'{column}_H8_D32'].astype(bool).to_numpy()
    rd, low, high = paired_risk_difference_bootstrap(wide, narrow, rng)
    mcnemar = exact_mcnemar_test(wide, narrow)
    wide_count = int(wide.sum())
    narrow_count = int(narrow.sum())
    wide_ci = clopper_pearson_interval(wide_count, len(wide))
    narrow_ci = clopper_pearson_interval(narrow_count, len(narrow))
    return {'endpoint': endpoint_name, 'n_paired_seeds': int(len(paired)), 'n_success_H8_D64': wide_count, 'p_success_H8_D64': float(wide.mean()), 'p_success_H8_D64_ci_low': wide_ci[0], 'p_success_H8_D64_ci_high': wide_ci[1], 'n_success_H8_D32': narrow_count, 'p_success_H8_D32': float(narrow.mean()), 'p_success_H8_D32_ci_low': narrow_ci[0], 'p_success_H8_D32_ci_high': narrow_ci[1], 'paired_risk_difference': rd, 'bootstrap_ci_low': low, 'bootstrap_ci_high': high, **mcnemar}

def cumulative_transition_frame(transitions: pd.DataFrame) -> pd.DataFrame:
    steps = sorted({int(step) for path in transitions['learning_curve_csv'] if isinstance(path, str) and os.path.exists(path) for step in pd.read_csv(path)['step'].tolist()})
    rows = []
    for architecture_id, group in transitions.groupby('architecture_id'):
        transition_values = group['first_transition_step'].to_numpy(dtype=float)
        for step in steps:
            rows.append({'architecture_id': architecture_id, 'step': int(step), 'cumulative_regime_entry_probability': float(np.mean(np.isfinite(transition_values) & (transition_values <= step)))})
    return pd.DataFrame(rows)

def save_confirmation_figures(merged: pd.DataFrame, paired: pd.DataFrame, primary: dict, cumulative: pd.DataFrame) -> None:
    figure_dir = os.path.join(FINAL_DIR, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    labels = ['H8_D32', 'H8_D64']
    probabilities = [primary['p_success_H8_D32'], primary['p_success_H8_D64']]
    lower = [probabilities[0] - primary['p_success_H8_D32_ci_low'], probabilities[1] - primary['p_success_H8_D64_ci_low']]
    upper = [primary['p_success_H8_D32_ci_high'] - probabilities[0], primary['p_success_H8_D64_ci_high'] - probabilities[1]]
    fig, axis = plt.subplots(figsize=(5.8, 4.4))
    x = np.arange(len(labels))
    axis.bar(x, probabilities)
    axis.errorbar(x, probabilities, yerr=np.vstack([lower, upper]), fmt='none', capsize=5)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel('Probability of normalized regime entry')
    axis.set_title('AG News, F=6, 120 new matched seeds')
    axis.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(os.path.join(figure_dir, f'agnews_confirmation_primary_success_probability.{extension}'), dpi=350 if extension == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    x_narrow = paired['maximum_logged_validation_binding_H8_D32'].to_numpy(dtype=float)
    x_wide = paired['maximum_logged_validation_binding_H8_D64'].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(x_narrow, x_wide, alpha=0.65, s=24)
    minimum = float(min(x_narrow.min(), x_wide.min()))
    maximum = float(max(x_narrow.max(), x_wide.max()))
    axis.plot([minimum, maximum], [minimum, maximum], linestyle='--')
    axis.axhline(NORMALIZED_BINDING_REGIME_THRESHOLD, linestyle=':', linewidth=1.0)
    axis.axvline(NORMALIZED_BINDING_REGIME_THRESHOLD, linestyle=':', linewidth=1.0)
    axis.set_xlabel('H8_D32 maximum validation binding')
    axis.set_ylabel('H8_D64 maximum validation binding')
    axis.set_title('Seed-matched associative outcomes')
    axis.grid(alpha=0.25)
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(os.path.join(figure_dir, f'agnews_confirmation_paired_maximum_binding.{extension}'), dpi=350 if extension == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for architecture_id in ['H8_D32', 'H8_D64']:
        subset = cumulative[cumulative['architecture_id'] == architecture_id]
        axis.step(subset['step'], subset['cumulative_regime_entry_probability'], where='post', label=architecture_id)
    axis.set_xlabel('Training step')
    axis.set_ylabel('Cumulative probability of regime entry')
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(os.path.join(figure_dir, f'agnews_confirmation_cumulative_regime_entry.{extension}'), dpi=350 if extension == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    differences = paired['test_binding_score_H8_D64'].to_numpy(dtype=float) - paired['test_binding_score_H8_D32'].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.hist(differences, bins=20)
    axis.axvline(0.0, linewidth=1.0)
    axis.axvline(float(differences.mean()), linestyle='--', linewidth=1.2, label=f'mean = {differences.mean():.3f}')
    axis.set_xlabel('Paired held-out binding difference (H8_D64 - H8_D32)')
    axis.set_ylabel('Number of seed pairs')
    axis.legend()
    axis.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(os.path.join(figure_dir, f'agnews_confirmation_paired_test_binding_difference.{extension}'), dpi=350 if extension == 'png' else None, bbox_inches='tight')
    plt.close(fig)

def optional_cross_corpus_synthesis(current_primary: dict) -> pd.DataFrame:
    rows = [{'corpus': 'AG News', 'load_F': SELECTED_F, 'n_paired_seeds': current_primary['n_paired_seeds'], 'p_success_H8_D64': current_primary['p_success_H8_D64'], 'p_success_H8_D32': current_primary['p_success_H8_D32'], 'paired_risk_difference': current_primary['paired_risk_difference'], 'ci_low': current_primary['bootstrap_ci_low'], 'ci_high': current_primary['bootstrap_ci_high'], 'exact_mcnemar_p_value': current_primary['exact_mcnemar_p_value'], 'status': 'independent external confirmation'}]
    wikitext_confirmation_primary = find_external_file('confirmatory_replication_PRIMARY_ENDPOINT.csv', preferred_tokens=('wikitext/confirmation', 'probability_success'))
    if wikitext_confirmation_primary is not None:
        old = pd.read_csv(wikitext_confirmation_primary).iloc[0]
        rows.insert(0, {'corpus': 'WikiText-103', 'load_F': 7, 'n_paired_seeds': int(old['n_paired_seeds']), 'p_success_H8_D64': float(old['p_success_H8_D64']), 'p_success_H8_D32': float(old['p_success_H8_D32']), 'paired_risk_difference': float(old['paired_risk_difference']), 'ci_low': float(old['bootstrap_ci_low']), 'ci_high': float(old['bootstrap_ci_high']), 'exact_mcnemar_p_value': float(old['exact_mcnemar_p_value']), 'status': 'independent original-corpus confirmation'})
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_CROSS_CORPUS_SYNTHESIS.csv'), index=False)
    if len(frame) > 1:
        figure_dir = os.path.join(FINAL_DIR, 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        fig, axis = plt.subplots(figsize=(6.2, 3.6))
        y = np.arange(len(frame))
        effects = frame['paired_risk_difference'].to_numpy(dtype=float)
        lower = effects - frame['ci_low'].to_numpy(dtype=float)
        upper = frame['ci_high'].to_numpy(dtype=float) - effects
        axis.errorbar(effects, y, xerr=np.vstack([lower, upper]), fmt='o', capsize=4)
        axis.axvline(0.0, linewidth=1.0)
        axis.set_yticks(y, frame['corpus'])
        axis.set_xlabel('Paired regime-entry risk difference (H8_D64 - H8_D32)')
        axis.grid(axis='x', alpha=0.25)
        fig.tight_layout()
        for extension in ['pdf', 'png']:
            fig.savefig(os.path.join(figure_dir, f'agnews_confirmation_cross_corpus_primary_forest.{extension}'), dpi=350 if extension == 'png' else None, bbox_inches='tight')
        plt.close(fig)
    return frame

def merge_results() -> None:
    csv_files = sorted(glob.glob(os.path.join(RESULT_DIR, '*.csv')))
    frames = [pd.read_csv(path) for path in csv_files if os.path.getsize(path) > 0 and 'learning_curves' not in path]
    if not frames:
        raise RuntimeError('[MERGE] No completed results.')
    merged = pd.concat(frames, ignore_index=True).sort_values(['architecture_id', 'seed']).reset_index(drop=True)
    expected_rows = len(ARCHITECTURES) * len(F_VALUES) * len(SEEDS)
    if len(merged) != expected_rows:
        raise RuntimeError(f'Incomplete result grid: found {len(merged)}, expected {expected_rows}.')
    duplicate_count = int(merged.duplicated(['architecture_id', 'F', 'seed']).sum())
    if duplicate_count:
        raise RuntimeError(f'Duplicate architecture/F/seed rows: {duplicate_count}')
    if set(merged['seed'].astype(int)) != set(SEEDS):
        raise RuntimeError('Completed seed set differs from preregistration.')
    if set(merged['F'].astype(int)) != {SELECTED_F}:
        raise RuntimeError('Completed load differs from frozen AG News capacity mapping load.')
    all_runs_path = os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_ALL_RUNS.csv')
    merged.to_csv(all_runs_path, index=False)
    summary_rows = []
    for architecture_id, group in merged.groupby('architecture_id'):
        success_count = int(group['entered_normalized_binding_regime'].sum())
        success_ci = clopper_pearson_interval(success_count, len(group))
        test_success_count = int(group['test_normalized_binding_regime'].sum())
        test_success_ci = clopper_pearson_interval(test_success_count, len(group))
        summary_rows.append({'architecture_id': architecture_id, 'n_runs': int(len(group)), 'mean_test_accuracy': float(group['test_accuracy'].mean()), 'median_test_accuracy': float(group['test_accuracy'].median()), 'std_test_accuracy': float(group['test_accuracy'].std(ddof=1)), 'mean_test_binding': float(group['test_binding_score'].mean()), 'mean_max_validation_accuracy': float(group['maximum_logged_validation_accuracy'].mean()), 'mean_max_validation_binding': float(group['maximum_logged_validation_binding'].mean()), 'n_primary_success': success_count, 'p_primary_success': float(success_count / len(group)), 'p_primary_success_ci_low': success_ci[0], 'p_primary_success_ci_high': success_ci[1], 'n_test_regime_success': test_success_count, 'p_test_regime_success': float(test_success_count / len(group)), 'p_test_regime_success_ci_low': test_success_ci[0], 'p_test_regime_success_ci_high': test_success_ci[1], 'mean_elapsed_minutes': float(group['elapsed_seconds'].mean() / 60.0)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_ARCHITECTURE_SUMMARY.csv'), index=False)
    wide = merged[merged['architecture_id'] == 'H8_D64'].copy()
    narrow = merged[merged['architecture_id'] == 'H8_D32'].copy()
    paired = wide.merge(narrow, on=['seed', 'F'], suffixes=('_H8_D64', '_H8_D32'), validate='one_to_one').sort_values('seed')
    if len(paired) != N_PAIRED_SEEDS:
        raise RuntimeError(f'Expected {N_PAIRED_SEEDS} paired seeds, found {len(paired)}.')
    rng = np.random.default_rng(ANALYSIS_RANDOM_SEED)
    primary = binary_endpoint_summary(paired=paired, column='entered_normalized_binding_regime', endpoint_name=f'maximum normalized validation binding >= 5/12 (raw accuracy >= {RAW_EQUIVALENT_VALIDATION_THRESHOLD:.10f})', rng=rng)
    primary['alpha'] = PRIMARY_ALPHA
    primary['confirmatory_primary_supported'] = bool(primary['paired_risk_difference'] > 0 and primary['exact_mcnemar_p_value'] < PRIMARY_ALPHA and (primary['bootstrap_ci_low'] > 0))
    pd.DataFrame([primary]).to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_PRIMARY_ENDPOINT.csv'), index=False)
    heldout_binary = binary_endpoint_summary(paired=paired, column='test_normalized_binding_regime', endpoint_name='held-out test normalized binding >= 5/12', rng=rng)
    pd.DataFrame([heldout_binary]).to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_HELDOUT_BINARY_ENDPOINT.csv'), index=False)
    stringent_binary = binary_endpoint_summary(paired=paired, column='high_test_success_080', endpoint_name='held-out test accuracy >= 0.80', rng=rng)
    pd.DataFrame([stringent_binary]).to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_STRINGENT_TEST_ENDPOINT.csv'), index=False)
    continuous_outcomes = ['maximum_logged_validation_binding', 'test_binding_score', 'test_accuracy', 'maximum_logged_validation_accuracy']
    continuous_rows = [paired_continuous_contrast_extended(paired, outcome, rng) for outcome in continuous_outcomes]
    adjusted = holm_adjust([row['exact_sign_flip_p_value'] for row in continuous_rows])
    for row, adjusted_p in zip(continuous_rows, adjusted):
        row['holm_adjusted_sign_flip_p_value'] = adjusted_p
    continuous = pd.DataFrame(continuous_rows)
    continuous.to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_CONTINUOUS_OUTCOMES.csv'), index=False)
    transition_rows = []
    for _, row in merged.iterrows():
        transition_rows.append({'architecture_id': row['architecture_id'], 'seed': int(row['seed']), 'entered_normalized_binding_regime': bool(row['entered_normalized_binding_regime']), 'first_transition_step': transition_step_from_curve(row['learning_curve_csv'], RAW_EQUIVALENT_VALIDATION_THRESHOLD), 'maximum_logged_validation_accuracy': float(row['maximum_logged_validation_accuracy']), 'maximum_logged_validation_binding': float(row['maximum_logged_validation_binding']), 'test_accuracy': float(row['test_accuracy']), 'test_binding_score': float(row['test_binding_score']), 'learning_curve_csv': row['learning_curve_csv']})
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_TRANSITIONS.csv'), index=False)
    cumulative = cumulative_transition_frame(transitions)
    cumulative.to_csv(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_CUMULATIVE_REGIME_ENTRY.csv'), index=False)
    save_confirmation_figures(merged, paired, primary, cumulative)
    cross_corpus = optional_cross_corpus_synthesis(primary)
    decision = {'primary_supported': bool(primary['confirmatory_primary_supported']), 'frozen_load': SELECTED_F, 'n_paired_seeds': N_PAIRED_SEEDS, 'interpretation': 'Independent AG News confirmation supported: at the AG News capacity mapping selected transition load, wider fully active heads significantly increase the probability of entering the associative-binding regime under an exactly matched parameter budget.' if primary['confirmatory_primary_supported'] else 'Independent AG News confirmation was not supported under the frozen paired binary rule. Report the result without changing the load, threshold, seed cohort, or sample size.', 'prohibited_post_result_actions': ['Do not extend the seed cohort after inspecting AG News confirmation.', 'Do not change F=6.', 'Do not change the normalized threshold 5/12.', 'Do not replace the exact two-sided McNemar primary test.', 'Do not relabel a secondary endpoint as primary.']}
    with open(os.path.join(FINAL_DIR, 'AGNEWS_CONFIRMATION_DECISION.json'), 'w', encoding='utf-8') as file:
        json.dump({'primary': primary, 'heldout_binary': heldout_binary, 'stringent_binary': stringent_binary, 'decision': decision}, file, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame([primary]).to_string(index=False), flush=True)
    print(pd.DataFrame([heldout_binary]).to_string(index=False), flush=True)
    print(continuous.to_string(index=False), flush=True)
    print(cross_corpus.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)

def package_outputs() -> str:
    archive_base = os.path.join(os.path.dirname(BASE_DIR), 'AGNEWS_CONFIRMATION_RESULTS')
    archive_path = shutil.make_archive(archive_base, 'zip', BASE_DIR)
    return archive_path

def verify_frozen_design_constants() -> None:
    if F_VALUES != [SELECTED_F]:
        raise RuntimeError(f'F_VALUES must remain [{SELECTED_F}], found {F_VALUES}.')
    if len(SEEDS) != 120 or SEEDS != list(range(300, 420)):
        raise RuntimeError('AG News confirmation confirmatory seeds were modified.')
    if ALLOW_INTERIM_INFERENCE:
        raise RuntimeError('Interim inference must remain disabled.')
    expected_raw = 37.0 / 72.0
    if not math.isclose(RAW_EQUIVALENT_VALIDATION_THRESHOLD, expected_raw, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError('Raw equivalent threshold changed.')

def parent_main() -> None:
    verify_frozen_design_constants()
    source_audit = verify_selection_lock_and_reuse_cache()
    prepare_data()
    preregistration = write_preregistration(source_audit)
    power_audit = write_power_audit()
    matched_parameter_count = verify_parameter_matching()
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError('At least one CUDA GPU is required.')
    assignments = GPU_ASSIGNMENTS if gpu_count >= 2 else {0: [architecture['architecture_id'] for architecture in ARCHITECTURES]}
    processes = []
    for physical_gpu, architecture_ids in assignments.items():
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
        architecture_text = ','.join(architecture_ids)
        command = [sys.executable, os.path.abspath(__file__), '--worker', '--architectures', architecture_text]
        processes.append((physical_gpu, subprocess.Popen(command, env=environment)))
    failures = []
    for physical_gpu, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((physical_gpu, return_code))
    if failures:
        raise RuntimeError(f'Worker failures: {failures}')
    merge_results()
    package_outputs()
if __name__ == '__main__':
    initialize_output_directories()
    if arguments.worker:
        if not arguments.architectures:
            raise ValueError('--architectures is required in worker mode.')
        worker_main(arguments.architectures)
    else:
        parent_main()
