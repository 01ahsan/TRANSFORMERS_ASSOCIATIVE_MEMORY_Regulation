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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--architectures', type=str, default='')
    return parser.parse_args()
if __name__ == '__main__':
    arguments = parse_arguments()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

def default_base_dir() -> str:
    return os.environ.get('AGNEWS_TRANSFER_DIR', os.path.join(os.environ.get('HEAD_GEOMETRY_ROOT', "./outputs"), 'agnews', 'transfer'))
BASE_DIR = default_base_dir()
CACHE_DIR = os.environ.get('AGNEWS_TRANSFER_CACHE_DIR', os.path.join(BASE_DIR, 'cache'))
RESULT_DIR = os.environ.get('AGNEWS_TRANSFER_RESULT_DIR', os.path.join(BASE_DIR, 'results'))
CHECKPOINT_DIR = os.environ.get('AGNEWS_TRANSFER_CHECKPOINT_DIR', os.path.join(BASE_DIR, 'run_checkpoints'))
FINAL_DIR = os.environ.get('AGNEWS_TRANSFER_FINAL_DIR', os.path.join(BASE_DIR, 'final'))

def initialize_output_directories():
    for path in [BASE_DIR, CACHE_DIR, RESULT_DIR, CHECKPOINT_DIR, FINAL_DIR]:
        os.makedirs(path, exist_ok=True)
PHASE = 'external_corpus_replication'
SEEDS = list(range(200, 230))
F_VALUES = [7]
EVALUATE_TEST = True
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
METADATA_CSV = os.environ.get('AGNEWS_TRANSFER_METADATA_CSV', os.path.join(CACHE_DIR, 'agnews_unique_value_metadata.csv'))
PREPARED_DATA_FILE = os.environ.get('AGNEWS_TRANSFER_PREPARED_DATA_FILE', os.path.join(CACHE_DIR, 'agnews_key_query_minilm.pt'))
N_VALUE_CLASSES = 128
REGIME_SUCCESS_THRESHOLD = 0.5
HIGH_TEST_SUCCESS_THRESHOLD = 0.8
SUCCESS_THRESHOLD = HIGH_TEST_SUCCESS_THRESHOLD
GLOBAL_CHANCE_ACCURACY = 1.0 / N_VALUE_CLASSES
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

def reuse_wikitext_capacity_cache_if_available() -> None:
    if os.path.exists(PREPARED_DATA_FILE):
        pass

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
    result = {'phase': PHASE, 'experiment': 'confirmatory_probability_of_success_replication', 'architecture_id': architecture_id, 'dataset': 'AG News', 'dataset_config': DATASET_CONFIG, 'encoder': TEXT_ENCODER_NAME, 'F': int(n_associations), 'seed': int(seed), 'validation_accuracy': float(validation_accuracy), 'validation_binding_score': float(normalized_binding_score(validation_accuracy, n_associations)), 'test_accuracy': float(test_accuracy), 'test_binding_score': float(normalized_binding_score(test_accuracy, n_associations)), 'primary_accuracy': float(test_accuracy), 'maximum_logged_validation_accuracy': float(max((row['validation_accuracy'] for row in learning_curve_rows))), 'entered_regime_050': bool(max((row['validation_accuracy'] for row in learning_curve_rows)) >= REGIME_SUCCESS_THRESHOLD), 'high_test_success_080': bool(test_accuracy >= HIGH_TEST_SUCCESS_THRESHOLD), 'success': bool(max((row['validation_accuracy'] for row in learning_curve_rows)) >= REGIME_SUCCESS_THRESHOLD), 'present_value_baseline': float(present_value_baseline(n_associations)), 'global_chance_accuracy': float(GLOBAL_CHANCE_ACCURACY), 'd_input': D_INPUT, 'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': int(architecture['n_heads']), 'head_dim': int(architecture['head_dim']), 'inner_dim': int(architecture['inner_dim']), 'd_ff': int(architecture['d_ff']), 'parameter_count': int(parameter_count), 'batch_size': BATCH_SIZE, 'max_steps': MAX_STEPS, 'final_step': int(final_step), 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'elapsed_seconds': float(elapsed_seconds), 'final_gradient_norm_before_clipping': float(last_gradient_norm), **{f'final_{key}': value for key, value in final_diagnostics.items()}, 'learning_curve_csv': curve_path, 'test_predictions_uint8': test_predictions, 'test_targets_uint8': test_targets}
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return result

def result_paths(architecture_id: str):
    stem = f'{PHASE}_{architecture_id}_F7'
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
    config = {'phase': PHASE, 'scientific_question': 'confirmatory probability-of-success effect of fully active head width at eight attention heads', 'F_values': F_VALUES, 'seeds': SEEDS, **architecture, 'parameter_count': parameter_count, 'd_model': D_MODEL, 'n_layers': N_LAYERS, 'max_steps': MAX_STEPS, 'primary_endpoint': 'maximum_logged_validation_accuracy >= 0.50', 'regime_success_threshold': REGIME_SUCCESS_THRESHOLD, 'secondary_endpoint': 'test_accuracy >= 0.80', 'high_test_success_threshold': HIGH_TEST_SUCCESS_THRESHOLD}
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
            print(f"RESULT | architecture={architecture_id} | heads={architecture['n_heads']} | head_dim={architecture['head_dim']} | F={n_associations} | seed={seed} | val={result['validation_accuracy']:.4f} | test={result['test_accuracy']:.4f} | binding={result['test_binding_score']:.4f} | success={result['success']} | time={result['elapsed_seconds'] / 60:.1f} min", flush=True)
    frame = export_csv(results, result_csv)
    if not frame.empty:
        print(frame.groupby('F').agg(n_runs=('seed', 'count'), mean_test_accuracy=('test_accuracy', 'mean'), std_test_accuracy=('test_accuracy', 'std'), mean_test_binding=('test_binding_score', 'mean'), n_success=('success', 'sum'), p_success=('success', 'mean')).to_string(), flush=True)

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
    all_runs_path = os.path.join(FINAL_DIR, 'external_corpus_replication_ALL_RUNS.csv')
    merged.to_csv(all_runs_path, index=False)
    summary = merged.groupby('architecture_id').agg(n_runs=('seed', 'count'), mean_test_accuracy=('test_accuracy', 'mean'), median_test_accuracy=('test_accuracy', 'median'), std_test_accuracy=('test_accuracy', 'std'), mean_max_validation_accuracy=('maximum_logged_validation_accuracy', 'mean'), n_regime_success=('entered_regime_050', 'sum'), p_regime_success=('entered_regime_050', 'mean'), n_high_test_success=('high_test_success_080', 'sum'), p_high_test_success=('high_test_success_080', 'mean'), mean_test_binding=('test_binding_score', 'mean'), mean_elapsed_minutes=('elapsed_seconds', lambda values: float(values.mean() / 60.0))).reset_index()
    summary_path = os.path.join(FINAL_DIR, 'external_corpus_replication_SUMMARY.csv')
    summary.to_csv(summary_path, index=False)
    wide = merged[merged['architecture_id'] == 'H8_D64'].copy()
    narrow = merged[merged['architecture_id'] == 'H8_D32'].copy()
    paired = wide.merge(narrow, on=['seed', 'F'], suffixes=('_H8_D64', '_H8_D32'), validate='one_to_one').sort_values('seed')
    if len(paired) != len(SEEDS):
        raise RuntimeError(f'Expected {len(SEEDS)} paired seeds, found {len(paired)}.')
    rng = np.random.default_rng(ANALYSIS_RANDOM_SEED)
    wide_success = paired['entered_regime_050_H8_D64'].astype(bool).to_numpy()
    narrow_success = paired['entered_regime_050_H8_D32'].astype(bool).to_numpy()
    risk_difference, rd_low, rd_high = paired_risk_difference_bootstrap(wide_success, narrow_success, rng)
    mcnemar = exact_mcnemar_test(wide_success, narrow_success)
    primary = {'primary_endpoint': 'maximum logged validation accuracy >= 0.50', 'n_paired_seeds': int(len(paired)), 'p_success_H8_D64': float(wide_success.mean()), 'p_success_H8_D32': float(narrow_success.mean()), 'paired_risk_difference': risk_difference, 'bootstrap_ci_low': rd_low, 'bootstrap_ci_high': rd_high, **mcnemar, 'alpha': PRIMARY_ALPHA}
    primary['confirmatory_primary_supported'] = bool(primary['paired_risk_difference'] > 0 and primary['exact_mcnemar_p_value'] < PRIMARY_ALPHA and (primary['bootstrap_ci_low'] > 0))
    primary_frame = pd.DataFrame([primary])
    primary_path = os.path.join(FINAL_DIR, 'external_corpus_replication_PRIMARY_ENDPOINT.csv')
    primary_frame.to_csv(primary_path, index=False)
    continuous_outcomes = ['test_accuracy', 'maximum_logged_validation_accuracy', 'test_binding_score']
    continuous = pd.DataFrame([paired_continuous_contrast(paired, outcome, rng) for outcome in continuous_outcomes])
    continuous_path = os.path.join(FINAL_DIR, 'external_corpus_replication_CONTINUOUS_OUTCOMES.csv')
    continuous.to_csv(continuous_path, index=False)
    wide_high = paired['high_test_success_080_H8_D64'].astype(bool).to_numpy()
    narrow_high = paired['high_test_success_080_H8_D32'].astype(bool).to_numpy()
    high_rd, high_low, high_high = paired_risk_difference_bootstrap(wide_high, narrow_high, rng)
    secondary_binary = {'secondary_endpoint': 'held-out test accuracy >= 0.80', 'n_paired_seeds': int(len(paired)), 'p_success_H8_D64': float(wide_high.mean()), 'p_success_H8_D32': float(narrow_high.mean()), 'paired_risk_difference': high_rd, 'bootstrap_ci_low': high_low, 'bootstrap_ci_high': high_high, **exact_mcnemar_test(wide_high, narrow_high)}
    secondary_binary_path = os.path.join(FINAL_DIR, 'external_corpus_replication_SECONDARY_BINARY.csv')
    pd.DataFrame([secondary_binary]).to_csv(secondary_binary_path, index=False)
    transition_rows = []
    for _, row in merged.iterrows():
        transition_rows.append({'architecture_id': row['architecture_id'], 'seed': int(row['seed']), 'entered_regime_050': bool(row['entered_regime_050']), 'first_transition_step_050': transition_step_from_curve(row['learning_curve_csv'], REGIME_SUCCESS_THRESHOLD), 'maximum_logged_validation_accuracy': float(row['maximum_logged_validation_accuracy']), 'test_accuracy': float(row['test_accuracy'])})
    transitions = pd.DataFrame(transition_rows)
    transition_path = os.path.join(FINAL_DIR, 'external_corpus_replication_TRANSITIONS.csv')
    transitions.to_csv(transition_path, index=False)
    decision = {'primary_supported': bool(primary['confirmatory_primary_supported']), 'interpretation': 'Supported: wider fully active heads at eight heads increase the probability of entering the successful binding regime.' if primary['confirmatory_primary_supported'] else 'Not supported under the predeclared confirmatory rule. Do not launch another seed expansion or architecture grid for this mechanism.'}
    decision_path = os.path.join(FINAL_DIR, 'external_corpus_replication_DECISION.json')
    with open(decision_path, 'w', encoding='utf-8') as file:
        json.dump({'primary': primary, 'secondary_binary': secondary_binary, **decision}, file, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(primary_frame.to_string(index=False), flush=True)
    print(continuous.to_string(index=False), flush=True)
    print(pd.DataFrame([secondary_binary]).to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)

def package_outputs() -> str:
    archive_base = os.path.join(os.path.dirname(BASE_DIR), 'AGNEWS_TRANSFER_RESULTS')
    archive_path = shutil.make_archive(archive_base, 'zip', BASE_DIR)
    return archive_path

def parent_main() -> None:
    reuse_wikitext_capacity_cache_if_available()
    prepare_data()
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
