"""Query/key subspace intervention."""

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
    parser = argparse.ArgumentParser(description='Query/key subspace intervention.')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--qk-dims', type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()



import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import minimize
from scipy.special import expit

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


OUTPUT_ROOT = os.environ.get("HEAD_GEOMETRY_ROOT", "./outputs")
BASE_DIR = os.environ.get("QK_INTERVENTION_DIR", os.path.join(OUTPUT_ROOT, "wikitext", "qk_intervention"))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "run_checkpoints")
FINAL_DIR = os.path.join(BASE_DIR, "final")


def initialize_output_directories():
    for path in [BASE_DIR, CACHE_DIR, RESULT_DIR, CHECKPOINT_DIR, FINAL_DIR]:
        os.makedirs(path, exist_ok=True)


PHASE = "confirmatory"
SEEDS = list(range(10))
F_VALUES = [7]
EVALUATE_TEST = True


DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TEXT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

D_INPUT = 384

MAX_TRAIN_SENTENCES = 50_000
MAX_VALIDATION_SENTENCES = 5_000
MAX_TEST_SENTENCES = 5_000

MIN_WORDS = 8
MAX_WORDS = 40


WORD_DROP_PROBABILITY = 0.10
MIN_QUERY_WORDS = 6
AUGMENTATION_SEED = 20260712

METADATA_CSV = os.path.join(
    CACHE_DIR,
    "wikitext103_unique_value_metadata.csv",
)

PREPARED_DATA_FILE = os.path.join(
    CACHE_DIR,
    "wikitext103_key_query_minilm.pt",
)


N_VALUE_CLASSES = 128


SUCCESS_THRESHOLD = 0.85


GLOBAL_CHANCE_ACCURACY = 1.0 / N_VALUE_CLASSES


D_MODEL = 512
N_LAYERS = 6
D_FF = 4 * D_MODEL

FIXED_N_HEADS = 8
VALUE_HEAD_DIM = D_MODEL // FIXED_N_HEADS
QK_MAX_DIM = 128
ACTIVE_QK_DIMS = [16, 32, 64, 128]


GPU_ASSIGNMENTS = {
    0: [16, 64],
    1: [32, 128],
}


BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512


VALIDATION_BATCHES = 8
TEST_BATCHES = 8

MAX_STEPS = 20_000
WARMUP_STEPS = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LOG_EVERY = 1_000
CHECKPOINT_EVERY = 1_000
GRADIENT_CLIP_NORM = 1.0


EARLY_STOP_ENABLED = False


VALIDATION_EPISODE_SEED_BASE = 310_000
TEST_EPISODE_SEED_BASE = 910_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state()
    return payload


def _as_cpu_byte_tensor(state) -> torch.Tensor:
    """Normalize legacy/checkpoint RNG states for current PyTorch versions."""
    if isinstance(state, torch.Tensor):
        tensor = state.detach().to(device="cpu", dtype=torch.uint8)
    else:
        tensor = torch.as_tensor(state, dtype=torch.uint8, device="cpu")
    return tensor.contiguous()


def restore_rng_state(payload: dict | None) -> None:
    """Restore Python, NumPy, CPU-Torch, and CUDA RNG states."""
    if not payload:
        return

    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(_as_cpu_byte_tensor(payload["torch_cpu"]))

    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state(
            _as_cpu_byte_tensor(payload["torch_cuda"])
        )


def atomic_pickle_save(payload, path: str) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_torch_save(payload, path: str) -> None:
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def is_valid_sentence(text: str) -> bool:
    if not text or text.startswith("="):
        return False
    word_count = len(text.split())
    return MIN_WORDS <= word_count <= MAX_WORDS


def stable_text_seed(text: str, split_name: str) -> int:
    message = f"{AUGMENTATION_SEED}|{split_name}|{text}".encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def make_query_view(text: str, split_name: str) -> str:
    """
    Deterministic 10% word-drop view.

    The original sentence is encoded as the key.
    This perturbed real-language sentence is encoded as the query.
    """
    words = text.split()

    if len(words) <= MIN_QUERY_WORDS:
        return text

    rng = np.random.default_rng(stable_text_seed(text, split_name))
    keep_mask = rng.random(len(words)) >= WORD_DROP_PROBABILITY


    if int(keep_mask.sum()) < MIN_QUERY_WORDS:
        priority = rng.permutation(len(words))[:MIN_QUERY_WORDS]
        keep_mask[:] = False
        keep_mask[priority] = True

    query = " ".join(word for word, keep in zip(words, keep_mask) if keep)
    return query if query.strip() else text


def collect_clean_split(
    dataset_split,
    split_name: str,
    maximum_sentences: int,
    globally_seen_texts: set[str],
) -> list[dict]:
    rows = []

    for record in dataset_split:
        text = clean_text(record["text"])

        if not is_valid_sentence(text):
            continue


        if text in globally_seen_texts:
            continue

        globally_seen_texts.add(text)

        rows.append({
            "split": split_name,
            "key_text": text,
            "query_text": make_query_view(text, split_name),
        })

        if len(rows) >= maximum_sentences:
            break

    return rows


def prepare_data() -> None:
    if os.path.exists(PREPARED_DATA_FILE):
        print(f"[CACHE] Prepared data exists:\n{PREPARED_DATA_FILE}", flush=True)
        return

    print("PREPARING WIKITEXT KEY/QUERY EMBEDDINGS", flush=True)

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    globally_seen_texts: set[str] = set()

    train_rows = collect_clean_split(
        dataset["train"], "train", MAX_TRAIN_SENTENCES, globally_seen_texts
    )
    validation_rows = collect_clean_split(
        dataset["validation"],
        "validation",
        MAX_VALIDATION_SENTENCES,
        globally_seen_texts,
    )
    test_rows = collect_clean_split(
        dataset["test"], "test", MAX_TEST_SENTENCES, globally_seen_texts
    )

    rows = train_rows + validation_rows + test_rows
    metadata = pd.DataFrame(rows)
    metadata.insert(0, "sentence_id", np.arange(len(metadata), dtype=np.int64))
    metadata.to_csv(METADATA_CSV, index=False)

    print(f"Train sentences:      {len(train_rows):,}", flush=True)
    print(f"Validation sentences: {len(validation_rows):,}", flush=True)
    print(f"Test sentences:       {len(test_rows):,}", flush=True)

    for split_name, split_rows in [
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ]:
        if len(split_rows) < max(F_VALUES):
            raise RuntimeError(
                f"{split_name} has only {len(split_rows)} usable sentences, "
                f"but max(F_VALUES)={max(F_VALUES)}."
            )

    encoder = SentenceTransformer(TEXT_ENCODER_NAME, device="cuda:0")

    key_texts = metadata["key_text"].tolist()
    query_texts = metadata["query_text"].tolist()

    print("[ENCODER] Encoding original key sentences...", flush=True)
    key_embeddings = encoder.encode(
        key_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).detach().cpu().to(torch.float16).contiguous()

    print("[ENCODER] Encoding perturbed query sentences...", flush=True)
    query_embeddings = encoder.encode(
        query_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).detach().cpu().to(torch.float16).contiguous()

    if key_embeddings.shape != query_embeddings.shape:
        raise RuntimeError(
            f"Key/query embedding shape mismatch: "
            f"{key_embeddings.shape} vs {query_embeddings.shape}"
        )

    if key_embeddings.shape[1] != D_INPUT:
        raise RuntimeError(
            f"Expected D_INPUT={D_INPUT}, found {key_embeddings.shape[1]}."
        )

    split_indices = {}
    for split_name in ["train", "validation", "test"]:
        split_indices[split_name] = torch.tensor(
            metadata.index[metadata["split"] == split_name].to_numpy(),
            dtype=torch.long,
        )

    payload = {
        "key_embeddings": key_embeddings,
        "query_embeddings": query_embeddings,
        "train_indices": split_indices["train"],
        "validation_indices": split_indices["validation"],
        "test_indices": split_indices["test"],
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "encoder_name": TEXT_ENCODER_NAME,
        "word_drop_probability": WORD_DROP_PROBABILITY,
        "metadata_csv": os.path.basename(METADATA_CSV),
    }

    atomic_torch_save(payload, PREPARED_DATA_FILE)
    print(f"[SAVED]\n{PREPARED_DATA_FILE}", flush=True)

    del encoder, key_embeddings, query_embeddings, payload, dataset
    torch.cuda.empty_cache()
    gc.collect()


class MaskedQKMultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, active_qk_dim: int):
        super().__init__()
        if active_qk_dim not in ACTIVE_QK_DIMS:
            raise ValueError(f"Unsupported active_qk_dim={active_qk_dim}.")
        if not 1 <= active_qk_dim <= QK_MAX_DIM:
            raise ValueError("active_qk_dim must lie in [1, QK_MAX_DIM].")

        self.d_model = d_model
        self.n_heads = FIXED_N_HEADS
        self.value_head_dim = VALUE_HEAD_DIM
        self.qk_max_dim = QK_MAX_DIM
        self.active_qk_dim = int(active_qk_dim)

        self.q_projection = nn.Linear(
            d_model, self.n_heads * self.qk_max_dim, bias=False
        )
        self.k_projection = nn.Linear(
            d_model, self.n_heads * self.qk_max_dim, bias=False
        )
        self.v_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        q = self.q_projection(x).view(
            batch_size, sequence_length, self.n_heads, self.qk_max_dim
        ).transpose(1, 2)
        k = self.k_projection(x).view(
            batch_size, sequence_length, self.n_heads, self.qk_max_dim
        ).transpose(1, 2)
        v = self.v_projection(x).view(
            batch_size, sequence_length, self.n_heads, self.value_head_dim
        ).transpose(1, 2)


        q_active = q[..., : self.active_qk_dim]
        k_active = k[..., : self.active_qk_dim]

        scores = torch.matmul(
            q_active, k_active.transpose(-2, -1)
        ) / math.sqrt(float(self.active_qk_dim))

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, v)

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, active_qk_dim: int, d_ff: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = MaskedQKMultiHeadSelfAttention(
            d_model=d_model,
            active_qk_dim=active_qk_dim,
        )
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class RealTextUniqueValueTransformer(nn.Module):
    def __init__(self, active_qk_dim: int):
        super().__init__()
        self.n_heads = FIXED_N_HEADS
        self.value_head_dim = VALUE_HEAD_DIM
        self.active_qk_dim = int(active_qk_dim)
        self.qk_max_dim = QK_MAX_DIM

        self.real_text_projection = nn.Linear(D_INPUT, D_MODEL, bias=False)
        self.value_embeddings = nn.Embedding(N_VALUE_CLASSES, D_MODEL)
        maximum_sequence_length = 2 * max(F_VALUES) + 1
        self.position_embeddings = nn.Embedding(
            maximum_sequence_length, D_MODEL
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(D_MODEL, active_qk_dim, D_FF)
                for _ in range(N_LAYERS)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(
            D_MODEL, N_VALUE_CLASSES, bias=False
        )
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

    def forward(
        self,
        key_embeddings: torch.Tensor,
        value_ids: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        projected_keys = self.real_text_projection(key_embeddings)
        projected_queries = self.real_text_projection(query_embeddings)
        value_vectors = self.value_embeddings(value_ids)

        batch_size, n_associations, _ = projected_keys.shape
        sequence_length = 2 * n_associations + 1
        sequence = torch.empty(
            batch_size,
            sequence_length,
            D_MODEL,
            dtype=projected_keys.dtype,
            device=projected_keys.device,
        )
        sequence[:, 0 : 2 * n_associations : 2, :] = projected_keys
        sequence[:, 1 : 2 * n_associations : 2, :] = value_vectors
        sequence[:, -1, :] = projected_queries

        positions = torch.arange(sequence_length, device=sequence.device)
        sequence = sequence + self.position_embeddings(positions)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.final_layer_norm(sequence)
        return self.classifier(sequence[:, -1, :])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reuse_wikitext_capacity_cache_if_available() -> None:
    if os.path.exists(PREPARED_DATA_FILE):
        return

    candidates = [
        os.environ.get("WIKITEXT_CAPACITY_PREPARED_DATA_FILE", ""),
        os.path.join(
            os.environ.get("WIKITEXT_CAPACITY_DIR", os.path.join(OUTPUT_ROOT, "wikitext", "capacity")),
            "cache", "wikitext103_key_query_minilm.pt",
        ),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            shutil.copy2(candidate, PREPARED_DATA_FILE)
            print(f"[CACHE REUSE]\n{candidate}\n->\n{PREPARED_DATA_FILE}", flush=True)
            return


def sample_unique_rows(
    population_size: int,
    batch_size: int,
    sample_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if sample_size > population_size:
        raise ValueError("sample_size exceeds population_size.")
    samples = torch.randint(
        0, population_size, (batch_size, sample_size),
        device=device, generator=generator,
    )
    while True:
        sorted_samples = samples.sort(dim=1).values
        duplicate_rows = (
            sorted_samples[:, 1:] == sorted_samples[:, :-1]
        ).any(dim=1)
        n_bad = int(duplicate_rows.sum().item())
        if n_bad == 0:
            return samples
        samples[duplicate_rows] = torch.randint(
            0, population_size, (n_bad, sample_size),
            device=device, generator=generator,
        )


def generate_episode_batch(
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    pool_indices: torch.Tensor,
    n_associations: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
):
    sampled_pool_positions = sample_unique_rows(
        pool_indices.numel(), batch_size, n_associations, device, generator
    )
    sentence_indices = pool_indices[sampled_pool_positions]
    keys = key_bank[sentence_indices]
    query_views = query_bank[sentence_indices]
    value_ids = sample_unique_rows(
        N_VALUE_CLASSES, batch_size, n_associations, device, generator
    )
    query_positions = torch.randint(
        0, n_associations, (batch_size,),
        device=device, generator=generator,
    )
    rows = torch.arange(batch_size, device=device)
    queries = query_views[rows, query_positions]
    targets = value_ids[rows, query_positions]
    return keys, value_ids, queries, targets


def learning_rate_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(MAX_STEPS - WARMUP_STEPS, 1)
    progress = min(max(progress, 0.0), 1.0)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_fixed_episodes(
    model: nn.Module,
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    pool_indices: torch.Tensor,
    n_associations: int,
    batch_size: int,
    number_of_batches: int,
    episode_seed: int,
    device: torch.device,
    return_predictions: bool = False,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(episode_seed))
    total_correct = 0
    total_examples = 0
    predictions_saved, targets_saved = [], []

    for _ in range(number_of_batches):
        keys, value_ids, queries, targets = generate_episode_batch(
            key_bank, query_bank, pool_indices, n_associations,
            batch_size, device, generator
        )
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = model(keys, value_ids, queries)
        predictions = logits.argmax(dim=-1)
        total_correct += int((predictions == targets).sum().item())
        total_examples += int(targets.numel())
        if return_predictions:
            predictions_saved.append(predictions.cpu().to(torch.uint8))
            targets_saved.append(targets.cpu().to(torch.uint8))

    accuracy = total_correct / total_examples
    if not return_predictions:
        return accuracy
    return (
        accuracy,
        torch.cat(predictions_saved).numpy(),
        torch.cat(targets_saved).numpy(),
    )


def present_value_baseline(n_associations: int) -> float:
    return 1.0 / n_associations


def normalized_binding_score(accuracy: float, n_associations: int) -> float:
    baseline = present_value_baseline(n_associations)
    return (accuracy - baseline) / (1.0 - baseline)


def run_checkpoint_path(active_qk_dim: int, n_associations: int, seed: int) -> str:
    return os.path.join(
        CHECKPOINT_DIR,
        f"{PHASE}_fixedheads{FIXED_N_HEADS}_qk{active_qk_dim}_F{n_associations}_seed{seed}.pt",
    )


def save_run_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scaler,
    step: int,
    active_qk_dim: int,
    n_associations: int,
    seed: int,
) -> None:
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": int(step),
            "active_qk_dim": int(active_qk_dim),
            "n_heads": FIXED_N_HEADS,
            "value_head_dim": VALUE_HEAD_DIM,
            "qk_max_dim": QK_MAX_DIM,
            "F": int(n_associations),
            "seed": int(seed),
            "phase": PHASE,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def train_one_run(
    active_qk_dim: int,
    n_associations: int,
    seed: int,
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    device: torch.device,
) -> dict:
    set_seed(seed)
    model = RealTextUniqueValueTransformer(active_qk_dim).to(device)
    parameter_count = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint_path = run_checkpoint_path(
        active_qk_dim, n_associations, seed
    )
    starting_step = 1

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if int(checkpoint["active_qk_dim"]) != int(active_qk_dim):
            raise RuntimeError("Checkpoint intervention mismatch.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        starting_step = int(checkpoint["step"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"[RESUME RUN] qk={active_qk_dim}, F={n_associations}, "
            f"seed={seed}, starting_step={starting_step}",
            flush=True,
        )

    run_start = time.time()
    final_step = starting_step - 1
    last_loss = float("nan")

    for step in range(starting_step, MAX_STEPS + 1):
        model.train()
        current_lr = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        keys, value_ids, queries, targets = generate_episode_batch(
            key_bank, query_bank, train_indices, n_associations,
            BATCH_SIZE, device, None
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = model(keys, value_ids, queries)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(loss.detach().item())
        final_step = step

        if step % LOG_EVERY == 0 or step == MAX_STEPS:
            validation_accuracy = evaluate_fixed_episodes(
                model, key_bank, query_bank, validation_indices,
                n_associations, VALIDATION_BATCH_SIZE,
                VALIDATION_BATCHES,
                VALIDATION_EPISODE_SEED_BASE + 1000 * n_associations,
                device, False,
            )
            print(
                f"heads={FIXED_N_HEADS} | value_dim={VALUE_HEAD_DIM} | "
                f"active_qk={active_qk_dim:>3} | F={n_associations} | "
                f"seed={seed:>2} | step={step:>5}/{MAX_STEPS} | "
                f"loss={last_loss:.5f} | val_acc={validation_accuracy:.4f} | "
                f"binding={normalized_binding_score(validation_accuracy, n_associations):.4f} | "
                f"lr={current_lr:.2e} | time={(time.time()-run_start)/60:.1f} min",
                flush=True,
            )

        if step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS:
            save_run_checkpoint(
                checkpoint_path, model, optimizer, scaler, step,
                active_qk_dim, n_associations, seed,
            )

    validation_accuracy = evaluate_fixed_episodes(
        model, key_bank, query_bank, validation_indices,
        n_associations, VALIDATION_BATCH_SIZE, VALIDATION_BATCHES,
        VALIDATION_EPISODE_SEED_BASE + 1000 * n_associations,
        device, False,
    )
    test_accuracy, test_predictions, test_targets = evaluate_fixed_episodes(
        model, key_bank, query_bank, test_indices,
        n_associations, TEST_BATCH_SIZE, TEST_BATCHES,
        TEST_EPISODE_SEED_BASE + 1000 * n_associations,
        device, True,
    )

    elapsed_seconds = time.time() - run_start
    result = {
        "phase": PHASE,
        "experiment": "fixed_heads_masked_qk_dimension",
        "dataset": "WikiText-103",
        "dataset_config": DATASET_CONFIG,
        "encoder": TEXT_ENCODER_NAME,
        "F": int(n_associations),
        "seed": int(seed),
        "validation_accuracy": float(validation_accuracy),
        "validation_binding_score": float(
            normalized_binding_score(validation_accuracy, n_associations)
        ),
        "test_accuracy": float(test_accuracy),
        "test_binding_score": float(
            normalized_binding_score(test_accuracy, n_associations)
        ),
        "primary_accuracy": float(test_accuracy),
        "success": bool(test_accuracy >= SUCCESS_THRESHOLD),
        "present_value_baseline": float(present_value_baseline(n_associations)),
        "global_chance_accuracy": float(GLOBAL_CHANCE_ACCURACY),
        "d_input": D_INPUT,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": FIXED_N_HEADS,
        "value_head_dim": VALUE_HEAD_DIM,
        "qk_max_dim": QK_MAX_DIM,
        "active_qk_dim": int(active_qk_dim),
        "d_ff": D_FF,
        "parameter_count": int(parameter_count),
        "batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "final_step": int(final_step),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "elapsed_seconds": float(elapsed_seconds),
        "test_predictions_uint8": test_predictions,
        "test_targets_uint8": test_targets,
    }

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return result


def result_paths(active_qk_dim: int):
    stem = (
        f"{PHASE}_fixedheads{FIXED_N_HEADS}_"
        f"valuedim{VALUE_HEAD_DIM}_qkmax{QK_MAX_DIM}_activeqk{active_qk_dim}"
    )
    return (
        os.path.join(RESULT_DIR, stem + ".pkl"),
        os.path.join(RESULT_DIR, stem + ".csv"),
        os.path.join(RESULT_DIR, stem + "_config.json"),
    )


def export_csv(results: dict, csv_path: str) -> pd.DataFrame:
    rows = []
    for f_value, seeds in sorted(results.items()):
        for seed, result in sorted(seeds.items()):
            rows.append({
                key: value for key, value in result.items()
                if key not in {"test_predictions_uint8", "test_targets_uint8"}
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    return frame


def run_intervention(
    active_qk_dim: int,
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    device: torch.device,
) -> None:
    result_pkl, result_csv, config_json = result_paths(active_qk_dim)
    temporary_model = RealTextUniqueValueTransformer(active_qk_dim)
    parameter_count = count_parameters(temporary_model)
    del temporary_model

    config = {
        "phase": PHASE,
        "scientific_question": "causal effect of active Q/K subspace dimension",
        "F_values": F_VALUES,
        "seeds": SEEDS,
        "fixed_n_heads": FIXED_N_HEADS,
        "fixed_value_head_dim": VALUE_HEAD_DIM,
        "qk_max_dim": QK_MAX_DIM,
        "active_qk_dim": active_qk_dim,
        "parameter_count": parameter_count,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_steps": MAX_STEPS,
        "success_threshold": SUCCESS_THRESHOLD,
    }
    with open(config_json, "w") as file:
        json.dump(config, file, indent=2)

    if os.path.exists(result_pkl):
        with open(result_pkl, "rb") as file:
            results = pickle.load(file)
        print(f"[RESUME INTERVENTION]\n{result_pkl}", flush=True)
    else:
        results = {}
        atomic_pickle_save(results, result_pkl)

    print(
        f"INTERVENTION | heads={FIXED_N_HEADS} | "
        f"value_dim={VALUE_HEAD_DIM} | active_qk={active_qk_dim} | "
        f"qk_max={QK_MAX_DIM} | parameters={parameter_count:,}",
        flush=True,
    )

    for n_associations in F_VALUES:
        results.setdefault(int(n_associations), {})
        for seed in SEEDS:
            if int(seed) in results[int(n_associations)]:
                print(
                    f"[SKIP COMPLETE] qk={active_qk_dim}, "
                    f"F={n_associations}, seed={seed}",
                    flush=True,
                )
                continue

            print(
                f"TRAINING | active_qk={active_qk_dim} | "
                f"F={n_associations} | seed={seed}",
                flush=True,
            )
            result = train_one_run(
                active_qk_dim, n_associations, seed,
                key_bank, query_bank, train_indices,
                validation_indices, test_indices, device,
            )
            results[int(n_associations)][int(seed)] = result
            atomic_pickle_save(results, result_pkl)
            export_csv(results, result_csv)
            print(
                f"RESULT | active_qk={active_qk_dim} | F={n_associations} | "
                f"seed={seed} | val={result['validation_accuracy']:.4f} | "
                f"test={result['test_accuracy']:.4f} | "
                f"binding={result['test_binding_score']:.4f} | "
                f"success={result['success']} | "
                f"time={result['elapsed_seconds']/60:.1f} min",
                flush=True,
            )

    frame = export_csv(results, result_csv)
    if not frame.empty:
        print(
            frame.groupby("F").agg(
                n_runs=("seed", "count"),
                mean_test_accuracy=("test_accuracy", "mean"),
                std_test_accuracy=("test_accuracy", "std"),
                mean_test_binding=("test_binding_score", "mean"),
                n_success=("success", "sum"),
                p_success=("success", "mean"),
            ).to_string(),
            flush=True,
        )


def worker_main(qk_dims_text: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    qk_dims = [
        int(value.strip()) for value in qk_dims_text.split(",")
        if value.strip()
    ]

    payload = torch.load(PREPARED_DATA_FILE, map_location="cpu", weights_only=False)
    key_bank = payload["key_embeddings"].to(device).contiguous()
    query_bank = payload["query_embeddings"].to(device).contiguous()
    train_indices = payload["train_indices"].to(device)
    validation_indices = payload["validation_indices"].to(device)
    test_indices = payload["test_indices"].to(device)


    for active_qk_dim in qk_dims:
        run_intervention(
            active_qk_dim, key_bank, query_bank,
            train_indices, validation_indices, test_indices, device,
        )
        torch.cuda.empty_cache()
        gc.collect()


def exact_sign_flip_p_value(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    exceedances = 0
    total = 2 ** len(differences)
    for assignment in range(total):
        signs = np.array([
            1.0 if ((assignment >> bit) & 1) else -1.0
            for bit in range(len(differences))
        ])
        if abs(float(np.mean(differences * signs))) >= observed - 1e-15:
            exceedances += 1
    return exceedances / total


def merge_results() -> None:
    csv_files = sorted(glob.glob(os.path.join(RESULT_DIR, "*.csv")))
    frames = [
        pd.read_csv(path) for path in csv_files if os.path.getsize(path) > 0
    ]
    if not frames:
        print("[MERGE] No completed results.", flush=True)
        return

    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["active_qk_dim", "F", "seed"]
    ).reset_index(drop=True)
    all_runs_path = os.path.join(FINAL_DIR, "confirmatory_ALL_RUNS.csv")
    merged.to_csv(all_runs_path, index=False)

    summary = merged.groupby(
        ["n_heads", "value_head_dim", "qk_max_dim", "active_qk_dim", "F"]
    ).agg(
        n_runs=("seed", "count"),
        mean_test_accuracy=("test_accuracy", "mean"),
        std_test_accuracy=("test_accuracy", "std"),
        mean_test_binding=("test_binding_score", "mean"),
        n_success=("success", "sum"),
        p_success=("success", "mean"),
        parameter_count=("parameter_count", "first"),
    ).reset_index()
    summary_path = os.path.join(FINAL_DIR, "confirmatory_POINT_SUMMARY.csv")
    summary.to_csv(summary_path, index=False)


    rng = np.random.default_rng(20260718)
    rows = []
    qk_dims = sorted(merged["active_qk_dim"].unique())
    for low, high in itertools.combinations(qk_dims, 2):
        a = merged[merged["active_qk_dim"] == high][
            ["F", "seed", "test_accuracy"]
        ].rename(columns={"test_accuracy": "high"})
        b = merged[merged["active_qk_dim"] == low][
            ["F", "seed", "test_accuracy"]
        ].rename(columns={"test_accuracy": "low"})
        paired = a.merge(b, on=["F", "seed"])
        differences = (paired["high"] - paired["low"]).to_numpy()
        bootstrap_indices = rng.integers(
            0, len(differences), size=(10000, len(differences))
        )
        bootstrap_means = differences[bootstrap_indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append({
            "active_qk_high": int(high),
            "active_qk_low": int(low),
            "paired_seeds": int(len(differences)),
            "mean_accuracy_difference_high_minus_low": float(differences.mean()),
            "median_accuracy_difference_high_minus_low": float(np.median(differences)),
            "bootstrap_ci_low": float(ci_low),
            "bootstrap_ci_high": float(ci_high),
            "exact_sign_flip_p_value": float(
                exact_sign_flip_p_value(differences)
            ),
            "wins_high": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "losses_high": int(np.sum(differences < 0)),
        })
    paired_frame = pd.DataFrame(rows)
    paired_path = os.path.join(FINAL_DIR, "confirmatory_PAIRED_QK_CONTRASTS.csv")
    paired_frame.to_csv(paired_path, index=False)

    parameter_counts = summary["parameter_count"].unique()
    if len(parameter_counts) != 1:
        raise RuntimeError(
            f"Parameter counts are not matched: {parameter_counts.tolist()}"
        )

    print(summary.to_string(index=False), flush=True)
    print("\nPaired Q/K contrasts:", flush=True)
    print(paired_frame.to_string(index=False), flush=True)
    print(f"\nAll runs:\n{all_runs_path}", flush=True)
    print(f"\nSummary:\n{summary_path}", flush=True)
    print(f"\nPaired contrasts:\n{paired_path}", flush=True)


def package_outputs() -> str:
    parent = os.path.dirname(BASE_DIR)
    archive_base = os.path.join(
        parent, "QK_INTERVENTION_RESULTS"
    )
    archive_path = shutil.make_archive(archive_base, "zip", BASE_DIR)
    print(f"[FINAL ZIP]\n{archive_path}", flush=True)
    return archive_path


def parent_main() -> None:
    print(f"FIXED HEADS: {FIXED_N_HEADS}", flush=True)
    print(f"FIXED VALUE HEAD DIM: {VALUE_HEAD_DIM}", flush=True)
    print(f"QK MAX DIM: {QK_MAX_DIM}", flush=True)
    print(f"ACTIVE QK DIMS: {ACTIVE_QK_DIMS}", flush=True)
    print(f"F VALUES: {F_VALUES}", flush=True)
    print(f"SEEDS: {SEEDS}", flush=True)

    reuse_wikitext_capacity_cache_if_available()
    prepare_data()

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("At least one CUDA GPU is required.")

    if gpu_count >= 2:
        assignments = GPU_ASSIGNMENTS
    else:
        assignments = {0: ACTIVE_QK_DIMS}

    processes = []
    for physical_gpu, qk_dims in assignments.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        qk_text = ",".join(str(value) for value in qk_dims)
        command = [
            sys.executable, os.path.abspath(__file__),
            "--worker", "--qk-dims", qk_text,
        ]
        processes.append((
            physical_gpu,
            subprocess.Popen(command, env=environment),
        ))

    failures = []
    for physical_gpu, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((physical_gpu, return_code))
    if failures:
        raise RuntimeError(f"Worker failures: {failures}")

    merge_results()
    package_outputs()


if __name__ == "__main__":
    initialize_output_directories()
    if arguments.worker:
        if not arguments.qk_dims:
            raise ValueError("--qk-dims is required in worker mode.")
        worker_main(arguments.qk_dims)
    else:
        parent_main()
