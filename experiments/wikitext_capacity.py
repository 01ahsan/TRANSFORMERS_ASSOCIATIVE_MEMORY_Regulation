"""WikiText capacity mapping."""

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
import subprocess
import warnings

def parse_arguments():
    parser = argparse.ArgumentParser(description='WikiText capacity mapping.')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--heads', type=str, default='')
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
BASE_DIR = os.environ.get("WIKITEXT_CAPACITY_DIR", os.path.join(OUTPUT_ROOT, "wikitext", "capacity"))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "run_checkpoints")
FINAL_DIR = os.path.join(BASE_DIR, "final")

def initialize_output_directories():
    for path in [BASE_DIR, CACHE_DIR, RESULT_DIR, CHECKPOINT_DIR, FINAL_DIR]:
        os.makedirs(path, exist_ok=True)


PHASE = "confirmatory"

if PHASE == "pilot":
    SEEDS = [0]
    F_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
    EVALUATE_TEST = False

elif PHASE == "confirmatory":
    SEEDS = list(range(10))


    CONFIRMATORY_F_VALUES = [5, 6, 7, 8]
    F_VALUES = CONFIRMATORY_F_VALUES
    EVALUATE_TEST = True

else:
    raise ValueError(f"Unknown PHASE={PHASE!r}")


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


GPU_ASSIGNMENTS = {
    0: [4, 16],
    1: [8, 32],
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"{d_model=} is not divisible by {n_heads=}.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.view(
                    batch_size,
                    sequence_length,
                    self.n_heads,
                    self.d_head,
                )
                .transpose(1, 2)
                .contiguous()
            )

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )

        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads)

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
    def __init__(self, n_heads: int):
        super().__init__()

        if D_MODEL % n_heads != 0:
            raise ValueError(f"{D_MODEL=} is not divisible by {n_heads=}.")

        self.n_heads = n_heads
        self.d_head = D_MODEL // n_heads

        self.real_text_projection = nn.Linear(D_INPUT, D_MODEL, bias=False)
        self.value_embeddings = nn.Embedding(N_VALUE_CLASSES, D_MODEL)

        maximum_sequence_length = 2 * max(F_VALUES) + 1
        self.position_embeddings = nn.Embedding(
            maximum_sequence_length,
            D_MODEL,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, n_heads, D_FF)
            for _ in range(N_LAYERS)
        ])

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
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def sample_unique_rows(
    population_size: int,
    batch_size: int,
    sample_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """
    Sample `sample_size` unique integers per row without constructing a huge
    [batch_size, population_size] random matrix.
    """
    if sample_size > population_size:
        raise ValueError(
            f"Cannot sample {sample_size} unique values from {population_size}."
        )

    samples = torch.randint(
        0,
        population_size,
        (batch_size, sample_size),
        device=device,
        generator=generator,
    )

    while True:
        sorted_samples = samples.sort(dim=1).values
        duplicate_rows = (
            sorted_samples[:, 1:] == sorted_samples[:, :-1]
        ).any(dim=1)

        number_bad = int(duplicate_rows.sum().item())
        if number_bad == 0:
            return samples

        samples[duplicate_rows] = torch.randint(
            0,
            population_size,
            (number_bad, sample_size),
            device=device,
            generator=generator,
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
        population_size=pool_indices.numel(),
        batch_size=batch_size,
        sample_size=n_associations,
        device=device,
        generator=generator,
    )

    sentence_indices = pool_indices[sampled_pool_positions]

    keys = key_bank[sentence_indices]
    query_views_for_selected_keys = query_bank[sentence_indices]


    value_ids = sample_unique_rows(
        population_size=N_VALUE_CLASSES,
        batch_size=batch_size,
        sample_size=n_associations,
        device=device,
        generator=generator,
    )

    query_positions = torch.randint(
        0,
        n_associations,
        (batch_size,),
        device=device,
        generator=generator,
    )

    row_indices = torch.arange(batch_size, device=device)

    queries = query_views_for_selected_keys[row_indices, query_positions]
    targets = value_ids[row_indices, query_positions]

    return keys, value_ids, queries, targets


def learning_rate_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(WARMUP_STEPS, 1)

    progress = (step - WARMUP_STEPS) / max(
        MAX_STEPS - WARMUP_STEPS,
        1,
    )
    progress = min(max(progress, 0.0), 1.0)

    return LEARNING_RATE * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


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

    saved_predictions = []
    saved_targets = []

    for _ in range(number_of_batches):
        keys, value_ids, queries, targets = generate_episode_batch(
            key_bank=key_bank,
            query_bank=query_bank,
            pool_indices=pool_indices,
            n_associations=n_associations,
            batch_size=batch_size,
            device=device,
            generator=generator,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            logits = model(keys, value_ids, queries)

        predictions = logits.argmax(dim=-1)

        total_correct += (predictions == targets).sum().item()
        total_examples += targets.numel()

        if return_predictions:
            saved_predictions.append(predictions.detach().cpu().to(torch.uint8))
            saved_targets.append(targets.detach().cpu().to(torch.uint8))

    accuracy = total_correct / total_examples

    if not return_predictions:
        return accuracy

    return (
        accuracy,
        torch.cat(saved_predictions).numpy(),
        torch.cat(saved_targets).numpy(),
    )


def present_value_baseline(n_associations: int) -> float:
    return 1.0 / n_associations


def normalized_binding_score(
    accuracy: float,
    n_associations: int,
) -> float:
    baseline = present_value_baseline(n_associations)
    return (accuracy - baseline) / (1.0 - baseline)


def run_checkpoint_path(
    n_heads: int,
    n_associations: int,
    seed: int,
) -> str:
    d_head = D_MODEL // n_heads
    return os.path.join(
        CHECKPOINT_DIR,
        (
            f"{PHASE}_heads{n_heads}_dhead{d_head}_"
            f"F{n_associations}_seed{seed}.pt"
        ),
    )


def save_run_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scaler,
    step: int,
    n_heads: int,
    n_associations: int,
    seed: int,
) -> None:
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": int(step),
            "n_heads": int(n_heads),
            "d_head": int(D_MODEL // n_heads),
            "F": int(n_associations),
            "seed": int(seed),
            "phase": PHASE,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def train_one_run(
    n_heads: int,
    n_associations: int,
    seed: int,
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor | None,
    device: torch.device,
) -> dict:
    set_seed(seed)

    model = RealTextUniqueValueTransformer(n_heads=n_heads).to(device)
    parameter_count = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    checkpoint_path = run_checkpoint_path(
        n_heads=n_heads,
        n_associations=n_associations,
        seed=seed,
    )

    starting_step = 1

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if checkpoint.get("phase") != PHASE:
            raise RuntimeError(
                f"Checkpoint phase={checkpoint.get('phase')} does not match "
                f"current PHASE={PHASE}."
            )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        starting_step = int(checkpoint["step"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))

        print(
            f"[RESUME RUN] heads={n_heads}, F={n_associations}, seed={seed}, "
            f"starting_step={starting_step}",
            flush=True,
        )

    run_start = time.time()
    last_loss = float("nan")
    final_step = starting_step - 1


    for step in range(starting_step, MAX_STEPS + 1):
        model.train()

        current_lr = learning_rate_at(step)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr

        keys, value_ids, queries, targets = generate_episode_batch(
            key_bank=key_bank,
            query_bank=query_bank,
            pool_indices=train_indices,
            n_associations=n_associations,
            batch_size=BATCH_SIZE,
            device=device,
            generator=None,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            logits = model(keys, value_ids, queries)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        last_loss = float(loss.detach().item())
        final_step = step

        if step % LOG_EVERY == 0 or step == MAX_STEPS:
            validation_seed = (
                VALIDATION_EPISODE_SEED_BASE
                + 1000 * n_associations
            )

            validation_accuracy = evaluate_fixed_episodes(
                model=model,
                key_bank=key_bank,
                query_bank=query_bank,
                pool_indices=validation_indices,
                n_associations=n_associations,
                batch_size=VALIDATION_BATCH_SIZE,
                number_of_batches=VALIDATION_BATCHES,
                episode_seed=validation_seed,
                device=device,
                return_predictions=False,
            )

            binding_score = normalized_binding_score(
                validation_accuracy,
                n_associations,
            )

            elapsed_minutes = (time.time() - run_start) / 60.0

            print(
                f"heads={n_heads:>2} | d_head={D_MODEL // n_heads:>3} | "
                f"F={n_associations:>2} | seed={seed:>2} | "
                f"step={step:>5}/{MAX_STEPS} | "
                f"loss={last_loss:.5f} | "
                f"val_acc={validation_accuracy:.4f} | "
                f"1/F={present_value_baseline(n_associations):.4f} | "
                f"binding={binding_score:.4f} | "
                f"lr={current_lr:.2e} | "
                f"time={elapsed_minutes:.1f} min",
                flush=True,
            )

        if step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS:
            save_run_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                n_heads=n_heads,
                n_associations=n_associations,
                seed=seed,
            )

    validation_episode_seed = (
        VALIDATION_EPISODE_SEED_BASE
        + 1000 * n_associations
    )

    final_validation_accuracy = evaluate_fixed_episodes(
        model=model,
        key_bank=key_bank,
        query_bank=query_bank,
        pool_indices=validation_indices,
        n_associations=n_associations,
        batch_size=VALIDATION_BATCH_SIZE,
        number_of_batches=VALIDATION_BATCHES,
        episode_seed=validation_episode_seed,
        device=device,
        return_predictions=False,
    )

    test_accuracy = None
    test_predictions = None
    test_targets = None

    if EVALUATE_TEST:
        if test_indices is None:
            raise RuntimeError("Confirmatory phase requires test indices.")

        test_episode_seed = (
            TEST_EPISODE_SEED_BASE
            + 1000 * n_associations
        )

        (
            test_accuracy,
            test_predictions,
            test_targets,
        ) = evaluate_fixed_episodes(
            model=model,
            key_bank=key_bank,
            query_bank=query_bank,
            pool_indices=test_indices,
            n_associations=n_associations,
            batch_size=TEST_BATCH_SIZE,
            number_of_batches=TEST_BATCHES,
            episode_seed=test_episode_seed,
            device=device,
            return_predictions=True,
        )

    elapsed_seconds = time.time() - run_start

    primary_accuracy = (
        test_accuracy
        if test_accuracy is not None
        else final_validation_accuracy
    )

    result = {
        "phase": PHASE,
        "dataset": "WikiText-103",
        "dataset_config": DATASET_CONFIG,
        "encoder": TEXT_ENCODER_NAME,
        "query_view": f"deterministic_word_drop_{WORD_DROP_PROBABILITY:.2f}",
        "F": int(n_associations),
        "seed": int(seed),
        "validation_accuracy": float(final_validation_accuracy),
        "validation_binding_score": float(
            normalized_binding_score(
                final_validation_accuracy,
                n_associations,
            )
        ),
        "test_accuracy": (
            None if test_accuracy is None else float(test_accuracy)
        ),
        "test_binding_score": (
            None
            if test_accuracy is None
            else float(
                normalized_binding_score(
                    test_accuracy,
                    n_associations,
                )
            )
        ),
        "primary_accuracy": float(primary_accuracy),
        "success": bool(primary_accuracy >= SUCCESS_THRESHOLD),
        "global_chance_accuracy": float(GLOBAL_CHANCE_ACCURACY),
        "present_value_baseline": float(
            present_value_baseline(n_associations)
        ),
        "d_input": D_INPUT,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": int(n_heads),
        "d_head": int(D_MODEL // n_heads),
        "d_ff": D_FF,
        "n_value_classes": N_VALUE_CLASSES,
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


def result_paths(n_heads: int):
    d_head = D_MODEL // n_heads
    stem = (
        f"{PHASE}_realtext_unique_"
        f"dmodel{D_MODEL}_layers{N_LAYERS}_"
        f"heads{n_heads}_dhead{d_head}"
    )

    return (
        os.path.join(RESULT_DIR, stem + ".pkl"),
        os.path.join(RESULT_DIR, stem + ".csv"),
        os.path.join(RESULT_DIR, stem + "_config.json"),
    )


def export_csv(results: dict, csv_path: str) -> pd.DataFrame:
    rows = []

    for f_value, seed_dictionary in sorted(results.items()):
        for seed, result in sorted(seed_dictionary.items()):
            csv_row = {
                key: value
                for key, value in result.items()
                if key not in {
                    "test_predictions_uint8",
                    "test_targets_uint8",
                }
            }
            rows.append(csv_row)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_path, index=False)
    return dataframe


def run_architecture(
    n_heads: int,
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor | None,
    device: torch.device,
) -> None:
    result_pkl, result_csv, config_json = result_paths(n_heads)

    temporary_model = RealTextUniqueValueTransformer(n_heads)
    parameter_count = count_parameters(temporary_model)
    del temporary_model

    config = {
        "phase": PHASE,
        "evaluate_test": EVALUATE_TEST,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "encoder": TEXT_ENCODER_NAME,
        "word_drop_probability": WORD_DROP_PROBABILITY,
        "F_values": F_VALUES,
        "seeds": SEEDS,
        "n_value_classes": N_VALUE_CLASSES,
        "success_threshold": SUCCESS_THRESHOLD,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": n_heads,
        "d_head": D_MODEL // n_heads,
        "d_ff": D_FF,
        "parameter_count": parameter_count,
        "batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stop_enabled": EARLY_STOP_ENABLED,
        "validation_episodes": VALIDATION_BATCH_SIZE * VALIDATION_BATCHES,
        "test_episodes": (
            TEST_BATCH_SIZE * TEST_BATCHES
            if EVALUATE_TEST
            else 0
        ),
    }

    with open(config_json, "w") as file:
        json.dump(config, file, indent=2)

    if os.path.exists(result_pkl):
        with open(result_pkl, "rb") as file:
            results = pickle.load(file)
        print(f"[RESUME ARCHITECTURE]\n{result_pkl}", flush=True)
    else:
        results = {}
        atomic_pickle_save(results, result_pkl)
        print(f"[NEW ARCHITECTURE]\n{result_pkl}", flush=True)

    print(
        f"ARCHITECTURE | heads={n_heads} | "
        f"d_head={D_MODEL // n_heads} | "
        f"parameters={parameter_count:,} | phase={PHASE}",
        flush=True,
    )

    for n_associations in F_VALUES:
        results.setdefault(int(n_associations), {})

        for seed in SEEDS:
            if int(seed) in results[int(n_associations)]:
                print(
                    f"[SKIP COMPLETE] heads={n_heads}, "
                    f"F={n_associations}, seed={seed}",
                    flush=True,
                )
                continue

            print(
                f"TRAINING | heads={n_heads} | "
                f"d_head={D_MODEL // n_heads} | "
                f"F={n_associations} | seed={seed}",
                flush=True,
            )

            result = train_one_run(
                n_heads=n_heads,
                n_associations=n_associations,
                seed=seed,
                key_bank=key_bank,
                query_bank=query_bank,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                device=device,
            )

            results[int(n_associations)][int(seed)] = result
            atomic_pickle_save(results, result_pkl)
            export_csv(results, result_csv)

            print(
                f"RESULT | heads={n_heads} | "
                f"d_head={D_MODEL // n_heads} | "
                f"F={n_associations} | seed={seed} | "
                f"val={result['validation_accuracy']:.4f} | "
                f"test={result['test_accuracy']} | "
                f"binding={result['validation_binding_score']:.4f} | "
                f"success={result['success']} | "
                f"time={result['elapsed_seconds'] / 60:.1f} min",
                flush=True,
            )

    dataframe = export_csv(results, result_csv)

    if not dataframe.empty:
        summary = (
            dataframe.groupby("F")
            .agg(
                n_runs=("seed", "count"),
                mean_validation_accuracy=("validation_accuracy", "mean"),
                mean_validation_binding=("validation_binding_score", "mean"),
                mean_primary_accuracy=("primary_accuracy", "mean"),
                n_success=("success", "sum"),
                p_success=("success", "mean"),
            )
        )

        if EVALUATE_TEST:
            test_summary = (
                dataframe.groupby("F")
                .agg(
                    mean_test_accuracy=("test_accuracy", "mean"),
                    mean_test_binding=("test_binding_score", "mean"),
                )
            )
            summary = summary.join(test_summary)

        print(summary.to_string(), flush=True)


def worker_main(heads_text: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Worker cannot see CUDA.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    head_configurations = [
        int(value.strip())
        for value in heads_text.split(",")
        if value.strip()
    ]

    payload = torch.load(PREPARED_DATA_FILE, map_location="cpu")

    key_bank = (
        payload["key_embeddings"]
        .to(device, non_blocking=True)
        .contiguous()
    )

    query_bank = (
        payload["query_embeddings"]
        .to(device, non_blocking=True)
        .contiguous()
    )

    train_indices = payload["train_indices"].to(device, non_blocking=True)
    validation_indices = payload["validation_indices"].to(
        device,
        non_blocking=True,
    )


    test_indices = None
    if EVALUATE_TEST:
        test_indices = payload["test_indices"].to(
            device,
            non_blocking=True,
        )


    for n_heads in head_configurations:
        run_architecture(
            n_heads=n_heads,
            key_bank=key_bank,
            query_bank=query_bank,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            device=device,
        )
        torch.cuda.empty_cache()
        gc.collect()


def merge_results() -> None:
    csv_files = sorted(
        glob.glob(
            os.path.join(
                RESULT_DIR,
                f"{PHASE}_realtext_unique_*.csv",
            )
        )
    )

    frames = [
        pd.read_csv(path)
        for path in csv_files
        if os.path.getsize(path) > 0
    ]

    if not frames:
        print("[MERGE] No completed CSVs found.", flush=True)
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        ["d_head", "F", "seed"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    all_runs_path = os.path.join(
        FINAL_DIR,
        f"{PHASE}_ALL_RUNS.csv",
    )

    merged.to_csv(all_runs_path, index=False)

    aggregation = {
        "n_runs": ("seed", "count"),
        "mean_validation_accuracy": ("validation_accuracy", "mean"),
        "std_validation_accuracy": ("validation_accuracy", "std"),
        "mean_validation_binding": ("validation_binding_score", "mean"),
        "mean_primary_accuracy": ("primary_accuracy", "mean"),
        "n_success": ("success", "sum"),
        "p_success": ("success", "mean"),
    }

    if EVALUATE_TEST:
        aggregation.update({
            "mean_test_accuracy": ("test_accuracy", "mean"),
            "std_test_accuracy": ("test_accuracy", "std"),
            "mean_test_binding": ("test_binding_score", "mean"),
        })

    summary = (
        merged.groupby(["n_heads", "d_head", "F"])
        .agg(**aggregation)
        .reset_index()
    )

    summary_path = os.path.join(
        FINAL_DIR,
        f"{PHASE}_POINT_SUMMARY.csv",
    )

    summary.to_csv(summary_path, index=False)

    print(f"FINAL MERGED {PHASE.upper()} RESULTS", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nAll runs:\n{all_runs_path}", flush=True)
    print(f"\nPoint summary:\n{summary_path}", flush=True)


def _fit_binomial_capacity(
    f_values: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
):
    f_values = np.asarray(f_values, dtype=np.float64)
    successes = np.asarray(successes, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)

    if len(f_values) < 2 or np.any(totals <= 0):
        return np.nan, np.nan, False


    proportions = successes / totals
    initial_f_star = float(f_values[np.argmin(np.abs(proportions - 0.5))])
    initial_log_slope = math.log(1.0)

    def negative_log_likelihood(parameters):
        f_star = parameters[0]
        slope = math.exp(parameters[1])

        probabilities = expit(
            slope * (f_star - f_values)
        )

        probabilities = np.clip(
            probabilities,
            1e-9,
            1.0 - 1e-9,
        )

        log_likelihood = (
            successes * np.log(probabilities)
            +
            (totals - successes) * np.log(1.0 - probabilities)
        ).sum()

        return -float(log_likelihood)

    result = minimize(
        negative_log_likelihood,
        x0=np.array(
            [initial_f_star, initial_log_slope],
            dtype=np.float64,
        ),
        method="L-BFGS-B",
        bounds=[
            (
                float(f_values.min() - 5.0),
                float(f_values.max() + 5.0),
            ),
            (
                math.log(1e-3),
                math.log(100.0),
            ),
        ],
    )

    if not result.success:
        return np.nan, np.nan, False

    f_star = float(result.x[0])
    slope = float(math.exp(result.x[1]))

    return f_star, slope, True


def _bootstrap_capacity_for_configuration(
    configuration_frame: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 20260712,
):
    """
    Resample completed seeds independently within every F point, then refit
    the binomial logistic capacity model.
    """
    rng = np.random.default_rng(bootstrap_seed)

    grouped_success = {
        int(f_value): group["success"].astype(bool).to_numpy()
        for f_value, group in configuration_frame.groupby("F")
    }

    f_values = np.array(
        sorted(grouped_success),
        dtype=np.float64,
    )

    observed_successes = np.array(
        [grouped_success[int(f)].sum() for f in f_values],
        dtype=np.float64,
    )

    totals = np.array(
        [len(grouped_success[int(f)]) for f in f_values],
        dtype=np.float64,
    )

    observed_f_star, observed_slope, converged = _fit_binomial_capacity(
        f_values=f_values,
        successes=observed_successes,
        totals=totals,
    )

    bootstrap_f_stars = []
    bootstrap_slopes = []

    for _ in range(bootstrap_iterations):
        bootstrap_successes = []

        for f_value in f_values:
            values = grouped_success[int(f_value)]

            sampled = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            bootstrap_successes.append(
                sampled.sum()
            )

        fitted_f_star, fitted_slope, fitted = _fit_binomial_capacity(
            f_values=f_values,
            successes=np.asarray(
                bootstrap_successes,
                dtype=np.float64,
            ),
            totals=totals,
        )

        if fitted and np.isfinite(fitted_f_star):
            bootstrap_f_stars.append(fitted_f_star)
            bootstrap_slopes.append(fitted_slope)

    bootstrap_f_stars = np.asarray(
        bootstrap_f_stars,
        dtype=np.float64,
    )

    bootstrap_slopes = np.asarray(
        bootstrap_slopes,
        dtype=np.float64,
    )

    if len(bootstrap_f_stars) >= 100:
        f_star_ci_low, f_star_ci_high = np.quantile(
            bootstrap_f_stars,
            [0.025, 0.975],
        )
        slope_ci_low, slope_ci_high = np.quantile(
            bootstrap_slopes,
            [0.025, 0.975],
        )
    else:
        f_star_ci_low = np.nan
        f_star_ci_high = np.nan
        slope_ci_low = np.nan
        slope_ci_high = np.nan

    return {
        "F_star": observed_f_star,
        "slope": observed_slope,
        "fit_converged": bool(converged),
        "F_star_ci_low": float(f_star_ci_low),
        "F_star_ci_high": float(f_star_ci_high),
        "slope_ci_low": float(slope_ci_low),
        "slope_ci_high": float(slope_ci_high),
        "bootstrap_successful_fits": int(
            len(bootstrap_f_stars)
        ),
        "bootstrap_iterations": int(
            bootstrap_iterations
        ),
    }


def _paired_seed_difference_analysis(
    merged: pd.DataFrame,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260712,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)

    configurations = (
        merged[
            ["n_heads", "d_head"]
        ]
        .drop_duplicates()
        .sort_values(
            "d_head",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    rows = []

    for f_value in sorted(merged["F"].unique()):
        f_frame = merged[
            merged["F"] == f_value
        ]

        for first_index in range(len(configurations)):
            for second_index in range(
                first_index + 1,
                len(configurations),
            ):
                configuration_a = configurations.iloc[first_index]
                configuration_b = configurations.iloc[second_index]

                a = f_frame[
                    f_frame["n_heads"]
                    ==
                    configuration_a["n_heads"]
                ][
                    ["seed", "test_accuracy"]
                ].rename(
                    columns={
                        "test_accuracy":
                            "test_accuracy_a"
                    }
                )

                b = f_frame[
                    f_frame["n_heads"]
                    ==
                    configuration_b["n_heads"]
                ][
                    ["seed", "test_accuracy"]
                ].rename(
                    columns={
                        "test_accuracy":
                            "test_accuracy_b"
                    }
                )

                paired = a.merge(
                    b,
                    on="seed",
                    how="inner",
                )

                if paired.empty:
                    continue

                differences = (
                    paired["test_accuracy_a"].to_numpy()
                    -
                    paired["test_accuracy_b"].to_numpy()
                )

                bootstrap_means = np.empty(
                    bootstrap_iterations,
                    dtype=np.float64,
                )

                for iteration in range(
                    bootstrap_iterations
                ):
                    sampled = rng.choice(
                        differences,
                        size=len(differences),
                        replace=True,
                    )
                    bootstrap_means[iteration] = (
                        sampled.mean()
                    )

                ci_low, ci_high = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                )


                observed_mean = abs(
                    float(differences.mean())
                )

                if len(differences) <= 20:
                    number_assignments = 2 ** len(differences)
                    exceedances = 0

                    for assignment in range(
                        number_assignments
                    ):
                        signs = np.array(
                            [
                                1.0
                                if (
                                    assignment
                                    >>
                                    bit
                                )
                                &
                                1
                                else
                                -1.0
                                for bit in range(
                                    len(differences)
                                )
                            ],
                            dtype=np.float64,
                        )

                        permuted_mean = abs(
                            float(
                                (
                                    differences
                                    *
                                    signs
                                ).mean()
                            )
                        )

                        if (
                            permuted_mean
                            >=
                            observed_mean
                            -
                            1e-15
                        ):
                            exceedances += 1

                    p_value = (
                        exceedances
                        /
                        number_assignments
                    )
                else:
                    p_value = np.nan

                rows.append(
                    {
                        "F": int(f_value),
                        "n_heads_a": int(
                            configuration_a[
                                "n_heads"
                            ]
                        ),
                        "d_head_a": int(
                            configuration_a[
                                "d_head"
                            ]
                        ),
                        "n_heads_b": int(
                            configuration_b[
                                "n_heads"
                            ]
                        ),
                        "d_head_b": int(
                            configuration_b[
                                "d_head"
                            ]
                        ),
                        "paired_seeds": int(
                            len(differences)
                        ),
                        "mean_accuracy_difference_a_minus_b":
                            float(
                                differences.mean()
                            ),
                        "median_accuracy_difference_a_minus_b":
                            float(
                                np.median(
                                    differences
                                )
                            ),
                        "bootstrap_ci_low":
                            float(ci_low),
                        "bootstrap_ci_high":
                            float(ci_high),
                        "exact_sign_flip_p_value":
                            float(p_value),
                    }
                )

    return pd.DataFrame(rows)


def run_confirmatory_analysis() -> None:
    if PHASE != "confirmatory":
        return

    all_runs_path = os.path.join(
        FINAL_DIR,
        "confirmatory_ALL_RUNS.csv",
    )

    if not os.path.exists(all_runs_path):
        print(
            "[ANALYSIS] Merged confirmatory CSV not found.",
            flush=True,
        )
        return

    merged = pd.read_csv(all_runs_path)

    required_columns = {
        "n_heads",
        "d_head",
        "F",
        "seed",
        "success",
        "test_accuracy",
    }

    missing = required_columns.difference(
        merged.columns
    )

    if missing:
        raise RuntimeError(
            f"Confirmatory analysis is missing columns: "
            f"{sorted(missing)}"
        )

    capacity_rows = []

    for (
        n_heads,
        d_head,
    ), configuration_frame in merged.groupby(
        ["n_heads", "d_head"]
    ):
        result = _bootstrap_capacity_for_configuration(
            configuration_frame=
                configuration_frame,
            bootstrap_iterations=5000,
            bootstrap_seed=(
                20260712
                +
                int(n_heads)
            ),
        )

        capacity_rows.append(
            {
                "n_heads": int(n_heads),
                "d_head": int(d_head),
                **result,
            }
        )

    capacity_dataframe = pd.DataFrame(
        capacity_rows
    ).sort_values(
        "d_head",
        ascending=False,
    )

    capacity_path = os.path.join(
        FINAL_DIR,
        "confirmatory_CAPACITY_FSTAR.csv",
    )

    capacity_dataframe.to_csv(
        capacity_path,
        index=False,
    )

    pairwise_dataframe = (
        _paired_seed_difference_analysis(
            merged=merged,
            bootstrap_iterations=10000,
            bootstrap_seed=20260712,
        )
    )

    pairwise_path = os.path.join(
        FINAL_DIR,
        "confirmatory_PAIRED_DIFFERENCES.csv",
    )

    pairwise_dataframe.to_csv(
        pairwise_path,
        index=False,
    )

    print(
        "\n"
        +
        "=" * 80,
        flush=True,
    )

    print(
        "CONFIRMATORY CAPACITY ESTIMATES",
        flush=True,
    )


    print(
        capacity_dataframe.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        f"\nCapacity estimates:\n"
        f"{capacity_path}",
        flush=True,
    )

    print(
        f"\nPaired accuracy differences:\n"
        f"{pairwise_path}",
        flush=True,
    )


def package_outputs() -> str:
    archive_base = os.path.join(
        os.path.dirname(BASE_DIR),
        f"wikitext_capacity_{PHASE}_RESULTS",
    )

    archive_path = shutil.make_archive(
        archive_base,
        "zip",
        BASE_DIR,
    )

    print(f"[FINAL ZIP]\n{archive_path}", flush=True)
    return archive_path


def parent_main() -> None:
    print(f"PHASE: {PHASE}", flush=True)
    print(f"SEEDS: {SEEDS}", flush=True)
    print(f"F_VALUES: {F_VALUES}", flush=True)
    print(f"N_VALUE_CLASSES: {N_VALUE_CLASSES}", flush=True)
    print(f"EVALUATE_TEST: {EVALUATE_TEST}", flush=True)

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This experiment requires two visible CUDA GPUs."
        )


    prepare_data()

    torch.cuda.empty_cache()
    gc.collect()

    processes = []

    for physical_gpu, heads in GPU_ASSIGNMENTS.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

        heads_text = ",".join(str(value) for value in heads)

        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--heads",
            heads_text,
        ]


        process = subprocess.Popen(command, env=environment)
        processes.append((physical_gpu, process))

    failures = []

    for physical_gpu, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((physical_gpu, return_code))

    if failures:
        print(f"[WARNING] Worker failures: {failures}", flush=True)

    merge_results()
    run_confirmatory_analysis()
    package_outputs()


if __name__ == "__main__":
    initialize_output_directories()

    if arguments.worker:
        if not arguments.heads:
            raise ValueError("--heads is required in worker mode.")
        worker_main(arguments.heads)
    else:
        parent_main()
