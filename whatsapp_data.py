from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


MODEL_NAME_DEFAULT = "distilgpt2"
SPECIAL_TOKENS: List[str] = ["xxspk", "xxeom"]

logger = logging.getLogger(__name__)


_INVISIBLE_CHARS_RE = re.compile(
    r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u200b]"
)


def _strip_invisible(s: str) -> str:
    return _INVISIBLE_CHARS_RE.sub("", s)


def _keep_letters_marks_spaces(s: str) -> str:
    # Keep letters (L*) + combining marks (M*) so scripts like Hindi
    # (which use vowel signs/marks) remain readable.
    out_chars: List[str] = []
    for ch in s:
        if ch.isspace():
            out_chars.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            out_chars.append(ch)
        else:
            out_chars.append(" ")
    return re.sub(r"\s+", " ", "".join(out_chars)).strip()


def _clean_sender(sender: str) -> str:
    sender = _strip_invisible(sender)
    sender = _keep_letters_marks_spaces(sender)
    return sender


def _clean_message(message: str) -> str:
    message = _strip_invisible(message)

    # Remove WhatsApp @mentions like: @⁨Sathwik Rao⁩ (U+2068 ... U+2069)
    message = re.sub(r"@\u2068.*?\u2069", " ", message)
    # Also remove simple @name mentions
    message = re.sub(r"@\S+", " ", message)

    # Remove common WhatsApp placeholders / events
    message = re.sub(r"<\s*this message was edited\s*>", " ", message, flags=re.I)
    message = re.sub(r"this message was edited", " ", message, flags=re.I)
    message = re.sub(r"you deleted this message", " ", message, flags=re.I)
    message = re.sub(r"this message was deleted", " ", message, flags=re.I)
    message = re.sub(r"<media omitted>", " ", message, flags=re.I)

    # Remove URLs
    message = re.sub(r"https?://\S+|www\.\S+", " ", message)

    # Final: keep only letters/marks/spaces
    message = _keep_letters_marks_spaces(message)
    return message


def parse_usa_format(text_file: str) -> pd.DataFrame:
    """Parse WhatsApp export in USA-style bracket format.

    Expected line format (as used in your current notebook/script):
        [MM/DD/YY, HH:MM:SS AM] Name: message

    Returns a DataFrame with columns: timestamp (datetime64), sender (str), text (str).
    """

    # Some exports include invisible direction markers before '[' (e.g. U+200E).
    pattern = re.compile(
        r"^[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u200b]*"
        r"\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}:\d{2}\s*[APMapm\u202f]*)\]"
        r"\s*(.*?):\s*(.*)",
        re.MULTILINE,
    )

    rows = []

    with open(text_file, encoding="utf-8", errors="replace") as f:
        text = f.read()

    logger.info("Parsing chat file (USA format): %s", text_file)

    matches = list(pattern.finditer(text))
    logger.info("Regex matches found: %d", len(matches))

    dropped_omitted = 0
    dropped_empty = 0
    dropped_placeholder = 0

    for match in matches:
        timestamp_str = match.group(1)
        sender = _clean_sender(match.group(2).strip())
        raw_message = match.group(3).strip()

        lowered_raw = _strip_invisible(raw_message).lower()

        # Skip media / stickers / videos (and other non-text payloads)
        if "omitted" in lowered_raw:
            dropped_omitted += 1
            continue

        # Skip common non-chat events; keep this conservative.
        if "added" in lowered_raw or "created this group" in lowered_raw or "end-to-end encrypted" in lowered_raw:
            dropped_placeholder += 1
            continue

        message = _clean_message(raw_message)
        if not sender or not message or len(message.split()) < 2:
            dropped_empty += 1
            continue

        rows.append((timestamp_str, sender, message))

    df = pd.DataFrame(rows, columns=["timestamp", "sender", "text"])
    logger.info("Rows before timestamp parsing: %d", len(df))
    logger.info(
        "Dropped: omitted=%d placeholder=%d empty/short=%d",
        dropped_omitted,
        dropped_placeholder,
        dropped_empty,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="%m/%d/%y, %I:%M:%S %p",
    )

    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df = df.loc[df["sender"].astype(str).ne("")].copy()
    df = df.loc[df["text"].astype(str).str.split().str.len() > 1].reset_index(drop=True)

    logger.info("Rows after cleaning: %d", len(df))

    return df


def extract_participants(df: pd.DataFrame) -> List[str]:
    participants = (
        df["sender"]
        .astype(str)
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .unique()
        .tolist()
    )
    return participants


def add_spk(text: str, participants: List[str]) -> str:
    tokens = text.split()
    out: List[str] = []
    participants_set = set(participants)

    for token in tokens:
        if token.lower() in participants_set:
            out.append("xxspk")
            out.append(token.lower())
        else:
            out.append(token)

    return " ".join(out)


def train_valid_text_by_time(
    df: pd.DataFrame, *, participants: List[str], frac: float = 0.8
) -> Tuple[str, str]:
    df_sorted: pd.DataFrame = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df_sorted) * frac)

    def _to_lm_text(frame: pd.DataFrame) -> str:
        sender_norm = frame["sender"].astype(str).str.replace(" ", "_", regex=False)
        text_series = frame["text"].astype(str)
        # Explicit structure prevents accidental speaker-token injection when a
        # participant name appears inside the message body.
        merged_series = ("xxspk " + sender_norm + " " + text_series + " xxeom").astype(str)
        merged = merged_series.str.cat(sep=" ")
        return merged

    train_text = _to_lm_text(df_sorted.iloc[:cut])
    valid_text = _to_lm_text(df_sorted.iloc[cut:])

    logger.info(
        "Prepared LM text. train_chars=%d valid_chars=%d",
        len(train_text),
        len(valid_text),
    )

    return train_text, valid_text


def build_tokenizer(model_name: str, *, special_tokens: List[str]):
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenizer.add_special_tokens({"extra_special_tokens": list(special_tokens)})
    logger.info("Added special tokens: %s", ",".join(list(special_tokens)))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def token_ids(text: str, tokenizer) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class GPTBlockDataset(Dataset):
    """Fixed-length blocks for causal LM fine-tuning (labels = input_ids)."""

    def __init__(self, token_ids_list: List[int], block_size: int = 256):
        n_blocks = len(token_ids_list) // block_size
        trimmed = token_ids_list[: n_blocks * block_size]
        self.blocks = torch.tensor(trimmed, dtype=torch.long).view(n_blocks, block_size)

    def __len__(self) -> int:
        return int(self.blocks.size(0))

    def __getitem__(self, idx: int):
        x = self.blocks[idx]
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x,
        }


@dataclass(frozen=True)
class PreparedData:
    model_name: str
    participants: List[str]
    tokenizer: object
    train_text: str
    valid_text: str
    train_ds: GPTBlockDataset
    valid_ds: GPTBlockDataset


def prepare_datasets(
    chat_path: str,
    *,
    model_name: str = MODEL_NAME_DEFAULT,
    block_size: int = 256,
    train_frac: float = 0.8,
    special_tokens: List[str] = SPECIAL_TOKENS,
) -> PreparedData:
    df = parse_usa_format(chat_path)
    participants = extract_participants(df)

    train_text, valid_text = train_valid_text_by_time(df, participants=participants, frac=train_frac)

    tokenizer = build_tokenizer(model_name, special_tokens=special_tokens)

    train_ids = token_ids(train_text, tokenizer)
    valid_ids = token_ids(valid_text, tokenizer)

    train_ds = GPTBlockDataset(train_ids, block_size=block_size)
    valid_ds = GPTBlockDataset(valid_ids, block_size=block_size)

    return PreparedData(
        model_name=model_name,
        participants=participants,
        tokenizer=tokenizer,
        train_text=train_text,
        valid_text=valid_text,
        train_ds=train_ds,
        valid_ds=valid_ds,
    )


@dataclass(frozen=True)
class PreparedTextData:
    model_name: str
    participants: List[str]
    train_text: str
    valid_text: str
    special_tokens: List[str]


def prepare_text_data(
    chat_path: str,
    *,
    model_name: str = MODEL_NAME_DEFAULT,
    train_frac: float = 0.8,
    special_tokens: List[str] = SPECIAL_TOKENS,
) -> PreparedTextData:
    df = parse_usa_format(chat_path)
    participants = extract_participants(df)
    logger.info("Participants extracted: %d", len(participants))
    train_text, valid_text = train_valid_text_by_time(df, participants=participants, frac=train_frac)

    return PreparedTextData(
        model_name=model_name,
        participants=participants,
        train_text=train_text,
        valid_text=valid_text,
        special_tokens=list(special_tokens),
    )


def save_prepared_text(prepared: PreparedTextData, out_dir: str) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving prepared text artifacts to: %s", str(out_path))

    (out_path / "train_text.txt").write_text(prepared.train_text, encoding="utf-8")
    (out_path / "valid_text.txt").write_text(prepared.valid_text, encoding="utf-8")
    (out_path / "participants.json").write_text(
        json.dumps(prepared.participants, indent=2), encoding="utf-8"
    )
    meta = {
        "model_name": prepared.model_name,
        "special_tokens": prepared.special_tokens,
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return str(out_path)


def load_prepared_text(prepared_dir: str) -> PreparedTextData:
    base = Path(prepared_dir)
    logger.info("Loading prepared text artifacts from: %s", str(base))
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    participants = json.loads((base / "participants.json").read_text(encoding="utf-8"))
    train_text = (base / "train_text.txt").read_text(encoding="utf-8")
    valid_text = (base / "valid_text.txt").read_text(encoding="utf-8")

    logger.info(
        "Loaded prepared text. participants=%d train_chars=%d valid_chars=%d",
        len(participants),
        len(train_text),
        len(valid_text),
    )

    return PreparedTextData(
        model_name=str(meta.get("model_name", MODEL_NAME_DEFAULT)),
        participants=list(participants),
        train_text=train_text,
        valid_text=valid_text,
        special_tokens=list(meta.get("special_tokens", SPECIAL_TOKENS)),
    )
