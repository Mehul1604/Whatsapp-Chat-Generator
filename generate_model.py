from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedGenerator:
    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    participants: List[str]
    device: torch.device
    xxeom_id: int
    xxspk_id: int


def load_generator(
    pth_path: str,
    *,
    tokenizer_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> LoadedGenerator:
    """Load the trained model weights saved by train_model.py.

    Expects a torch-saved dict with keys:
      - model_name
      - participants
      - special_tokens
      - state_dict

    Tokenizer is loaded from <output_dir>/tokenizer by default.
    """

    logger.info("Loading generator artifact: %s", pth_path)
    artifact = torch.load(pth_path, map_location="cpu")
    model_name = artifact["model_name"]
    participants = list(artifact.get("participants", []))

    logger.info("Artifact model_name=%s participants=%d", model_name, len(participants))

    pth_parent = Path(pth_path).resolve().parent
    tok_dir = Path(tokenizer_dir) if tokenizer_dir else (pth_parent / "tokenizer")

    logger.info("Loading tokenizer from: %s", str(tok_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(artifact["state_dict"], strict=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)
    logger.info("Using device: %s", str(device_obj))

    cast(nn.Module, model).to(device_obj)
    model.eval()

    xxeom_id = tokenizer.convert_tokens_to_ids("xxeom")
    xxspk_id = tokenizer.convert_tokens_to_ids("xxspk")

    return LoadedGenerator(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        participants=participants,
        device=device_obj,
        xxeom_id=xxeom_id,
        xxspk_id=xxspk_id,
    )


def generate_one_turn_strict(
    gen: LoadedGenerator,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 40,
    temperature: float = 0.65,
    top_p: float = 0.8,
    repetition_penalty: float = 1.2,
) -> torch.Tensor:
    """Generate exactly one message, stopping immediately at xxeom.

    Note: top_p and repetition_penalty are kept for API compatibility with your
    original notebook/script; sampling matches the original logic.
    """

    logger.info(
        "Generating one turn (HF generate). max_new_tokens=%d temperature=%s top_p=%s repetition_penalty=%s",
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty,
    )

    input_ids = input_ids.to(gen.device)

    model = cast(PreTrainedModel, gen.model)

    with torch.no_grad():
        out = model.generate(  # pyright: ignore[reportCallIssue]
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=gen.xxeom_id,
            pad_token_id=(gen.tokenizer.eos_token_id if gen.tokenizer.eos_token_id is not None else gen.xxeom_id),
        )

    return out


def render_generated(text: str, participants: List[str]) -> str:
    # Important: do NOT blindly replace participant substrings.
    # Only label speakers when they appear as an explicit header: `xxspk <participant>`.
    text = (
        text.replace("xxeom", "\n")
        .replace(" \\\\\u0027", "\\\\u0027")
        .replace(" n\\u0027t", "n\\u0027t")
    )

    for participant in participants:
        pat = rf"\bxxspk\s+{re.escape(participant)}\b"
        text = re.sub(pat, f"\n{participant}: ", text)

    # Fallback: if any stray xxspk remains, break the line.
    text = text.replace("xxspk", "\n")

    text = re.sub(r"\s([?.!\"](?:\s|$))", r"\1", text)
    text = re.sub(r"\n+", "\n", text).strip()
    return text


def build_prompt(participants: List[str], message: str) -> str:
    if len(participants) < 2:
        raise ValueError("Need at least 2 participants to build a two-speaker prompt")

    p0, p1 = participants[0], participants[1]
    logger.info("Building prompt with participants: %s, %s", p0, p1)
    return f"xxspk {p0} {message} xxeom xxspk {p1} "


def generate_chat_structured(
    gen: LoadedGenerator,
    prompt: str,
    *,
    n_turns: int = 5,
    temperature: float = 0.65,
    top_p: float = 0.8,
) -> str:
    """Generate a multi-turn chat continuing from a prompt.

    Mirrors your original script behavior:
    - start from prompt tokens
    - generate one message at a time until xxeom
    - alternate speaker deterministically (picks the first participant != last)
    - inject a forced header: xxspk <speaker_name>
    """

    if len(gen.participants) < 2:
        raise ValueError("Need at least 2 participants for structured chat generation")

    logger.info("Generating chat. n_turns=%d", n_turns)

    enc = gen.tokenizer(prompt, return_tensors="pt")
    input_ids = cast(torch.Tensor, enc["input_ids"]).to(gen.device)
    generated_ids = input_ids

    last_speaker = gen.participants[1]

    for _ in range(n_turns):
        generated_ids = generate_one_turn_strict(
            gen,
            generated_ids,
            temperature=temperature,
            top_p=top_p,
        )

        next_speaker = None
        for p in gen.participants:
            if p != last_speaker:
                next_speaker = p
                break

        last_speaker = next_speaker or last_speaker

        assert next_speaker is not None
        header_ids = gen.tokenizer.encode(f"xxspk {next_speaker} ", add_special_tokens=False)
        header = torch.tensor([header_ids], device=gen.device)
        generated_ids = torch.cat([generated_ids, header], dim=1)

    decoded = cast(str, gen.tokenizer.decode(generated_ids[0], skip_special_tokens=False))
    return render_generated(decoded, gen.participants)
