from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from whatsapp_data import (
    MODEL_NAME_DEFAULT,
    GPTBlockDataset,
    build_tokenizer,
    load_prepared_text,
    prepare_text_data,
    save_prepared_text,
    token_ids,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    eval_loss: Optional[float]
    perplexity: Optional[float]
    output_dir: str
    pth_path: str


class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()},
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def prepare_text_artifacts(
    chat_path: str,
    *,
    out_dir: str,
    model_name: str = MODEL_NAME_DEFAULT,
    train_frac: float = 0.8,
) -> str:
    """Creates (or overwrites) prepared text artifacts in <out_dir>/prepared_text.

    This is the Option-A preparation step moved into training code as requested.
    """

    prepared_text = prepare_text_data(
        chat_path,
        model_name=model_name,
        train_frac=train_frac,
    )
    prepared_dir = Path(out_dir) / "prepared_text"
    logger.info("Prepared text artifacts created in: %s", str(prepared_dir))
    return save_prepared_text(prepared_text, str(prepared_dir))


def train_model(
    *,
    chat_path: Optional[str] = None,
    prepared_dir: Optional[str] = None,
    output_dir: str = "gpt2_whatsapp",
    model_name: Optional[str] = None,
    seed: int = 42,
    block_size: int = 128,
    train_frac: float = 0.8,
    num_train_epochs: float = 5,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    eval_steps: int = 10,
    save_steps: int = 10,
    early_stopping_patience: int = 3,
    log_file: Optional[str] = None,
    pth_name: str = "whatsapp_model.pth",
) -> TrainResult:
    set_seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training start. output_dir=%s", str(out_dir))

    log_path = Path(log_file) if log_file else (out_dir / "train_log.jsonl")

    if (chat_path is None) == (prepared_dir is None):
        raise ValueError("Provide exactly one of chat_path or prepared_dir")

    if prepared_dir is None:
        base_model_name = model_name or MODEL_NAME_DEFAULT
        logger.info("No prepared_dir provided; preparing from chat: %s", chat_path)
        assert chat_path is not None
        prepared_dir = prepare_text_artifacts(
            chat_path, out_dir=str(out_dir), model_name=base_model_name, train_frac=train_frac
        )
    else:
        logger.info("Using existing prepared_dir: %s", prepared_dir)

    prepared = load_prepared_text(prepared_dir)
    effective_model_name = model_name or prepared.model_name or MODEL_NAME_DEFAULT
    logger.info("Effective model name: %s", effective_model_name)

    tokenizer = build_tokenizer(effective_model_name, special_tokens=prepared.special_tokens)

    train_ids = token_ids(prepared.train_text, tokenizer)
    valid_ids = token_ids(prepared.valid_text, tokenizer)
    train_ds = GPTBlockDataset(train_ids, block_size=block_size)
    valid_ds = GPTBlockDataset(valid_ids, block_size=block_size)

    logger.info(
        "Datasets built. block_size=%d train_blocks=%d valid_blocks=%d",
        block_size,
        len(train_ds),
        len(valid_ds),
    )

    model = AutoModelForCausalLM.from_pretrained(effective_model_name)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=eval_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        report_to="none",
    )

    callbacks = [
        JsonlLoggerCallback(log_path),
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        callbacks=callbacks,
    )

    trainer.train()
    eval_metrics: Dict[str, Any] = trainer.evaluate()

    eval_loss = eval_metrics.get("eval_loss")
    eval_loss_f = float(eval_loss) if eval_loss is not None else None
    ppl = math.exp(eval_loss_f) if eval_loss_f is not None else None

    logger.info("Eval done. eval_loss=%s perplexity=%s", eval_loss_f, ppl)

    # Save a single .pth artifact for the Streamlit app + generator script
    pth_path = out_dir / pth_name
    trained_model = trainer.model or model

    artifact = {
        "model_name": effective_model_name,
        "special_tokens": prepared.special_tokens,
        "participants": prepared.participants,
        "state_dict": trained_model.state_dict(),
        "eval_loss": eval_loss_f,
        "perplexity": ppl,
    }
    torch.save(artifact, str(pth_path))
    logger.info("Saved .pth artifact: %s", str(pth_path))

    # Also persist tokenizer (so you can use HF-native load if desired)
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    logger.info("Saved tokenizer folder: %s", str(out_dir / "tokenizer"))

    # Write a small summary json for convenience
    summary = {
        "eval_loss": eval_loss_f,
        "perplexity": ppl,
        "output_dir": str(out_dir),
        "pth_path": str(pth_path),
        "n_participants": len(prepared.participants),
        "participants": prepared.participants,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return TrainResult(
        eval_loss=eval_loss_f,
        perplexity=ppl,
        output_dir=str(out_dir),
        pth_path=str(pth_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--chat", help="Path to WhatsApp exported chat .txt")
    src.add_argument(
        "--prepared",
        help="Prepared folder containing train_text.txt, valid_text.txt, participants.json, meta.json",
    )
    parser.add_argument("--out", default="gpt2_whatsapp", help="Output directory")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Override base model name (default: from prepared meta.json, fallback {MODEL_NAME_DEFAULT})",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction (used only with --chat)")
    parser.add_argument("--log", default=None, help="JSONL log path (defaults to <out>/train_log.jsonl)")

    args = parser.parse_args()

    result = train_model(
        chat_path=args.chat,
        prepared_dir=args.prepared,
        output_dir=args.out,
        model_name=args.model,
        train_frac=args.train_frac,
        log_file=args.log,
    )

    print(f"EVAL LOSS: {result.eval_loss}")
    print(f"PERPLEXITY: {result.perplexity}")
    print(f"PTH SAVED: {result.pth_path}")


if __name__ == "__main__":
    main()
