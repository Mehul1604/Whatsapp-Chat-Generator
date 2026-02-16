from __future__ import annotations

import logging

import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from generate_model import build_prompt, generate_chat_structured, load_generator
from train_model import TrainResult, train_model
from whatsapp_data import prepare_text_data, save_prepared_text


APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")


def _read_last_lines(path: Path, n: int = 50) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n:])


st.title("WhatsApp Chat Generator")

st.subheader("1) Upload")
uploaded = st.file_uploader("Upload WhatsApp chat export (.txt)", type=["txt"])

if uploaded is not None:
    run_name = st.text_input("Run name", value="run")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    chat_path = run_dir / "chat.txt"
    chat_path.write_bytes(uploaded.getvalue())

    logger.info("Upload saved: %s (%d bytes)", str(chat_path), chat_path.stat().st_size)

    st.success(f"Saved upload to {chat_path}")

    st.subheader("2) Prepare text")
    if st.button("Prepare", type="primary"):
        logger.info("Prepare clicked. chat_path=%s", str(chat_path))
        prepared = prepare_text_data(str(chat_path))
        prepared_dir = Path(save_prepared_text(prepared, str(run_dir / "prepared_text")))

        st.session_state["prepared_dir"] = str(prepared_dir)
        st.session_state["participants"] = prepared.participants

        logger.info(
            "Prepared saved. prepared_dir=%s participants=%d",
            str(prepared_dir),
            len(prepared.participants),
        )

        st.success(f"Prepared text saved to {prepared_dir}")
        st.write(f"Participants: {len(prepared.participants)}")
        st.json(prepared.participants)

    prepared_dir = st.session_state.get("prepared_dir")

    st.subheader("3) Train")
    out_dir = str(run_dir / "model_out")
    log_path = str(Path(out_dir) / "train_log.jsonl")

    if prepared_dir:
        st.write(f"Prepared folder: {prepared_dir}")

        if st.button("Start training"):
            logger.info("Train clicked. prepared_dir=%s out_dir=%s", prepared_dir, out_dir)
            st.session_state["training_running"] = True

            status = st.status("Training…", expanded=True)
            status.write("Logging to JSONL and updating live.")

            # Run training synchronously but keep updating the UI by reading the log file.
            # This is intentionally simple.
            result_container = st.empty()
            log_container = st.empty()

            # Kick off training in a lightweight way by calling the function directly.
            # The callback writes logs; we tail the file in this UI loop.
            import threading

            holder: dict[str, Optional[object]] = {"result": None, "error": None}

            def _train():
                try:
                    logger.info("Training thread start")
                    holder["result"] = train_model(
                        prepared_dir=str(prepared_dir),
                        output_dir=out_dir,
                        log_file=log_path,
                    )
                    logger.info("Training thread done")
                except Exception as e:
                    logger.exception("Training thread failed")
                    holder["error"] = str(e)

            t = threading.Thread(target=_train, daemon=True)
            t.start()

            while t.is_alive():
                tail = _read_last_lines(Path(log_path), n=40)
                if tail:
                    log_container.code(tail, language="json")
                time.sleep(1.0)

            err = holder.get("error")
            if err:
                status.update(label="Training failed", state="error")
                st.error(str(err))
            else:
                res = holder.get("result")
                assert isinstance(res, TrainResult)
                st.session_state["train_out_dir"] = res.output_dir
                st.session_state["pth_path"] = res.pth_path

                logger.info(
                    "Training complete. eval_loss=%s perplexity=%s pth_path=%s",
                    res.eval_loss,
                    res.perplexity,
                    res.pth_path,
                )

                status.update(label="Training complete", state="complete")
                result_container.write(
                    {
                        "eval_loss": res.eval_loss,
                        "perplexity": res.perplexity,
                        "pth_path": res.pth_path,
                    }
                )

    else:
        st.info("Prepare the text first.")

    st.subheader("4) Generate")
    pth_path = st.session_state.get("pth_path")

    if pth_path:
        st.write(f"Using model: {pth_path}")

        user_message = st.text_input("Message for speaker 1", value="who ate lunch ?")
        n_turns = st.slider("Turns", min_value=1, max_value=12, value=6)
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.65)

        if st.button("Generate"):
            logger.info("Generate clicked. pth_path=%s", pth_path)
            gen = load_generator(pth_path)
            prompt = build_prompt(gen.participants, user_message)
            logger.info(
                "Generation prompt built. n_turns=%d temperature=%s",
                n_turns,
                temperature,
            )
            text = generate_chat_structured(gen, prompt, n_turns=n_turns, temperature=temperature)
            st.text_area("Generated chat", value=text, height=400)
    else:
        st.info("Train a model first to enable generation.")

else:
    st.info("Upload a chat .txt file to begin.")
