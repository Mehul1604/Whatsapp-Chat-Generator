# WhatsApp Chatbot (Fastai AWD‑LSTM)

Trains a lightweight Fastai language model (AWD‑LSTM) on a personal WhatsApp chat export (`chat.txt`) to generate chat-style text.

This repo is notebook-first and optimized for experimentation: you can quickly iterate on preprocessing choices and see how they affect loss curves and generations.

## Files

- Main notebook (Fastai AWD‑LSTM): [whatsapp_generator_lstm.ipynb](whatsapp_generator_lstm.ipynb)
- Raw chat dataset (example): [chat.txt](chat.txt)

## What the notebook does

- **Parses WhatsApp exports** into a dataframe (`timestamp`, `sender`, `text`).
- **Builds a language-model training stream** by concatenating turns and injecting a speaker control token:
   - `xxspk <speaker_name>` (lets the LM learn multi-speaker turn-taking from plain text)
- **Trains an AWD‑LSTM language model** with Fastai (`language_model_learner`) using next-token prediction.

## Training process (Fastai)

The notebook uses the standard Fastai LM fine-tuning pattern:

1. **Create `TextDataLoaders`** (language modeling mode, e.g. `seq_len=128`).
2. **Initialize the learner** with `language_model_learner(dls, AWD_LSTM, ...)`.
3. **Two-stage training**:
    - `learn.freeze()` then `fit_one_cycle(...)` for a short warmup stage.
    - `learn.unfreeze()` then `fit_one_cycle(...)` to fine-tune the full model.

Recommended early stopping for language models is to monitor `valid_loss` (rather than token accuracy), since loss/perplexity better reflect LM generalization.

## Straightforward upgrades (recommended)

These are high-impact improvements for better learning and more realistic chat outputs:

- **Tolerant timestamp parsing**: avoid filtering rows by a fixed timestamp string length; parse robustly and drop only rows that truly fail parsing.
- **Preserve emojis/punctuation** (or normalize to tokens): emojis are strong tone/style signals in chat.
- **Explicit message boundaries**: add `xxeom` after each message so the model learns message-length rhythm and turn boundaries.
- **Time-aware split**: train on earlier messages and validate on later messages for a realistic evaluation.

If you want a stronger baseline model, this repo also contains an experimental GPT‑2–style Transformer notebook: [whatsapp_generator_pytorch.ipynb](whatsapp_generator_pytorch.ipynb).

## Requirements

- Python 3.8+
- Jupyter (Notebook or Lab)
- Packages:
   - `fastai`
   - `pandas`

Install:

```bash
pip install -U fastai pandas jupyter
```

## How to run

1. Put your exported WhatsApp chat file at `chat.txt` in the project folder.
2. Start Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open [whatsapp_generator_lstm.ipynb](whatsapp_generator_lstm.ipynb) and run cells in order.

## Notes & privacy

- Keep your chat logs private; they can contain sensitive data.
- Generative models can memorize and reproduce training text. Treat outputs as potentially sensitive.