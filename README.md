# WhatsApp Chat Generator (Transformer + Streamlit)

Train a small GPT-2-style Transformer language model on a WhatsApp exported chat file and generate multi-speaker chat in the same style.

This project is intentionally simple and is organized into four Python modules:

- `whatsapp_data.py` – parsing + cleaning + prepared-text artifacts
- `train_model.py` – fine-tuning (Hugging Face `Trainer`) + saving a single `.pth` artifact + tokenizer folder
- `generate_model.py` – loading + structured generation (speaker turns, stop at `xxeom`)
- `app.py` – Streamlit UI to upload → prepare → train → generate

## How it works

### 1) Data preparation (`whatsapp_data.py`)

**Input format (USA export style)**

The parser expects lines like:

```
[MM/DD/YY, HH:MM:SS AM] Name: message
```

**Parsing + cleaning**

`parse_usa_format()`:

- Extracts `(timestamp, sender, text)` rows.
- Strips common invisible direction markers sometimes present in exports.
- Drops media/placeholder-like content (e.g. “omitted”, “end-to-end encrypted”).
- Cleans text by removing URLs and @mentions.

Note: the current cleaner keeps only Unicode letters/combining marks/spaces (so Devanagari stays readable). This simplifies the modeling task but also removes punctuation/digits.

**Participants**

`extract_participants()` produces a normalized participant list (lowercased, spaces replaced with underscores). These names are used both for training and for rendering generated output.

**Training text format: special control tokens**

Each message is converted to a single training string with explicit boundaries:

```
xxspk <sender_name> <message text> xxeom
```

- `xxspk` marks “speaker header begins”
- `xxeom` marks “end of message”

`train_valid_text_by_time()` splits the chat by time (earlier messages = train, later messages = validation) and concatenates all messages into two long strings.

**Prepared-text artifacts (Option A)**

`prepare_text_data()` + `save_prepared_text()` create a folder containing:

- `train_text.txt`
- `valid_text.txt`
- `participants.json`
- `meta.json` (model name + special tokens)

These artifacts are what training consumes.

### 2) Tokenization (`whatsapp_data.py` → `build_tokenizer()`)

This project fine-tunes a pretrained GPT-2 family tokenizer (default base model: `distilgpt2`).

- `build_tokenizer()` loads the base tokenizer and adds `xxspk` and `xxeom` as extra special tokens.
- The model embeddings are resized during training/loading so these new tokens are learnable.

### 3) Model + training (`train_model.py`)

**Model architecture**

The training uses Hugging Face `AutoModelForCausalLM` with a GPT-2-style causal language modeling head (next-token prediction). By default it starts from `distilgpt2` weights.

**Dataset**

Training uses fixed-length blocks (see `GPTBlockDataset` in `whatsapp_data.py`):

- Tokenize the concatenated `train_text.txt` / `valid_text.txt`
- Slice into blocks of `block_size` tokens
- Use `labels = input_ids` (standard causal LM objective)

**Trainer**

`train_model()` runs Hugging Face `Trainer` with step-based evaluation and early stopping.

**Outputs**

Training writes:

- `<out>/whatsapp_model.pth` – a single torch artifact containing:
  - `state_dict`
  - `model_name`
  - `participants`
  - `special_tokens`
  - eval metrics (`eval_loss`, `perplexity`)
- `<out>/tokenizer/` – tokenizer saved via `save_pretrained()`
- `<out>/train_log.jsonl` – step logs (used by the Streamlit UI)
- `<out>/train_summary.json`

### 4) Generation (`generate_model.py`)

**Loading**

`load_generator()`:

- Loads the `.pth` artifact
- Loads the tokenizer from `<out>/tokenizer/`
- Reconstructs the base model (`AutoModelForCausalLM.from_pretrained(model_name)`), resizes embeddings, then loads your fine-tuned weights

**Structured multi-turn generation**

The generator continues a prompt like:

```
xxspk <p0> <your message> xxeom xxspk <p1>
```

Then it repeats:

- generate one message with Hugging Face `model.generate(...)`
- stop at `xxeom` (used as `eos_token_id`)
- inject the next forced header `xxspk <speaker> `

This keeps turns bounded and makes output rendering reliable.

**Decoding settings**

Sampling is controlled mainly by:

- `temperature` (lower = safer)
- `top_p` nucleus sampling

If you see gibberish / “language drift”, try lowering `temperature` and `top_p`.

## Setup

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run (Streamlit UI)

```bash
streamlit run app.py
```

In the UI:

1. Upload your WhatsApp export `.txt`
2. Click **Prepare** (writes `runs/<run_name>/prepared_text/*`)
3. Click **Start training** (writes `runs/<run_name>/model_out/*` and shows live JSONL logs)
4. Click **Generate**

## Run (CLI)

### Train from a raw chat export

```bash
python train_model.py --chat /path/to/chat.txt --out runs/my_run/model_out
```

This automatically creates `runs/my_run/model_out/prepared_text/` and then trains.

### Train from an existing prepared folder

```bash
python train_model.py --prepared runs/my_run/model_out/prepared_text --out runs/my_run/model_out
```

## Notes & privacy

- Chat exports can contain sensitive information. Don’t commit them.
- Language models can memorize training text; generated output may contain verbatim snippets.