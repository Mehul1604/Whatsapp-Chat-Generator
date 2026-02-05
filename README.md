# WhatsApp Chatbot (LSTM)

LSTM-based chatbot trained on personal WhatsApp group chats. The notebook preprocesses conversation history to learn unique texting styles, slang, and informal patterns. It aims for realistic, personalized responses while optimizing for low-resource local training — no expensive cloud compute or large transformer models required.

**What it does**
- **Preprocesses:** Cleans WhatsApp exports (timestamps, names, media, URLs, emojis) and filters repeated lines.
- **Tokenizes:** Builds a tokenizer and converts chat lines into n-gram sequences for next-word prediction.
- **Trains:** Trains an LSTM-based language model on the processed text to generate chat-style responses.
- **Local & Lightweight:** Designed to run on a local machine with modest resources for experimentation.

**Files**
- Project notebook: [Whatsapp-Chat-Generator/whatsapp_generator.ipynb](Whatsapp-Chat-Generator/whatsapp_generator.ipynb)
- Raw chat dataset: [Whatsapp-Chat-Generator/chat.txt](Whatsapp-Chat-Generator/chat.txt)

**Requirements**
- Python 3.8+
- pip
- Recommended packages (install with pip):

```bash
pip install tensorflow numpy jupyter
```

(If on macOS with Apple Silicon, consider `tensorflow-macos` and `tensorflow-metal` for better performance.)

**How to run**
1. Open the project folder and ensure `chat.txt` is present in the same folder as the notebook.
2. Start Jupyter:

```bash
jupyter notebook
# or
jupyter lab
```

3. Open [Whatsapp-Chat-Generator/whatsapp_generator.ipynb](Whatsapp-Chat-Generator/whatsapp_generator.ipynb) and run cells in order.
   - Preprocessing cells produce `cleaned_lines` and prepare `data`.
   - Tokenization and sequence creation cells produce `X` and `y`.
   - The model is defined and trained in the later cells (`model.fit(...)`).
4. Adjust hyperparameters as needed (e.g., `epochs`, `batch_size`, LSTM units) before training.

**Saving a trained model (optional)**
To persist the trained model after training, add a cell after training with:

```python
model.save('lstm_whatsapp_bot.h5')
```

**Notes & privacy**
- This project is intended for personal use and local training on private chat data. Do not share sensitive conversations.
- The model learns and reproduces informal language and slang present in the dataset. Use responsibly.