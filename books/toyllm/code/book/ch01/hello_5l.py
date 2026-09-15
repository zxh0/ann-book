"""第一章 · 五行

整本书后面十章要手写的东西，用 HuggingFace transformers 就是这五行。

运行：
    cd code
    uv run python book/ch01/hello_5l.py
"""

from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()  # 只是让输出干净些，和正题无关

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"
PROMPT = "Once upon a time"

# ── 就是这五行 ─────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
ids = tok(PROMPT, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=20, do_sample=False)
text = tok.decode(out[0])
# ──────────────────────────────────────────────────────────────────────────

print(text)
