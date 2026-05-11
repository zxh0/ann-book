import re
import numpy as np


# Split into word and punctuation tokens; newlines become <nl>.
def tokenize(text):
    raw = re.findall(r"\n|[A-Za-z']+|[^A-Za-z'\s]", text)
    return ["<nl>" if t == "\n" else t.lower() for t in raw]


def word_one_hot(txt: str):
    # 1. 收集不重复的 token，排序
    vocab = sorted(set(tokenize(txt)))

    # 2. 生成 one-hot 矩阵
    one_hot = np.eye(len(vocab), dtype=int)

    # 3. 打印每个 token 及其 one-hot vector
    max_len = max(len(repr(token)) for token in vocab)
    for idx, token in enumerate(vocab):
        print(f"{repr(token):<{max_len}}: {one_hot[idx].tolist()}")


txt = "To be, or not to be, that is the question."
print(tokenize(txt))
# ['to', 'be', ',', 'or', 'not', 'to', 'be', ',', 'that', 'is', 'the', 'question', '.']

word_one_hot(txt)
','       : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'.'       : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
'be'      : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
'is'      : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
'not'     : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
'or'      : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
'question': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
'that'    : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
'the'     : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
'to'      : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
