import re
from gensim.models import Word2Vec

# Split into word and punctuation tokens; newlines become <nl>.
def tokenize(text):
    raw = re.findall(r"\n|[A-Za-z']+|[^A-Za-z'\s]", text)
    return ["<nl>" if t == "\n" else t.lower() for t in raw]


# 待处理文本
text = "Uneasy lies the head that wears a crown."

# 简单分词
words = tokenize(text)
print("\nwords", words)

# 训练需要二维列表
sentences = [words]

# 训练简易 Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=20, # 词向量维度，可自己改
    window=2,
    min_count=1,
)

# 3. 逐个获取每个词的向量（对应 tiktoken 每个token编号）
max_len = max(len(repr(word)) for word in words)
for word in words:
    vec = model.wv[word]
    print(f"word: {word:<{max_len}}, vector: {vec.round(4)}")
