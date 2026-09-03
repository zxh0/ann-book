## 第六章：词元和词嵌入

Token

Word Embedding



先介绍Token和Embeding？

BOW/Word2Vec



word2vec:
https://remykarem.github.io/word2vec-demo/


tiktoken:
https://github.com/openai/tiktoken
https://platform.openai.com/tokenizer



byte pair encoding (BPE), (widely used by GPT models)
WordPiece (used by BERT)



Word tokens
Subword tokens
Character tokens
Byte tokens





### 词元和分词

编译原理，也要分词

（Token）

分词，代码：

```python
import re

def tokenize(text):
    raw = re.findall(r"\n|[A-Za-z']+|[^A-Za-z'\s]", text)
    return ["<nl>" if t == "\n" else t.lower() for t in raw]

txt = "To be, or not to be, that is the question."
print(tokenize(txt))
```

输出：

```
['to', 'be', ',', 'or', 'not', 'to', 'be', ',', 
 'that', 'is', 'the', 'question', '.']
```



TODO：byte pair encoding (BPE)、tiktoken



### 独热编码

One-hot

```python
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
word_one_hot(txt)
```

输出：

```
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
```





### 词嵌入

TODO



### Word2Vec

TODO



### 本章小结

TODO
