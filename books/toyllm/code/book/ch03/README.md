# 第三章 · 分词（Tokenization）

这一章全程不打开 `model.safetensors`。词表和合并规则住在 `tokenizer.json` 里，
2 MB，和那 269 MB 的权重毫无关系。把权重文件改个名字藏起来，这一章的代码照跑不误。

## 运行

```bash
cd code
uv sync --group dev                                  # tokenizers 在 dev 组里
uv run python book/ch03/tokenize_demo.py             # 看一段文字被切成哪些词元
uv run python book/ch03/tokenize_demo.py "任意一段话"
uv run python book/ch03/fake_engine.py               # 把两头接起来，转一圈
```

## 文件

| 文件 | 作用 |
|---|---|
| `tokenize_demo.py` | 给一段文字，打印切出来的词元和 ID，以及字符数与词元数之比 |
| `fake_engine.py` | 文字 → ID →（随机数冒充 LLM）→ ID → 文字，走完整条流水线，最后演示逐个词元解码为什么不行 |

## 只需要一个文件

`tokenizers` 库读的就是 `tokenizer.json` 一个文件，`Tokenizer.from_file()` 的参数是
单个文件路径，其余四个分词相关的文件它不知道存在。想验证的话，把 `tokenizer.json`
单独拷进一个空目录，两个脚本照跑：

```bash
mkdir -p /tmp/only_tj && cp models/SmolLM2-135M/tokenizer.json /tmp/only_tj/
uv run python -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('/tmp/only_tj/tokenizer.json')
print(t.encode('Once upon a time', add_special_tokens=False).ids)
"
# [6403, 1980, 253, 655]
```

## fake_engine.py 的输出

核心就三行，一头一尾是本章的正主，中间那个 `fake_llm` 是假的：

```python
enc = tok.encode(text, add_special_tokens=False)   # Tokenizer：文字到ID
ids = fake_llm(tok, list(enc.ids), 10)             # 冒充LLM和采样器，转10圈
print(tok.decode(ids))                             # Detokenizer：ID回到文字
```

```
文字   'Once upon a time'
词元   ['Once', 'Ġupon', 'Ġa', 'Ġtime']
ID     [6403, 1980, 253, 655]

冒充LLM，转10圈：
  第 5个词元   ID 43417   'survey'
  ...
  第14个词元   ID 15809   'Ġdop'

文字   'Once upon a timesurvey enjoyoicesEntry harnessvehOLDocument lighthouse dop'
```

随机数种子写死在 `SEED` 里，所以书里引用的这段输出可以原样复现。文字通顺不了是
应该的，中间那个 LLM 还不存在，第四章才开始造。

最后还演示一件事：`'第三章'` 切成六个词元，每一个单独解码都是 `�`，六个拼起来
还是六个 `�`，整串一次解码才是 `'第三章'`。解码的单位是整串 ID，不是单个词元。
这个坑第十章做流式输出时会真的踩到。

## 引擎现状

两头通了，中间是空的。文字能变成 ID，ID 能变回文字，但从输入 ID 到输出 ID
那一步还完全不存在。
