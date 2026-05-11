import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Uneasy lies the head that wears a crown."
print(tokenizer.encode(text))
# [52, 710, 4107, 7363, 262, 1182, 326, 17326, 257, 12389, 13]
