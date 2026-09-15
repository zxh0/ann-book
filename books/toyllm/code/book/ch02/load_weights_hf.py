"""第二章 · 同样的事，交给 safetensors 库做

和 load_weights.py 对照着看：那个是自己拆字节，这个是调库。

运行：
    cd code
    uv run python book/ch02/load_weights_hf.py
"""

from pathlib import Path

from safetensors import safe_open
from tabulate import tabulate

# code/book/ch02/load_weights_hf.py → 上三层就是 code/
PATH = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M" / "model.safetensors"

stat = {"词嵌入": 0, "30个Decoder块": 0, "最后的norm": 0}
total = 0

with safe_open(PATH, framework="pt") as f:
    for name in f.keys():
        t = f.get_tensor(name)  # 直接拿到张量，不是裸字节
        print(f"{name:<48}{str(tuple(t.shape)):>14}  {t.dtype}")

        nbytes = t.numel() * t.element_size()  # 库不给 offset，长度只能这么算出来
        total += nbytes
        if name.startswith("model.layers."):
            stat["30个Decoder块"] += nbytes
        elif name == "model.embed_tokens.weight":
            stat["词嵌入"] += nbytes
        else:
            stat["最后的norm"] += nbytes

    meta = f.metadata()
    count = len(f.keys())

size = PATH.stat().st_size
print()
print(f"张量   {count} 块")
print(f"元信息 {meta}")
print(f"数据   {total} 字节")
print(f"文件   {size} 字节，比数据多 {size - total} 字节")
print("多出来的就是 8 字节长度加 JSON 头，但库不告诉我们它有多长，只能这么倒推。")

print()
print(tabulate(
    [(k, f"{v:,}", f"{v // 2:,}", f"{v / total:.1%}") for k, v in stat.items()]
    + [("合计", f"{total:,}", f"{total // 2:,}", "")],
    headers=["part", "bytes", "params", "ratio"],
    tablefmt="simple_outline", colalign=("left", "right", "right", "right"),
))

print()
print("库做不到的两件事：")
print("  一、拿不到 data_offsets。keys() 只给名字，get_tensor() 只给张量，")
print("      每块数据在文件里的位置是看不见的，上面的长度是拿形状乘 dtype 算的。")
print("  二、因此也无法验证“数据区正好读完、中间没有空隙”。")
print("      库当然自己做了这些检查，只是不让你看见。")
