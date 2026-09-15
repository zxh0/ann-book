"""第二章 · 自己把权重加载一遍

按 JSON 头里的索引，把数据区一块一块读出来，读完验证正好到文件末尾，
既没有剩下没人认领的字节，也没有读过头。

运行：
    cd code
    uv run python book/ch02/load_weights.py
"""

import json
from pathlib import Path

from tabulate import tabulate

# code/book/ch02/load_weights.py → 上三层就是 code/
PATH = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M" / "model.safetensors"


with open(PATH, "rb") as f:
    # 一、头部长度 N
    n = int.from_bytes(f.read(8), "little")

    # 二、JSON 头。__metadata__ 不是张量，挑出去
    header = json.loads(f.read(n))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    # 三、按 offset 顺序把每块数据读出来。
    #     不能按 JSON 里的键顺序读，那只是碰巧和数据区一致，格式并不保证。
    cursor = 0
    rows = []
    groups = {}                              # 按层折叠：组名 → [张量数, 起点, 字节数]
    stat = {"词嵌入": 0, "30个Decoder块": 0, "最后的norm": 0}
    for name, meta in sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        begin, end = meta["data_offsets"]
        if begin != cursor:
            raise SystemExit(f"{name} 之前有 {begin - cursor} 字节没人认领")

        blob = f.read(end - begin)          # 顺序读下来，指针本来就停在 begin
        if len(blob) != end - begin:
            raise SystemExit(f"{name} 读短了：要 {end - begin} 字节，只拿到 {len(blob)}")

        # 272 个名字都以 model. 开头、以 .weight 结尾，列表里省掉，只留中间那截，
        # 再把 layers. 缩成 l.、post_attention 缩成 post_attn，表格能窄 10 列
        short = (name.removeprefix("model.").removesuffix(".weight")
                 .replace("layers.", "l.").replace("post_attention", "post_attn"))
        shape = "(" + ", ".join(str(d) for d in meta["shape"]) + ")"
        rows.append((short, shape, str(begin), str(end - begin)))
        cursor = end

        # 同一层的 9 个张量在数据区里本来就是连着的，所以一组的起点就是它第一个张量的起点。
        # 这张表列窄，layers. 不必缩写
        group = "layers." + name.split(".")[2] if name.startswith("model.layers.") else short
        if group not in groups:
            groups[group] = [0, begin, 0]
        groups[group][0] += 1
        groups[group][2] += end - begin

        # 顺手归个类：名字里带 layers.N 的属于 30 个 Decoder 块，其余两个在块外面
        if name.startswith("model.layers."):
            stat["30个Decoder块"] += len(blob)
        elif name == "model.embed_tokens.weight":
            stat["词嵌入"] += len(blob)
        else:
            stat["最后的norm"] += len(blob)

    # 四、验证：数据区正好读完，后面一个字节都不剩
    trailing = len(f.read())

# 272 个张量按层折叠起来看：30 层每层都是一模一样的 9 个张量、7080192 字节
print("按层折叠：\n")
print(tabulate([(g, str(cnt), str(begin), str(size)) for g, (cnt, begin, size) in groups.items()],
               headers=["part", "tensors", "offset", "bytes"],
               tablefmt="simple_outline", colalign=("left", "right", "right", "right")))

print("\n逐个张量。名字去掉了 model. 前缀和 .weight 后缀，layers. 缩写成 l.，post_attention 缩写成 post_attn\n")
print(tabulate(rows, headers=["tensor", "shape", "offset", "bytes"],
               tablefmt="simple_outline", colalign=("left", "right", "right", "right")))

size = PATH.stat().st_size
print()
print(f"张量   {len(tensors)} 块")
print(f"头部   8 + {n} = {8 + n} 字节")
print(f"数据   {cursor} 字节，读完剩 {trailing} 字节")
print(f"总计   {8 + n + cursor} 字节，文件实际 {size} 字节")

if trailing == 0 and 8 + n + cursor == size:
    print("\n正好读完，一个字节不多不少。")
else:
    raise SystemExit("\n对不上，说明索引理解错了")

# 这些真实数据都花在哪了。bf16 每个数 2 字节，所以参数量就是字节数的一半
print()
print(tabulate(
    [(k, f"{v:,}", f"{v // 2:,}", f"{v / cursor:.1%}") for k, v in stat.items()]
    + [("合计", f"{cursor:,}", f"{cursor // 2:,}", "")],
    headers=["part", "bytes", "params", "ratio"],
    tablefmt="simple_outline", colalign=("left", "right", "right", "right"),
))
