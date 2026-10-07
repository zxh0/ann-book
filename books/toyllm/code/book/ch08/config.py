"""第八章 · 模型的规格说明

`Config`是第五章写的，那时只为RMSNorm要一个`rms_norm_eps`，第六章为注意力补了
两个头数，第七章为RoPE补了三个。这一章FFN只要一个字段：

    intermediate_size        1536     FFN中间层多宽，576 × 8/3

一个`Config`对象就是这个JSON文件。这里只给用得上的字段起了名字，其余的都留在
`raw`里，后面章节用到哪个，就在这里补一个名字，顺便讲清楚它是干什么的。

注意这个类**只读不算**：它不碰权重，也不做任何计算，就是把一份配置搬进内存。
`Weights`管数据，`Config`管形状和超参数，`Model`管计算。

运行：
    cd code
    uv run python book/ch08/config.py
"""

import json

from weights import MODEL_DIR


class Config:
    """SmolLM2的规格说明，读自`config.json`。"""

    def __init__(self, model_dir=MODEL_DIR):
        with open(model_dir / "config.json", encoding="utf-8") as f:
            self.raw = json.load(f)

        # 已经讲过的几个。第二章在元数据里见过576和30，第三章数过49152，
        # 第四章靠tie_word_embeddings解释了为什么找不到lm_head.weight。
        self.hidden_size = self.raw["hidden_size"]                    # 576
        self.num_hidden_layers = self.raw["num_hidden_layers"]        # 30
        self.vocab_size = self.raw["vocab_size"]                      # 49152
        self.tie_word_embeddings = self.raw["tie_word_embeddings"]    # true

        # 第五章新要的那个。RMSNorm算均方根时要加上它，防止除以0。
        self.rms_norm_eps = self.raw["rms_norm_eps"]                  # 1e-05

        # 第六章新要的两个。q有9个头，k和v只有3个头，这就是GQA。
        self.num_attention_heads = self.raw["num_attention_heads"]    # 9
        self.num_key_value_heads = self.raw["num_key_value_heads"]    # 3

        # 第七章新要的三个，都是RoPE用的。RoPE一个可训练参数都没有，
        # 全靠这几个数算出一张三角函数表。
        self.rope_theta = self.raw["rope_theta"]                      # 100000
        self.rope_interleaved = self.raw["rope_interleaved"]          # false
        self.max_position_embeddings = self.raw["max_position_embeddings"]  # 8192

        # 第八章新要的一个。FFN先升到这么宽，再降回576。
        self.intermediate_size = self.raw["intermediate_size"]        # 1536

    def __repr__(self):
        return (f"Config(hidden_size={self.hidden_size}, "
                f"num_hidden_layers={self.num_hidden_layers}, "
                f"vocab_size={self.vocab_size}, "
                f"rms_norm_eps={self.rms_norm_eps}, "
                f"num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}, "
                f"rope_theta={self.rope_theta}, "
                f"rope_interleaved={self.rope_interleaved}, "
                f"max_position_embeddings={self.max_position_embeddings}, "
                f"intermediate_size={self.intermediate_size})")


def main():
    cfg = Config()

    print(f"{'字段':<28}{'值':<22}这本书在哪用到它")
    print("-" * 78)
    # 用得上的排在前面，剩下的按原顺序跟在后面，让读者看一眼整个文件的全貌。
    notes = {
        "hidden_size":           "第四章，词嵌入的维度",
        "num_hidden_layers":     "第五章，Decoder块堆几层",
        "vocab_size":            "第三章，词表大小",
        "tie_word_embeddings":   "第四章，权重共享",
        "rms_norm_eps":          "第五章，RMSNorm",
        "num_attention_heads":   "第六章，注意力头数",
        "num_key_value_heads":   "第六章，GQA",
        "intermediate_size":     "第八章，FFN",
        "rope_theta":            "第七章，RoPE",
        "rope_interleaved":      "第七章，RoPE的配对方式",
        "max_position_embeddings": "第七章，RoPE表的行数",
        "eos_token_id":          "第十章，生成循环靠它停下来",
    }
    named = [k for k in notes if k in cfg.raw]
    rest = [k for k in cfg.raw if k not in notes]
    for k in named + rest:
        print(f"{k:<28}{str(cfg.raw[k]):<22}{notes.get(k, '')}")

    print(f"\n一共{len(cfg.raw)}个字段，这一章新用到`intermediate_size`一个。")


if __name__ == "__main__":
    main()
