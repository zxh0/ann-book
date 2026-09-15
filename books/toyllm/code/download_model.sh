#!/bin/sh
# 下载 SmolLM2-135M 的权重和分词器，放到 code/models/SmolLM2-135M/ 下面。
# 十个文件，一共约 272 MB，和 HF 仓库里的内容一一对应。
# 这个目录不入库，所以拿到代码之后要先跑一遍。
#
#     cd code
#     ./download_model.sh
#
# 已经下过的文件会跳过，中途断了再跑一遍会接着下。
# 连不上 HuggingFace 的话，换个镜像站：
#
#     HF_ENDPOINT=https://hf-mirror.com ./download_model.sh
set -eu

REPO=${REPO:-HuggingFaceTB/SmolLM2-135M}
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
DIR=$(cd "$(dirname "$0")" && pwd)/models/${REPO##*/}

echo "仓库   $REPO"
echo "来源   $ENDPOINT"
echo "目录   $DIR"
echo

mkdir -p "$DIR"

# 仓库里有什么就下什么，具体用不用以后再说。最大的是 model.safetensors，269 MB，
# 其余九个加起来 3 MB 出头。第二、三章会解释它们分别是干嘛的。
#
# vocab.json 和 merges.txt 是老式（GPT-2 风格）的分词器文件，内容和 tokenizer.json
# 重复：后者把词表、合并规则和预处理管线打包进了一个文件。但 merges.txt 是纯文本，
# 一行一条合并规则，肉眼就能读，讲 BPE 的时候比啃 2 MB 的 JSON 舒服。
for f in config.json \
         model.safetensors \
         generation_config.json \
         tokenizer.json \
         tokenizer_config.json \
         special_tokens_map.json \
         vocab.json \
         merges.txt \
         README.md \
         .gitattributes; do
    if [ -f "$DIR/$f" ]; then
        # 花括号不能省：紧跟中文的 $f 会被 shell 当成变量名的一部分
        echo "跳过 ${f}（已存在）"
        continue
    fi

    echo "下载 $f"
    # -L 跟随跳转，权重本体存在 CDN 上；-C - 断点续传；--fail 让 HTTP 错误变成
    # 非零退出码，而不是把一页 HTML 错误信息存成权重文件。
    # 注意是 /resolve/main/，不是 /raw/main/，后者给的是 LFS 指针而不是文件本身。
    if ! curl -L -C - --fail --progress-bar \
         -o "$DIR/$f.part" "$ENDPOINT/$REPO/resolve/main/$f"; then
        echo
        echo "下载失败。两条常见的路："
        echo "  1. 换镜像站：HF_ENDPOINT=https://hf-mirror.com ./download_model.sh"
        echo "  2. 公司网络的代理挡的：先 unset HTTPS_PROXY https_proxy"
        # 只有真下到一半才有 .part，HTTP 报错（比如仓库名写错）是一个字节都没下
        if [ -f "$DIR/$f.part" ]; then
            echo
            echo "下了一半的东西在 $DIR/$f.part，再跑一遍会接着下。"
        fi
        exit 1
    fi
    # 下完了才改成正式名字。中途断掉不会留下一个看着是好的、其实缺一截的文件。
    mv "$DIR/$f.part" "$DIR/$f"
done

echo
# -A 而不是 -l：清单里有 .gitattributes，点文件不加 -A 就看不见
ls -lA "$DIR"
echo
echo "下一步：uv run python book/ch01/hello_5l.py"
