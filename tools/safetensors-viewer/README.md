# safetensors 参数布局查看器

一个纯前端的单文件工具：给它一个 HF 模型仓库名或者一个 `model.safetensors` 的地址，
它把模型里每个张量在文件中的字节排布画出来。整个过程只下载文件头，通常是几十 KB。

打开 `index.html` 即可，没有构建步骤，也没有任何依赖。

## 它为什么不用下整个模型

safetensors 的文件结构很简单：

```
[ 8 字节 ][ N 字节 JSON 头 ][ 张量数据区 ]
   N 值      每个张量的 dtype / shape / data_offsets
```

开头 8 字节是小端 `uint64`，给出紧随其后的 JSON 头长度 N。JSON 头里记录了每个张量的名字、
`dtype`、`shape` 和它在数据区中的字节区间。所以两次 HTTP Range 请求就够了：

1. `Range: bytes=0-7` 拿到 N；
2. `Range: bytes=8-(8+N-1)` 拿到 JSON 头。

一个 256MB 的模型，实际传输 32KB，占 0.012%。工具里的“网络请求记录”会把这两次请求原样列出来。

如果服务器不认 Range 而是返回整个文件，代码会退化成流式读取，拿够字节后立刻中断连接，
不会真的把整个文件拉下来。

## 分片模型

分片的每个文件都是完整独立的 safetensors，各自带自己的 8 字节长度和 JSON 头，
头里的 `data_offsets` 是相对本片数据区起点的。第一个文件里没有全局头，
张量名到文件的映射在另一个普通 JSON 里（`model.safetensors.index.json` 的 `weight_map`）。
所以 N 个分片要发 2N 次请求。

分片一多，工具会：

- 并发 6 路读取，不再一片一片排队；
- 用确定进度的进度条显示 `12/40` 和累计传输字节；
- 随时可以点“取消”中断，已发出的请求会被 `AbortController` 掐掉。

## 三种加载方式

| 方式 | 说明 |
|------|------|
| HF 模型仓库 | 填 `用户名/模型名`，自动读 `config.json`、识别分片、需要授权的仓库可填 token。可切换到 `hf-mirror.com` 镜像 |
| 本地文件 | 浏览器只读文件开头，几十 GB 的文件也是秒开。可以一次多选分片文件 |
| 任意 URL | 指向 `.safetensors`，或者指向 `model.safetensors.index.json` 自动展开全部分片 |

## 五个视图

- **内存布局**：整个文件按字节顺序铺成带子，宽度正比于字节数，颜色区分参数角色。深灰是文件头，斜纹是空隙。
- **结构树**：按张量名里的点号做两层嵌套的矩形树图，面积可切换字节数或参数量，点击下钻。
- **逐层对比**：每层一行，把 Q/K/V/O、门控、升维、降维横向堆叠，一眼看出各层是否一致。比例尺按各层定，超出的行在右端画断口。
- **张量表**：可排序、可用正则过滤，支持导出 CSV。
- **原始 JSON 头**：直接看解析前的内容，可下载。

前三个视图都能导出 SVG 和 PNG，方便直接放进文章。

## 发布到 GitHub Pages

它是一个静态文件，放哪儿都行：

```bash
# 例如放进已有的 Pages 仓库
cp -r safetensors-viewer /path/to/pages-repo/
git add safetensors-viewer && git commit -m "Add safetensors viewer" && git push
```

然后访问 `https://<用户名>.github.io/<仓库名>/safetensors-viewer/`。
仓库设置里把 Pages 的 Source 指到对应分支和目录即可，不需要任何后端。

## URL 参数

可以用链接直接定位，写文章时很方便：

```
index.html?repo=HuggingFaceTB/SmolLM2-135M&view=layer
index.html?repo=Qwen/Qwen2.5-0.5B&rev=main&site=hf-mirror.com
index.html?url=https://example.com/model.safetensors&view=ribbon
```

`view` 取值：`ribbon`、`tree`、`layer`、`table`、`json`。

## 已知限制

- 远端文件要求服务器允许跨域（CORS）并支持 Range。huggingface.co 两者都支持；
  自建或第三方地址不一定，取不到时换成“本地文件”方式即可。
- 需要授权的仓库（比如 Llama 系列）要填 HF token，token 只在浏览器里用，直接发给所选站点。
- 张量角色是按名字猜的，覆盖了常见的 Llama、Qwen、GPT-2、BERT 命名；遇到没见过的命名会归到“其他”。
- 张量特别多的模型（比如细粒度 MoE，动辄几万个张量）绘制内存布局那一条会比较吃力，还没做合并简化。
