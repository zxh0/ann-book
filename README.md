# 人人能懂的人工神经网络

在线阅读：<https://zxh0.github.io/ann-book/>

书：

* [人人能懂的人工神经网络](book/Book.md)（WIP）

笔记：

* [图解DeepSeek-V4(Pro)核心公式](notes/2026-06-25-DSv4.md)
* [图解DeepSeek-V2 MLA公式](notes/2026-06-27-DSv2MLA.md)
* [图解Flash Attention核心原理](notes/2026-07-02-FA.md)
* [图解位置编码](notes/2026-07-05-PE.md)
* [图解MiniMax Sparse Attention](notes/2026-07-11-MSA.md)
* [图解DeepSeek训练过程](notes/2026-07-14-DSTrain.md)
* [图解Kimi K3 Stable LatentMoE](notes/2026-07-30-LatentMoE.md)
* [图解Kimi Attention Residuals](notes/2026-08-01-AttnRes.md)
* [图解Kimi Delta Attention](notes/2026-08-08-KDA.md)
* [图解KDA续（数学知识补充）](notes/2026-08-12-LAMath.md)
* [图解LongCat Sparse Attention](notes/2026-08-22-LSA.md)
* [图解N-gram Embedding](notes/2026-08-28-NgE.md)
* [图解LLM（大白话版）](notes/2026-08-30-LLM.md)

## 站点

站点用 [VitePress](https://vitepress.dev/) 生成，推送到 `main` 后由
[GitHub Actions](.github/workflows/deploy.yml) 自动部署。本地预览：

```sh
npm install
npm run dev      # http://localhost:5173/ann-book/
```

`scripts/build-site.mjs` 会把 `notes/*.md` 转成 `docs/notes/`、把 `notes/images/` 复制到
`docs/public/`，并生成侧边栏，这些都是生成物，不进版本库。新写的笔记只要文件名符合
`YYYY-MM-DD-Slug.md`，就会自动出现在站点上，不用改配置。

书还在草稿阶段，站点上只有 `docs/book.md` 那个 WIP 页面（目录从 `book/chapters/` 自动读取），
正文暂时只在仓库里。
