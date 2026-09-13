---
layout: home

hero:
  name: 学 AI，从零开始
  text: 两本书，一个笔记系列
  tagline: 一本只用中学数学，从一个人工神经元讲到大语言模型；一本从零手写 LLM 推理引擎，跑在笔记本的 CPU 上。
  actions:
    - theme: brand
      text: 读笔记
      link: /notes
    - theme: alt
      text: 人工神经网络（WIP）
      link: /ann4us
    - theme: alt
      text: LLM推理引擎（WIP）
      link: /toyllm
    - theme: alt
      text: GitHub
      link: https://github.com/zxh0/ann-book

features:
  - title: 图解优先
    details: 能画图的地方都画图。注意力、KV Cache、MoE 路由这些绕来绕去的东西，一张图往往比一段话讲得清楚。
  - title: 盯着公式读论文
    details: DeepSeek、Kimi、MiniMax、LongCat 的注意力和 MoE 变体，逐个公式拆开看，符号在全系列里尽量统一。
  - title: 从零手写
    details: 推理引擎那本书不用 transformers 的建模代码，分词、RoPE、GQA、KV Cache 全部自己写，每个数字都能和 HuggingFace 对上。
  - title: 原汁原味
    details: 文字都是自己敲的，图都是自己画的。只用 AI 检查错别字和病句，引用 AI 的片段会标出来。
---

<script setup>
import { withBase } from 'vitepress'
import { books } from './.vitepress/sidebar.json'

const cards = [
  {
    title: '人人能懂的人工神经网络',
    subtitle: 'ANN for the Rest of Us',
    link: '/ann4us',
    cover: '/ann.jpg',
    blurb: '一本入门书，从最简单的人工神经元入手，循序渐进搭起完整的知识体系。只要有中学数学基础就能跟下来。',
    book: books.ann,
  },
  {
    title: '自己动手写LLM推理引擎',
    subtitle: 'Build a Toy LLM Inference Engine',
    link: '/toyllm',
    cover: '/toyllm.png',
    blurb: '用 Python + PyTorch 从零手写一个 LLM 推理引擎，借此理解 LLM 到底是怎么推理的。目标模型 SmolLM2-135M，跑在笔记本 CPU 上。',
    book: books.toyllm,
  },
]
</script>

## 两本书

两本都还在写，都是写完一章发一章，下面能点开的就是已经发布的章节。

<div class="book-cards">
  <a v-for="c in cards" :key="c.link" class="book-card" :href="withBase(c.link)">
    <img :src="withBase(c.cover)" :alt="c.title" />
    <div class="book-card-body">
      <h3>{{ c.title }}</h3>
      <p class="book-card-sub">{{ c.subtitle }}</p>
      <p class="book-card-blurb">{{ c.blurb }}</p>
      <p class="book-card-progress">
        目录 {{ c.book.toc.length }} 章，已发布 {{ c.book.sidebar.length }} 章
      </p>
    </div>
  </a>
</div>

## 一个笔记系列

**[图解 LLM 系列笔记](/notes)**：一篇一个主题，解读 LLM 相关的论文和核心公式。从 DeepSeek 的 MLA、Flash Attention、位置编码，一路写到 Kimi Delta Attention、LongCat Sparse Attention。系列里也有一篇[大白话版](/notes/2026-08-30-LLM)，不讲公式不贴代码，从拼音输入法说起，写给完全没有基础的读者。

## 反馈贡献

我还在慢慢学习中，笔记和书里难免有错误和疏漏。如果你发现任何问题，或者有优化建议，非常欢迎[提交 Issue](https://github.com/zxh0/ann-book/issues)，具体包括但不限于：

- 错别字、语病或表述模糊的内容
- 公式推导和概念解释里的错误、逻辑漏洞
- 认为应当补充的重要知识点
- 能提升阅读体验的排版、配图建议

你的每一份反馈，都是让这些内容更完善、更易读的重要帮助。
