# 图解 LLM 系列笔记

<script setup>
import { withBase } from 'vitepress'
import { latest } from './.vitepress/sidebar.json'
</script>

这个系列的每一篇都围绕一个主题，解读 LLM 相关的论文和核心公式，能画图的地方都画图。由于这些论文所使用的字母和符号并不统一，为了便于阅读，我在系列里尽量使用统一的符号，并在第一次出现时给出说明。

<ul class="note-list">
  <li v-for="n in latest" :key="n.link">
    <span class="date">{{ n.date }}</span>
    <a :href="withBase(n.link)">{{ n.title }}</a>
  </li>
</ul>

::: tip 从哪读起
如果你已经了解 LLM 的大部分术语，随便挑一篇感兴趣的看就行。如果是零基础，建议先看
[图解LLM（大白话版）](/notes/2026-08-30-LLM)，那一篇不讲数学公式，也不贴代码，从拼音输入法说起。
:::
