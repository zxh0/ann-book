# 人人能懂的人工神经网络

<script setup>
import { bookToc } from './.vitepress/sidebar.json'
</script>

> ANN for the Rest of Us

::: warning 施工中（WIP）
本书还在很早期的草稿阶段，可能有很多逻辑混乱、甚至胡言乱语的地方，我会慢慢改进。所以暂时还没在这个站点上发布，想提前看的话可以直接读仓库里的
[Book.md](https://github.com/zxh0/ann-book/blob/main/book/Book.md)（全书单文件），或者按章节读
[book/chapters/](https://github.com/zxh0/ann-book/tree/main/book/chapters)。
:::

![人人能懂的人工神经网络](/ann.jpg)

## 这本书想做什么

智能体（Agent）、技能（Skills）、大语言模型（LLM）、深度学习（DL）…… 随着人工智能浪潮席卷全球，这些专业词汇大家早已耳熟能详。但支撑这一切的底层基石，早在八十余年前便已悄然萌芽，它就是**人工神经网络**（Artificial Neural Network，简称 ANN）。

也许你也曾试着去了解过相关资料，可里面满是概率论、线性代数、微积分等知识，还有一大堆复杂难懂的数学公式。一看到反向传播、损失函数、梯度消失这些概念，就觉得头都大了，于是默默放弃。

但其实，人工神经网络的整体思路非常容易理解。我们完全可以把少数过于艰深的细节暂时当成「黑盒子」，不用死磕每一个数学推导。**只要掌握中学阶段的基础数学知识**，再配合循序渐进、由浅入深的学习节奏，普通读者也完全能够读懂并理解人工神经网络。

这本书正是为实现这一目标而创作：从最简单的人工神经元入手，循序渐进搭建完整的知识体系。不堆砌复杂公式，不生硬灌输晦涩理论，带你揭开人工智能底层原理的神秘面纱。

## 目录

章节标题已经定下来了，正文还在填坑：

<ul class="book-toc">
  <li v-for="t in bookToc" :key="t">{{ t }}</li>
</ul>
