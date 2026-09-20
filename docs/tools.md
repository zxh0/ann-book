# 小工具

写书和写笔记的过程中顺手做的几个纯前端小工具，都是单个 HTML 文件，打开就能用。所有计算都在浏览器里完成，不需要后端，也不会把你的数据传到我这里来。

<script setup>
import { withBase } from 'vitepress'
import { tools } from './.vitepress/sidebar.json'
</script>

<div class="tool-cards">
  <a
    v-for="t in tools"
    :key="t.link"
    class="tool-card"
    :href="withBase(t.link)"
    target="_blank"
    rel="noopener"
  >
    <h3>{{ t.title }}</h3>
    <p class="tool-card-blurb">{{ t.blurb }}</p>
    <p v-if="t.note" class="tool-card-note">{{ t.note }}</p>
    <p class="tool-card-open">打开工具 →</p>
  </a>
</div>

::: tip 源码在仓库里
每个工具的源码都在 [tools/](https://github.com/zxh0/ann-book/tree/main/tools) 下，一个目录一个工具，各自带一份 README 说明原理。把 `index.html` 下载到本地双击打开，效果和在线版完全一样。
:::
