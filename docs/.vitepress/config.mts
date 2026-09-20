import { defineConfig } from 'vitepress'
import { sidebar, latest, books, rewrites } from './sidebar.json'

// sidebar.json 由 scripts/build-site.mjs 从 notes/ 和 books/ 生成
const newest = latest[0].link

export default defineConfig({
  title: '学 AI，从零开始',
  description:
    '两本还在写的书《人人能懂的人工神经网络》《自己动手写LLM推理引擎》，和一个「图解 LLM」笔记系列。',
  lang: 'zh-CN',
  base: '/ann-book/',
  cleanUrls: true,
  lastUpdated: true,
  // docs/ann4us/ 和 docs/toyllm/ 下的页面文件名沿用各自 chapters/ 的原名，这里映射成干净的地址
  rewrites,
  head: [
    ['meta', { property: 'og:title', content: '学 AI，从零开始' }],
    ['meta', { property: 'og:image', content: 'https://zxh0.github.io/ann-book/ann.png' }],
  ],

  markdown: {
    lineNumbers: false,
    image: { lazyLoading: true },
    // 笔记里有 $...$ 和 $$...$$ 公式，需要 markdown-it-mathjax3
    math: true,
  },

  themeConfig: {
    outline: { level: [2, 3], label: '本页目录' },

    nav: [
      { text: '笔记', link: '/notes', activeMatch: '^/notes' },
      { text: '最新一篇', link: newest },
      {
        text: '书',
        activeMatch: '^/(ann4us|toyllm)',
        items: [
          { text: '人人能懂的人工神经网络', link: '/ann4us' },
          { text: '自己动手写LLM推理引擎', link: '/toyllm' },
        ],
      },
      { text: '工具', link: '/tools', activeMatch: '^/tools' },
      { text: '提问纠错', link: 'https://github.com/zxh0/ann-book/issues' },
    ],

    sidebar: {
      '/notes': [{ text: '图解 LLM 系列笔记', link: '/notes', items: sidebar }],
      '/ann4us': [
        {
          text: '人人能懂的人工神经网络',
          items: [{ text: '进度和目录', link: '/ann4us' }, ...books.ann.sidebar],
        },
      ],
      '/toyllm': [
        {
          text: '自己动手写LLM推理引擎',
          items: [{ text: '进度和目录', link: '/toyllm' }, ...books.toyllm.sidebar],
        },
      ],
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/zxh0/ann-book' }],

    editLink: {
      // 站点里的 /notes/xxx 和仓库里的 notes/xxx.md 一一对应；
      // 两本书的页面文件名都保留了各自 chapters/ 的原名，把站点地址前缀换成
      // books/<目录>/chapters 就是源文件。
      // 注意：这个函数和下面的分词器一样，会被序列化后送到浏览器执行，必须自包含，
      // 不能引用 sidebar.json 里的东西，所以下面这行书名是 build-site.mjs 里
      // BOOKS 的手抄版（站点地址前缀和 books/ 下的目录名同名），加书时两边一起改。
      pattern: ({ filePath }) => {
        const edit = (p) => `https://github.com/zxh0/ann-book/edit/main/${p}`
        if (filePath.startsWith('notes/')) return edit(filePath)
        for (const dir of ['ann4us', 'toyllm']) {
          if (filePath.startsWith(`${dir}/`))
            return edit(`books/${dir}/chapters/${filePath.slice(dir.length + 1)}`)
        }
        return 'https://github.com/zxh0/ann-book/issues'
      },
      text: '在 GitHub 上纠错',
    },

    search: {
      provider: 'local',
      options: {
        // 默认分词器按空白切词，对中文几乎无效。这里把中文按字拆开建索引，
        // 配合 AND 组合，搜「注意力」= 同时包含这三个字的页面。
        // 注意：这个函数会被序列化后送到浏览器执行，必须自包含。
        miniSearch: {
          options: {
            tokenize: (text: string) =>
              text
                .split(/[^\p{L}\p{N}_]+/u)
                .flatMap((w) => (/[一-鿿]/.test(w) ? w.split('') : [w]))
                .filter(Boolean),
            processTerm: (term: string) => term.toLowerCase(),
          },
          searchOptions: {
            combineWith: 'AND',
            prefix: true,
            boost: { title: 4, text: 2, titles: 1 },
          },
        },
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详情',
            resetButtonTitle: '清除',
            noResultsText: '没有找到',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    lastUpdated: { text: '最后更新' },
    outlineTitle: '本页目录',

    footer: {
      message: '文字和插图均为原创，转载请注明出处',
      copyright: 'Copyright © 2026 zxh0',
    },
  },
})
