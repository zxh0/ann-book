// 从 notes/、books/ 和 tools/ 生成 VitePress 站点内容。
// notes/*.md、books/*/chapters/*.md 和 tools/*/ 是唯一的内容来源，
// docs/notes/、docs/ann4us/、docs/toyllm/、docs/public/、sidebar.json 都是生成物。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const notesDir = path.join(root, 'notes')
const booksDir = path.join(root, 'books')
const toolsDir = path.join(root, 'tools')
const docsDir = path.join(root, 'docs')
const outDir = path.join(docsDir, 'notes')
const publicDir = path.join(docsDir, 'public')

// 两本书，都是「写完一章发一章」：往 published 里加一行文件名，那一章就上站，
// 没列进来的章节只在目录页里露个标题，正文继续待在仓库里。
const BOOKS = [
  {
    // 《人人能懂的人工神经网络》：分章文件名是 NN_slug.md，章标题写成 ##，
    // 插图在 books/ann4us/images/ 下，正文里引作 ../images/xxx，封面另放在 aigc/ 下
    key: 'ann',
    dir: 'ann4us', // books/ 下的目录名
    route: 'ann4us', // 站点地址前缀，同时也是 docs/ 下的目录名
    chapterRe: /^\d+_(.+)\.md$/,
    skipTitle: '人人能懂的人工神经网络', // 书名页不算章节
    promote: true, // 章标题 ## 提到 #，正文各级标题跟着上移一级
    cover: { from: path.join('aigc', 'ann.png'), name: 'ann.png' },
    published: new Set(['02_ch00_basics.md', '03_ch01_neuron.md']),
  },
  {
    // 《自己动手写LLM推理引擎》：分章文件名是 chNN_slug.md，章标题本来就是 #，
    // _front.md / _preface.md / todo.md 不匹配 chapterRe，自然被挡在外面
    key: 'toyllm',
    dir: 'toyllm',
    route: 'toyllm',
    chapterRe: /^(ch\d+_.+)\.md$/,
    promote: false,
    cover: { from: path.join('aigc', 'front.png'), name: 'toyllm.png' },
    published: new Set(['ch01_overview.md', 'ch02_weights.md', 'ch03_tokenizer.md']),
  },
]

// 工具：tools/<目录>/ 下的纯静态单页，整个目录原样拷到 docs/public/tools/<目录>/，
// 站点地址就是 /tools/<目录>/。加一个工具 = 放一个目录进 tools/，再往这里加一行。
// README.md 是写给仓库读者的，不上站。
const TOOLS = [
  {
    dir: 'safetensors-viewer',
    title: 'safetensors 参数布局查看器',
    blurb:
      '给它一个 HF 模型仓库名或者 model.safetensors 的地址，它把模型里每个张量在文件中的字节排布画出来：' +
      '内存布局、结构树、逐层对比、张量表、原始 JSON 头五个视图，前三个能导出 SVG 和 PNG。',
    note: '只下载文件头，通常几十 KB。纯前端，模型数据不经过任何服务器。',
  },
]

const chaptersDirOf = (book) => path.join(booksDir, book.dir, 'chapters')

// 笔记文件名约定：YYYY-MM-DD-Slug.md
const NOTE_RE = /^(\d{4}-\d{2}-\d{2})-(.+)\.md$/

/** 把 <img> 标签转成标准 Markdown 图片语法，并把仓库里的相对路径改成站点绝对路径。
 *  rebases 是一串 [前缀, 换成什么]，第一个匹配上的生效：笔记只有 ./ -> /，
 *  书有两条，插图 ../images/ -> /images/<书>/ 和 AI 生成的章首图 ../aigc/ -> /aigc/<书>/。
 *  转成 Markdown 语法是为了让 VitePress 自动加上 base 前缀（裸 HTML 它不管）。
 *  Typora 写的 style="zoom:50%" 直接丢掉，图片宽度由正文栏宽兜住。 */
function normalizeImages(md, rebases) {
  const rebase = (url) => {
    for (const [from, to] of rebases) {
      if (url.startsWith(from)) return to + url.slice(from.length)
    }
    return url
  }
  return md
    .replace(/<img\s+([^>]*?)\/?>/g, (whole, attrs) => {
      const src = attrs.match(/src="([^"]+)"/)
      if (!src) return whole
      const alt = attrs.match(/alt="([^"]*)"/)
      return `![${alt ? alt[1] : ''}](${rebase(src[1])})`
    })
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (whole, alt, url) => `![${alt}](${rebase(url)})`)
}

function cleanup(md, rebases) {
  return normalizeImages(md, rebases)
    .replace(/<div style="page-break-after: always;"><\/div>/g, '') // 只对 PDF 有意义
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** 分章文件里章标题是 ##，单独成页时要提到 #，正文各级标题跟着上移一级。
 *  代码块里的 # 是注释，不能动。 */
function promoteHeadings(md) {
  let inFence = false
  return md
    .split('\n')
    .map((line) => {
      if (/^\s*```/.test(line)) inFence = !inFence
      return !inFence && /^#{2,6} /.test(line) ? line.slice(1) : line
    })
    .join('\n')
}

/** 收集页面内的 ## 小节，用于侧边栏展开。 */
function subsectionsOf(md) {
  const items = []
  for (const line of md.split('\n')) {
    const h2 = line.match(/^## (.+?)\s*$/)
    if (h2) items.push(h2[1])
  }
  return items
}

function firstHeadingOf(md) {
  return md.match(/^#{1,2} (.+?)\s*$/m)?.[1]
}

// 复刻 VitePress 默认的标题锚点规则（来自 @mdit-vue/shared 的 slugify，
// 它被打包进 vitepress 里了，没法直接 import，所以照抄一份）。
// 注意 NFKD 会把全角字符归一成半角，所以「CSA（Compressed Sparse Attention）」
// 里的（）会跟半角括号一样变成连字符，不能想当然地当成普通中文字符留着。
const rControl = /[\u0000-\u001f]/g
const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g
const rCombining = /[\u0300-\u036F]/g

function anchor(text) {
  return text
    .normalize('NFKD')
    .replace(rCombining, '')
    .replace(rControl, '')
    .replace(rSpecial, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function copyDir(src, dest) {
  fs.rmSync(dest, { recursive: true, force: true })
  fs.cpSync(src, dest, { recursive: true, filter: (f) => !f.endsWith('.DS_Store') })
}

// ---- 笔记 ----

fs.rmSync(outDir, { recursive: true, force: true })
fs.mkdirSync(outDir, { recursive: true })

const notes = []

for (const name of fs.readdirSync(notesDir).sort()) {
  const m = name.match(NOTE_RE)
  if (!m) continue

  const slug = name.replace(/\.md$/, '')
  const body = cleanup(fs.readFileSync(path.join(notesDir, name), 'utf8'), [['./', '/']])
  fs.writeFileSync(path.join(outDir, `${slug}.md`), body + '\n')

  const link = `/notes/${slug}`
  const subs = subsectionsOf(body)
  notes.push({
    date: m[1],
    title: firstHeadingOf(body) ?? slug,
    link,
    collapsed: subs.length > 0 ? true : undefined,
    items: subs.map((t) => ({ text: t, link: `${link}#${anchor(t)}` })),
  })
}

if (notes.length === 0) throw new Error(`notes/ 下没有找到符合 YYYY-MM-DD-Slug.md 的文件`)

// 侧边栏按时间正序（跟 README 一致，系列文章前后有依赖），首页列表倒序
const sidebar = notes.map(({ date, title, link, collapsed, items }) => ({
  text: title,
  link,
  collapsed,
  items,
}))

// ---- 书 ----

// 书按章上站：published 里的章节生成页面，其余的只在目录页里列个标题。
// 页面文件保留 chapters/ 的原始文件名，再用 VitePress 的 rewrites 映射到干净的
// 地址（/ann4us/ch00-basics、/toyllm/ch01-overview），这样 editLink 拿到的 filePath
// 还能直接对回各自 chapters/ 下的源文件。
const rewrites = {}

function buildBook(book) {
  const bookOutDir = path.join(docsDir, book.route)
  fs.rmSync(bookOutDir, { recursive: true, force: true })
  fs.mkdirSync(bookOutDir, { recursive: true })

  const chapters = []

  for (const name of fs.readdirSync(chaptersDirOf(book)).sort()) {
    if (!book.chapterRe.test(name)) continue

    const raw = fs.readFileSync(path.join(chaptersDirOf(book), name), 'utf8')
    const title = firstHeadingOf(raw)
    if (!title || title === book.skipTitle) continue

    if (!book.published.has(name)) {
      chapters.push({ title })
      continue
    }

    const slug = name.replace(book.chapterRe, '$1').replace(/_/g, '-')
    const cleaned = cleanup(raw, [
      ['../images/', `/images/${book.route}/`],
      ['../aigc/', `/aigc/${book.route}/`],
    ])
    const body = book.promote ? promoteHeadings(cleaned) : cleaned
    fs.writeFileSync(path.join(bookOutDir, name), body + '\n')
    rewrites[`${book.route}/${name}`] = `${book.route}/${slug}.md`

    const link = `/${book.route}/${slug}`
    const subs = subsectionsOf(body)
    chapters.push({
      title,
      link,
      collapsed: subs.length > 0 ? true : undefined,
      items: subs.map((t) => ({ text: t, link: `${link}#${anchor(t)}` })),
    })
  }

  if (chapters.length === 0) throw new Error(`${chaptersDirOf(book)} 下没有找到分章文件`)

  return {
    // 目录页要列全部章节，侧边栏只放已经上站的
    toc: chapters.map(({ title, link }) => ({ text: title, link })),
    sidebar: chapters
      .filter((c) => c.link)
      .map(({ title, link, collapsed, items }) => ({ text: title, link, collapsed, items })),
  }
}

const books = Object.fromEntries(BOOKS.map((b) => [b.key, buildBook(b)]))

// ---- 输出 ----

fs.mkdirSync(path.join(docsDir, '.vitepress'), { recursive: true })
fs.writeFileSync(
  path.join(docsDir, '.vitepress', 'sidebar.json'),
  JSON.stringify(
    {
      sidebar,
      // 首页要用：最新的几篇笔记（倒序）
      latest: [...notes].reverse().map(({ date, title, link }) => ({ date, title, link })),
      books,
      // 首页和 /tools 页要用
      tools: TOOLS.map(({ dir, title, blurb, note }) => ({
        title,
        blurb,
        note,
        link: `/tools/${dir}/`,
      })),
      rewrites,
    },
    null,
    2
  ) + '\n'
)

fs.mkdirSync(publicDir, { recursive: true })
copyDir(path.join(notesDir, 'images'), path.join(publicDir, 'images'))
// 每本书的插图各占 images/ 下的一个子目录，跟笔记的插图分开，免得几边目录重名；
// aigc/ 下 AI 生成的章首图同理，另占 aigc/ 下的一个子目录
for (const book of BOOKS) {
  copyDir(path.join(booksDir, book.dir, 'images'), path.join(publicDir, 'images', book.route))
  copyDir(path.join(booksDir, book.dir, 'aigc'), path.join(publicDir, 'aigc', book.route))
  fs.copyFileSync(path.join(booksDir, book.dir, book.cover.from), path.join(publicDir, book.cover.name))
}

const toolsOutDir = path.join(publicDir, 'tools')
fs.rmSync(toolsOutDir, { recursive: true, force: true })
for (const tool of TOOLS) {
  const src = path.join(toolsDir, tool.dir)
  if (!fs.existsSync(path.join(src, 'index.html')))
    throw new Error(`tools/${tool.dir}/ 下没有 index.html`)
  fs.cpSync(src, path.join(toolsOutDir, tool.dir), {
    recursive: true,
    filter: (f) => !f.endsWith('.DS_Store') && !f.endsWith('README.md'),
  })
}

console.log(
  `生成 ${notes.length} 篇笔记 -> docs/notes/，` +
    BOOKS.map(
      (b) => `${books[b.key].sidebar.length}/${books[b.key].toc.length} 章正文 -> docs/${b.route}/`
    ).join('，') +
    `，${TOOLS.length} 个工具 -> docs/public/tools/`
)
