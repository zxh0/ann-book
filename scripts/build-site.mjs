// 从 notes/ 和 book/ 生成 VitePress 站点内容。
// notes/*.md 和 book/chapters/*.md 是唯一的内容来源，
// docs/notes/、docs/public/、sidebar.json 都是生成物。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const notesDir = path.join(root, 'notes')
const chaptersDir = path.join(root, 'book', 'chapters')
const docsDir = path.join(root, 'docs')
const outDir = path.join(docsDir, 'notes')
const publicDir = path.join(docsDir, 'public')

// 笔记文件名约定：YYYY-MM-DD-Slug.md
const NOTE_RE = /^(\d{4}-\d{2}-\d{2})-(.+)\.md$/
// 分章文件名约定：NN_slug.md
const CHAPTER_RE = /^\d+_(.+)\.md$/

/** 把 <img> 标签转成标准 Markdown 图片语法，并把 ./ 路径改成站点绝对路径。
 *  转成 Markdown 语法是为了让 VitePress 自动加上 base 前缀（裸 HTML 它不管）。
 *  Typora 写的 style="zoom:50%" 直接丢掉，图片宽度由正文栏宽兜住。 */
function normalizeImages(md, prefix) {
  return md.replace(/<img\s+([^>]*?)\/?>/g, (whole, attrs) => {
    const src = attrs.match(/src="([^"]+)"/)
    if (!src) return whole
    const alt = attrs.match(/alt="([^"]*)"/)
    // ./images/llm0/ime.png -> /images/llm0/ime.png
    const url = src[1].startsWith(prefix) ? '/' + src[1].slice(prefix.length) : src[1]
    return `![${alt ? alt[1] : ''}](${url})`
  })
}

function cleanup(md, prefix) {
  return normalizeImages(md, prefix)
    .replace(/<div style="page-break-after: always;"><\/div>/g, '') // 只对 PDF 有意义
    .replace(/\n{3,}/g, '\n\n')
    .trim()
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
  const body = cleanup(fs.readFileSync(path.join(notesDir, name), 'utf8'), './')
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

// 书还在草稿阶段，站点上只放一个 WIP 页面和目录，正文继续读仓库里的 Book.md
const bookToc = fs
  .readdirSync(chaptersDir)
  .sort()
  .filter((name) => CHAPTER_RE.test(name))
  .map((name) => firstHeadingOf(fs.readFileSync(path.join(chaptersDir, name), 'utf8')))
  .filter((title) => title && title !== '人人能懂的人工神经网络') // 书名页不算章节

// ---- 输出 ----

fs.mkdirSync(path.join(docsDir, '.vitepress'), { recursive: true })
fs.writeFileSync(
  path.join(docsDir, '.vitepress', 'sidebar.json'),
  JSON.stringify(
    {
      sidebar,
      // 首页要用：最新的几篇笔记（倒序）
      latest: [...notes].reverse().map(({ date, title, link }) => ({ date, title, link })),
      bookToc,
    },
    null,
    2
  ) + '\n'
)

fs.mkdirSync(publicDir, { recursive: true })
copyDir(path.join(notesDir, 'images'), path.join(publicDir, 'images'))
fs.copyFileSync(path.join(root, 'book', 'images', 'ann.jpg'), path.join(publicDir, 'ann.jpg'))

console.log(`生成 ${notes.length} 篇笔记 -> docs/notes/，书的目录 ${bookToc.length} 项`)
