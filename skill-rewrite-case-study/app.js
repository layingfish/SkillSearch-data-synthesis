// Case-study viewer: loads data/index.json, then data/<slug>.json per selected case.
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const cache = {};

async function getJSON(p) {
  if (!cache[p]) cache[p] = fetch(p).then(r => { if (!r.ok) throw new Error(p + " " + r.status); return r.json(); });
  return cache[p];
}

function skillCard(s, rank, otherIds) {
  const shared = otherIds.has(s.id) ? '<span class="shared">两边都检索到</span>' : "";
  return `<div class="sk">
    <div><span class="rk">#${rank}</span><span class="nm">${esc(s.name)}</span>${shared}</div>
    <div class="id">${esc(s.id)}</div>
    <div class="ds">${esc(s.desc)}</div>
    <details><summary>展开完整内容（${s.body.length.toLocaleString()} 字符）</summary><pre>${esc(s.body)}</pre></details>
  </div>`;
}

function retrievalCol(cls, title, list, other) {
  const ids = new Set(other.map(s => s.id));
  return `<div class="col ${cls}"><h3>${title}</h3>${list.map((s, i) => skillCard(s, i + 1, ids)).join("")}</div>`;
}

// mark lines of `prompt` that do not occur in `base` (the plain GR prompt)
function diffPrompt(prompt, base) {
  if (base === null) return esc(prompt);
  const baseLines = new Set(base.split("\n"));
  return prompt.split("\n").map(l => baseLines.has(l) ? esc(l) : `<span class="add">${esc(l) || "&nbsp;"}</span>`).join("\n");
}

const NOTES = {
  "GR": "提示词模板与 SkillRouter 条件完全相同，只是候选来自 SkillGR 的 top-10。",
  "DR": "提示词模板与 SkillGR 条件完全相同，只是候选来自 SkillRouter 的 top-10。",
  "GR+SID_RAW": "候选与 SkillGR 条件相同；每个候选后多一行 semantic_id（64 位 OPQ 码），候选列表后多一段 ID 含义说明（黄色为相对 SkillGR 提示词新增的行）。",
  "GR+SID_GROUPS": "候选与 SkillGR 条件相同；候选列表后多一段候选之间共享码位数的统计（黄色为相对 SkillGR 提示词新增的行）。",
};

function rewritePanel(r, basePrompt) {
  const words = r.output.split(/\s+/).filter(Boolean).length;
  return `<div class="cfg">
      <b>改写模型</b><span>deepseek-v4-flash（thinking 关闭）</span>
      <b>temperature</b><span>0.3</span>
      <b>max_tokens</b><span>3000</span>
      <b>候选来源</b><span>${r.variant === "DR" ? "SkillRouter top-10" : "SkillGR top-10"}</span>
    </div>
    <div class="note">${NOTES[r.variant]}</div>
    <details class="pr"><summary>改写模型接收的提示词（system + user，${(r.system.length + r.prompt.length).toLocaleString()} 字符）</summary>
      <div class="meta">system</div><pre>${esc(r.system)}</pre>
      <div class="meta">user</div><pre>${r.variant.startsWith("GR+SID") ? diffPrompt(r.prompt, basePrompt) : esc(r.prompt)}</pre>
    </details>
    <h3 style="margin:14px 0 4px;font-size:14px">改写结果（${words} 词）</h3>
    <pre>${esc(r.output)}</pre>`;
}

async function showCase(c) {
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("on", a.dataset.slug === c.slug));
  const box = $("#case");
  box.innerHTML = `<section><div class="meta">加载中…</div></section>`;
  const d = await getJSON(`data/${c.slug}.json`);
  const base = d.rewrites.find(r => r.variant === "GR").prompt;
  box.innerHTML = `
    <section><h2>${esc(d.key.split(":")[1])} <span class="meta">${esc(d.bench)}</span></h2>
      <details${d.task.length < 1500 ? " open" : ""}><summary>任务描述（${d.task.length.toLocaleString()} 字符）</summary><pre>${esc(d.task)}</pre></details></section>
    <section><h2>检索结果 top-10</h2><div class="cols">
      ${retrievalCol("gr", "SkillGR", d.GR, d.DR)}${retrievalCol("dr", "SkillRouter", d.DR, d.GR)}
    </div></section>
    <section><h2>改写结果</h2>
      <div class="tabs">${d.rewrites.map((r, i) => `<button data-i="${i}">${esc(r.label)}</button>`).join("")}</div>
      <div id="rw"></div></section>`;
  const tabs = box.querySelectorAll(".tabs button");
  const pick = i => { tabs.forEach(b => b.classList.toggle("on", +b.dataset.i === i)); $("#rw").innerHTML = rewritePanel(d.rewrites[i], base); };
  tabs.forEach(b => b.onclick = () => pick(+b.dataset.i));
  pick(0);
  history.replaceState(null, "", "#" + c.slug);
}

(async () => {
  const idx = await getJSON("data/index.json");
  $("#sampleRule").textContent = `抽样：随机种子 ${idx.sample.seed}；SkillsBench 从 49 个执行任务中均匀抽 5 个，SRA-Bench 每个执行过的数据集抽 2 题，共 ${idx.cases.length} 个 query，全部展示。`;
  const groups = {};
  idx.cases.forEach(c => (groups[c.bench.split(" / ")[0]] ||= []).push(c));
  $("#nav").innerHTML = Object.entries(groups).map(([g, cs]) =>
    `<h3>${esc(g)}</h3>` + cs.map(c => `<a href="#${c.slug}" data-slug="${c.slug}" title="${esc(c.preview)}">${esc(c.title)}</a>`).join("")).join("");
  document.querySelectorAll("nav a").forEach(a => a.onclick = e => { e.preventDefault(); showCase(idx.cases.find(c => c.slug === a.dataset.slug)); });
  const want = location.hash.slice(1);
  showCase(idx.cases.find(c => c.slug === want) || idx.cases[0]);
})().catch(e => { $("#case").innerHTML = `<section>加载失败：${esc(e.message)}</section>`; });
