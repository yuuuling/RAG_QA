# -*- coding: utf-8 -*-
"""
RAG for 《海外旅行不便險條款.pdf》
- Parser: pypdf
- Index: faiss-cpu if available, else NumPy cosine
- Generator: Ollama (env OLLAMA_MODEL, default qwen2:7b)
- Topic routing: 全份 PDF，多主題與全域檢索
- Hard guard: 主題 allowlist + 條文子類別 mapping + 30–32 標題覆寫
- Deterministic composers:
    * 班機延誤：第30(承保) / 第31(不保) / 第32(文件)，固定「航空業者」
    * 行李延誤：第36/37/38（保/不保/文件）
    * 行李損失：第39/40/41/42/43/44（保/不保(物/事)/處理/文件/追回）
    * 「不保/不可理賠/除外」查詢：依問題語義選定險種；未指明則輸出三大險種的不保總覽
- 不輸出頁碼
"""

import os
import re
import json
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "8"))
MIN_SIM = float(os.getenv("MIN_SIM", "0.25"))
MAX_SNIPPETS_CHARS = int(os.getenv("MAX_SNIPPETS_CHARS", "8000"))

# ---------- Optional FAISS ----------
_USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    _USE_FAISS = False

# ---------- Data ----------
@dataclass
class Chunk:
    text: str
    chapter: str
    article_no: str
    article_title: str
    page_from: int
    page_to: int
    tags: List[str]

# ---------- Regex ----------
CN_NUM = "〇一二三四五六七八九十百千"
RE_CHAP = re.compile(r"^第[" + CN_NUM + r"]+章")
RE_ART  = re.compile(r"^第([" + CN_NUM + r"]+)條\s*(.*)$")

def normalize(s: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", s).strip()

# ---------- Auto-tags ----------
def auto_tags(text: str) -> List[str]:
    mapping = {
        "旅程取消": ["旅程取消","取消","罷工","暴動","病危","死亡","火災","天災","退款"],
        "班機延誤": ["班機延誤","延誤期間","延誤證明","航空業者","替代班機","四小時","航班延誤","轉機"],
        "旅程更改": ["旅程更改","更改行程","護照遺失","交通意外"],
        "行李延誤": ["行李延誤","六小時","未領得","領取單"],
        "行李損失": ["行李損失","竊盜","強盜","搶奪","託運","毀損","遺失","事故與損失證明","追回處理"],
        "文件損失": ["旅行文件","護照","遺失","竊盜","搶奪","報案證明"],
        "改降機場": ["改降非原定機場","改降"],
        "劫機": ["劫機","最高補償日數"],
        "食品中毒": ["食品中毒","診斷證明書"],
        "現金竊盜": ["現金竊盜","現金","匯票","旅行支票"],
        "信用卡盜用": ["信用卡","盜刷","掛失","止付"],
        "居家竊盜": ["居家竊盜","住宅"],
        "租車事故": ["租車","交通事故","國際駕照"],
        "特殊活動取消": ["特殊活動","表演","遊樂場","滑雪","取消"],
        "賽事取消": ["賽事取消","天災","主辦單位"],
        "行動電話被竊": ["行動電話","手機","被竊","搶奪"],
        "急難救助": ["未成年子女送回","親友探視","醫療轉送","遺體運送","搜索救助","高山症"],
        "費率": ["短期費率","內插法","費率係數"],
    }
    out = set()
    for tag, kws in mapping.items():
        if any(kw in text for kw in kws):
            out.add(tag)
    return sorted(out)

# ---------- PDF -> 條為單位 ----------
def pdf_to_chunks(pdf_path: str) -> List[Chunk]:
    reader = PdfReader(pdf_path)
    lines = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        for ln in txt.splitlines():
            ln = normalize(ln)
            if ln:
                lines.append((i, ln))

    chunks: List[Chunk] = []
    cur_chap = ""
    cur_art_no = ""
    cur_art_title = ""
    buf: List[Tuple[int,str]] = []

    def flush():
        nonlocal buf, chunks, cur_chap, cur_art_no, cur_art_title
        if not cur_art_no or not buf:
            buf = []
            return
        page_from = buf[0][0]; page_to = buf[-1][0]
        text = "\n".join([b for _, b in buf])
        chunks.append(Chunk(
            text=text,
            chapter=cur_chap,
            article_no=cur_art_no,
            article_title=cur_art_title,
            page_from=page_from,
            page_to=page_to,
            tags=auto_tags(text),
        ))
        buf = []

    for pageno, line in lines:
        if RE_CHAP.match(line):
            flush()
            cur_chap = line
            continue
        m = RE_ART.match(line)
        if m:
            flush()
            cn = m.group(1)
            cur_art_no = f"第{cn}條"
            cur_art_title = m.group(2).strip()
            buf = [(pageno, line)]
        else:
            if cur_art_no:
                buf.append((pageno, line))
    flush()
    return chunks

# ---------- Vector index ----------
class VectorIndex:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.emb = None
        self.meta: List[Chunk] = []
        self.index = None

    def build(self, chunks: List[Chunk]):
        self.meta = chunks
        texts = [c.text for c in chunks]
        self.emb = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        if _USE_FAISS:
            d = self.emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.emb)
        else:
            self.index = None

    def _numpy_search(self, q_emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = (self.emb @ q_emb.T).reshape(-1)
        idx = np.argsort(-sims)[:topk]
        return sims[idx], idx

    def search(self, query: str, topk: int) -> List[Tuple[float, Chunk]]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        if _USE_FAISS:
            D, I = self.index.search(q_emb, topk)
            sims, ids = D[0], I[0]
        else:
            sims, ids = self._numpy_search(q_emb, topk)
        out: List[Tuple[float, Chunk]] = []
        for s, i in zip(sims, ids):
            if i < 0: continue
            if s < MIN_SIM: continue
            out.append((float(s), self.meta[int(i)]))
        return out

# ---------- Topic routing ----------
TOPIC_TAGS = {
    "旅程取消": ["旅程取消"],
    "班機延誤": ["班機延誤"],
    "旅程更改": ["旅程更改"],
    "行李延誤": ["行李延誤"],
    "行李損失": ["行李損失"],
    "文件損失": ["文件損失"],
    "改降機場": ["改降機場"],
    "劫機": ["劫機"],
    "食品中毒": ["食品中毒"],
    "現金竊盜": ["現金竊盜"],
    "信用卡盜用": ["信用卡盜用"],
    "居家竊盜": ["居家竊盜"],
    "租車事故": ["租車事故"],
    "特殊活動取消": ["特殊活動取消","賽事取消"],
    "賽事取消": ["賽事取消"],
    "行動電話被竊": ["行動電話被竊"],
    "急難救助": ["急難救助"],
    "費率": ["費率"],
    "全域": [],
    "不保查詢": [],  # 查詢不保的特別路由
}

# 條號 allowlist
ARTICLE_ALLOWLIST = {
    "班機延誤": ["第三十條", "第三十一條", "第三十二條"],
    "行李延誤": ["第三十六條","第三十七條","第三十八條"],
    "行李損失": ["第三十九條","第四十條","第四十一條","第四十二條","第四十三條","第四十四條"],
}

# 條文子類別
ARTICLE_SUBCATS = {
    "第三十條": "承保範圍",
    "第三十一條": "特別不保事項",
    "第三十二條": "理賠文件",
    "第三十六條": "承保範圍",
    "第三十七條": "特別不保事項",
    "第三十八條": "理賠文件",
    "第三十九條": "承保範圍",
    "第四十條": "特別不保事項（物品）",
    "第四十一條": "特別不保事項（事故）",
    "第四十二條": "事故發生時之處理",
    "第四十三條": "理賠文件",
    "第四十四條": "追回處理",
}

# 標題覆寫
ARTICLE_TITLES_OVERRIDE = {
    "第三十條": "班機延誤保險承保範圍",
    "第三十一條": "班機延誤保險特別不保事項",
    "第三十二條": "班機延誤保險理賠文件",
}

def route(query: str) -> str:
    q = query
    # 不保/除外/不賠 → 不保查詢（再細分）
    if any(k in q for k in ["不保","不可理賠","除外","不賠","排除"]):
        return "不保查詢"
    # 旅遊/航班延誤 → 班機延誤
    if any(k in q for k in ["旅遊延誤","旅行延誤","旅途延誤","航班延誤","飛機延誤","轉機延誤","延誤賠償"]):
        return "班機延誤"
    if "行李" in q and "延誤" in q:
        return "行李延誤"
    if "行李" in q and any(k in q for k in ["遺失","毀損","竊盜","搶奪","託運","行李箱"]):
        return "行李損失"
    if "旅程" in q and any(k in q for k in ["取消","病危","死亡","罷工","暴動","天災"]):
        return "旅程取消"
    if any(k in q for k in ["更改","改期","改班","護照","交通意外"]):
        return "旅程更改"
    if "班機" in q and "延誤" in q:
        return "班機延誤"
    if any(k in q for k in ["護照","旅行文件"]):
        return "文件損失"
    if any(k in q for k in ["信用卡","盜刷","掛失","止付"]):
        return "信用卡盜用"
    if any(k in q for k in ["現金","房間","旅館","失竊"]):
        return "現金竊盜"
    if any(k in q for k in ["租車","交通事故"]) and "租" in q:
        return "租車事故"
    if "改降" in q:
        return "改降機場"
    if "劫機" in q:
        return "劫機"
    if any(k in q for k in ["食品中毒","食物中毒"]):
        return "食品中毒"
    if any(k in q for k in ["居家","住宅","遭竊"]):
        return "居家竊盜"
    if any(k in q for k in ["表演","遊樂場","滑雪","賽事","門票","活動取消"]):
        return "特殊活動取消" if any(x in q for x in ["表演","遊樂場","滑雪"]) else "賽事取消"
    if any(k in q for k in ["手機","行動電話","被竊"]):
        return "行動電話被竊"
    if any(k in q for k in ["醫療轉送","遺體運送","搜索","高山症","探視","未成年"]):
        return "急難救助"
    if any(k in q for k in ["費率","係數","內插"]):
        return "費率"
    return "全域"

def is_allowed_by_topic(chunk_tags: List[str], topic: str) -> bool:
    if topic == "全域" or topic == "不保查詢":
        return True
    allowed_tags = set(TOPIC_TAGS.get(topic, []))
    return bool(allowed_tags.intersection(set(chunk_tags)))

def is_allowed_by_article(article_no: str, topic: str) -> bool:
    allow = ARTICLE_ALLOWLIST.get(topic)
    if not allow:
        return True
    return any(article_no.startswith(p) for p in allow)

# ---------- Helpers ----------
def find_article(chunks: List[Chunk], article_no: str) -> Optional[Chunk]:
    for c in chunks:
        if c.article_no.startswith(article_no):
            return c
    return None

def title_with_override(c: Chunk) -> str:
    return ARTICLE_TITLES_OVERRIDE.get(c.article_no, c.article_title)

# ---------- Snippets ----------
def make_snippets(hits: List[Tuple[float, Chunk]], topic: str, max_chars: int = MAX_SNIPPETS_CHARS) -> str:
    filtered = [(s, c) for s, c in hits if is_allowed_by_topic(c.tags, topic)]
    if not filtered:
        filtered = hits
    # 不保查詢不套 allowlist，避免誤刪；其餘主題照舊
    if topic != "不保查詢":
        filtered = [(s, c) for s, c in filtered if is_allowed_by_article(c.article_no, topic)] or filtered

    order_key = {
        "第三十條": 30, "第三十一條": 31, "第三十二條": 32,
        "第三十六條": 36, "第三十七條": 37, "第三十八條": 38,
        "第三十九條": 39, "第四十條": 40, "第四十一條": 41,
        "第四十二條": 42, "第四十三條": 43, "第四十四條": 44
    }
    sorted_hits = sorted(filtered, key=lambda t: (order_key.get(t[1].article_no, 999), -t[0]))

    out, total = [], 0
    for s, c in sorted_hits:
        subcat = ARTICLE_SUBCATS.get(c.article_no, "")
        t0 = title_with_override(c)
        title = f"{t0}" + (f"｜{subcat}" if subcat else "")
        block = f"【{c.article_no} {title}｜{c.chapter}】\n{c.text}\n\n"
        if total + len(block) > max_chars: break
        out.append(block); total += len(block)
    return "".join(out).strip()

# ---------- Ollama ----------
def run_ollama(prompt: str) -> str:
    proc = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8"))
    return proc.stdout.decode("utf-8")

PROMPT_TMPL = """你是保險條款助理。請「只依據」提供的條文回答，禁止臆測。
使用繁體中文，產出流程導向答案（申請步驟 → 應檢附文件 → 常見不保與限制），每點標註條號與章節名稱，不要寫頁碼。
主題：{topic}（主題為「全域」時，可引用任何條款）
若主題為「班機延誤」，請嚴格區分：第30條=承保範圍，第31條=特別不保事項，第32條=理賠文件；不得引用行李延誤或行李損失條款。

[問題]
{question}

[條文片段]
{snippets}

請輸出固定格式：
結論：<一句話>
申請步驟：
- <步驟>（第X條）
應檢附文件：
- <文件>（第X條）
常見不保與限制：
- <限制>（第X條）
依據：
- 第X條《條名/子類別》｜章名
"""

# ---------- Deterministic composers ----------
def compose_flight_delay(chunks: List[Chunk]) -> Optional[str]:
    a30 = find_article(chunks, "第三十條")
    a31 = find_article(chunks, "第三十一條")
    a32 = find_article(chunks, "第三十二條")
    if not (a30 and a31 and a32): return None
    t30, t31, t32 = map(title_with_override, (a30, a31, a32))
    ans = []
    ans.append("結論：申請班機延誤賠償需符合第30條之承保範圍。")
    ans.append("\n申請步驟：")
    ans.append(f"- 向航空業者取得延誤相關證明（{a32.article_no}）")
    ans.append(f"- 向保險公司提交理賠申請（{a32.article_no}）")
    ans.append("\n應檢附文件：")
    ans.append(f"- 理賠申請書（{a32.article_no}）")
    ans.append(f"- 機票與登機證，或航空業者出具之搭乘證明（{a32.article_no}）")
    ans.append(f"- 航空業者出具之載有班機延誤期間之證明（{a32.article_no}）")
    ans.append("\n常見不保與限制：")
    ans.append(f"- 依第31條之特別不保事項審核（{a31.article_no}）")
    ans.append("\n依據：")
    ans.append(f"- {a30.article_no}《{t30}》｜{a30.chapter}")
    ans.append(f"- {a31.article_no}《{t31}》｜{a31.chapter}")
    ans.append(f"- {a32.article_no}《{t32}》｜{a32.chapter}")
    return "\n".join(ans)

def compose_baggage_delay(chunks: List[Chunk]) -> Optional[str]:
    a36 = find_article(chunks, "第三十六條")
    a37 = find_article(chunks, "第三十七條")
    a38 = find_article(chunks, "第三十八條")
    if not (a36 and a37 and a38): return None
    t36, t37, t38 = map(title_with_override, (a36, a37, a38))
    ans = []
    ans.append("結論：申請行李延誤理賠需符合第36條之承保範圍。")
    ans.append("\n申請步驟：")
    ans.append(f"- 向運具業者（如航空公司）取得延誤相關證明（{a38.article_no}）")
    ans.append(f"- 向保險公司提交理賠申請（{a38.article_no}）")
    ans.append("\n應檢附文件：")
    ans.append(f"- 理賠申請書（{a38.article_no}）")
    ans.append(f"- 運具業者出具之行李延誤相關證明（{a38.article_no}）")
    ans.append("\n常見不保與限制：")
    ans.append(f"- 依第37條之特別不保事項審核（{a37.article_no}）")
    ans.append("\n依據：")
    ans.append(f"- {a36.article_no}《{t36}》｜{a36.chapter}")
    ans.append(f"- {a37.article_no}《{t37}》｜{a37.chapter}")
    ans.append(f"- {a38.article_no}《{t38}》｜{a38.chapter}")
    return "\n".join(ans)

def compose_baggage_loss(chunks: List[Chunk]) -> Optional[str]:
    need = ["第三十九條","第四十條","第四十一條","第四十二條","第四十三條","第四十四條"]
    arts = {n: find_article(chunks, n) for n in need}
    if not all(arts.values()): return None
    t = {n: title_with_override(arts[n]) for n in need}
    ans = []
    ans.append("結論：申請行李損失理賠需符合第39條之承保範圍，並依事故性質與不保事項辦理。")
    ans.append("\n申請步驟：")
    ans.append(f"- 事故發生時依{arts['第四十二條'].article_no}之處理規定辦理")
    ans.append(f"- 向保險公司提交理賠申請與必要文件（{arts['第四十三條'].article_no}）")
    ans.append("\n應檢附文件：")
    ans.append(f"- 依{arts['第四十三條'].article_no}規定準備文件（例如事故與損失證明等）")
    ans.append("\n常見不保與限制：")
    ans.append(f"- 物品類不保事項依{arts['第四十條'].article_no}")
    ans.append(f"- 事故類不保事項依{arts['第四十一條'].article_no}")
    ans.append("\n依據：")
    for n in need:
        ans.append(f"- {arts[n].article_no}《{t[n]}》｜{arts[n].chapter}")
    return "\n".join(ans)

def detect_exclusion_topic(q: str) -> str:
    # 回傳 "班機延誤" / "行李延誤" / "行李損失" / "全域"
    if any(k in q for k in ["班機","航班","飛機","轉機"]):
        return "班機延誤"
    if "行李" in q and "延誤" in q:
        return "行李延誤"
    if "行李" in q and any(k in q for k in ["損失","遺失","竊","毀損","搶奪","託運"]):
        return "行李損失"
    return "全域"

def compose_exclusions(chunks: List[Chunk], q: str) -> Optional[str]:
    tgt = detect_exclusion_topic(q)
    if tgt == "班機延誤":
        a31 = find_article(chunks, "第三十一條")
        if not a31: return None
        t31 = title_with_override(a31)
        return "\n".join([
            "結論：以下原因屬於不可理賠範圍（班機延誤）。",
            f"- 依{a31.article_no}《{t31}》審核（章節：{a31.chapter}）。",
        ])
    if tgt == "行李延誤":
        a37 = find_article(chunks, "第三十七條")
        if not a37: return None
        t37 = title_with_override(a37)
        return "\n".join([
            "結論：以下原因屬於不可理賠範圍（行李延誤）。",
            f"- 依{a37.article_no}《{t37}》審核（章節：{a37.chapter}）。",
            "提示：常見排除含返抵國內或出發地之延誤、事先運送非隨身託運等（以原文為準）。",
        ])
    if tgt == "行李損失":
        a40 = find_article(chunks, "第四十條")
        a41 = find_article(chunks, "第四十一條")
        if not (a40 and a41): return None
        t40 = title_with_override(a40); t41 = title_with_override(a41)
        return "\n".join([
            "結論：以下原因屬於不可理賠範圍（行李損失）。",
            f"- 物品類除外依{a40.article_no}《{t40}》（章節：{a40.chapter}）。",
            f"- 事故類除外依{a41.article_no}《{t41}》（章節：{a41.chapter}）。",
        ])
    # 全域總覽
    a31 = find_article(chunks, "第三十一條")
    a37 = find_article(chunks, "第三十七條")
    a40 = find_article(chunks, "第四十條"); a41 = find_article(chunks, "第四十一條")
    if not (a31 and a37 and a40 and a41): return None
    t31 = title_with_override(a31); t37 = title_with_override(a37)
    t40 = title_with_override(a40); t41 = title_with_override(a41)
    return "\n".join([
        "結論：以下為主要險種的不可理賠條文入口。",
        f"- 班機延誤：{a31.article_no}《{t31}》｜{a31.chapter}",
        f"- 行李延誤：{a37.article_no}《{t37}》｜{a37.chapter}",
        f"- 行李損失（物品類）：{a40.article_no}《{t40}》｜{a40.chapter}",
        f"- 行李損失（事故類）：{a41.article_no}《{t41}》｜{a41.chapter}",
    ])

COMPOSERS: Dict[str, Any] = {
    "班機延誤": compose_flight_delay,
    "行李延誤": compose_baggage_delay,
    "行李損失": compose_baggage_loss,
}

# ---------- Engine ----------
class RAG:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks: List[Chunk] = []
        self.index = VectorIndex(EMB_MODEL)

    def build(self):
        self.chunks = pdf_to_chunks(self.pdf_path)
        self.index.build(self.chunks)

    def ask(self, question: str, topk: int = TOP_K) -> Dict[str, Any]:
        topic = route(question)

        # 不保查詢：走專用 deterministic
        if topic == "不保查詢":
            det = compose_exclusions(self.chunks, question)
            if det:
                # 彙整對應的 sources
                srcs = []
                for art in ["第三十一條","第三十七條","第四十條","第四十一條"]:
                    c = find_article(self.chunks, art)
                    if c:
                        srcs.append({
                            "article_no": c.article_no,
                            "article_title": title_with_override(c),
                            "subcat": ARTICLE_SUBCATS.get(c.article_no, ""),
                            "chapter": c.chapter,
                        })
                return {"route": "不保查詢", "answer": det, "sources": srcs}

        # 其他 deterministic 主題
        if topic in COMPOSERS:
            det = COMPOSERS[topic](self.chunks)
            if det:
                sources = []
                for art in ARTICLE_ALLOWLIST.get(topic, []):
                    c = find_article(self.chunks, art)
                    if c:
                        sources.append({
                            "article_no": c.article_no,
                            "article_title": title_with_override(c),
                            "subcat": ARTICLE_SUBCATS.get(c.article_no, ""),
                            "chapter": c.chapter,
                        })
                return {"route": topic, "answer": det, "sources": sources}

        # Default LLM path（保留泛問能力）
        raw_hits = self.index.search(question, topk*3)
        snippets = make_snippets(raw_hits, topic, MAX_SNIPPETS_CHARS)
        prompt = PROMPT_TMPL.format(question=question, snippets=snippets, topic=topic)
        answer = run_ollama(prompt)

        src = [{
            "article_no": c.article_no,
            "article_title": title_with_override(c),
            "subcat": ARTICLE_SUBCATS.get(c.article_no, ""),
            "chapter": c.chapter,
            "tags": c.tags,
            "score": round(s, 3)
        } for s, c in raw_hits
          if (is_allowed_by_topic(c.tags, topic) or topic in ["全域","不保查詢"])
          and (topic in ["全域","不保查詢"] or is_allowed_by_article(c.article_no, topic))]

        return {"route": topic, "answer": answer, "sources": src}

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="path to 海外旅行不便險條款.pdf")
    ap.add_argument("--q", default="哪些原因屬於不可理賠範圍？")
    ap.add_argument("--topk", type=int, default=TOP_K)
    args = ap.parse_args()

    rag = RAG(args.pdf)
    print("Building index ...")
    rag.build()
    out = rag.ask(args.q, topk=args.topk)

    print("=== Route ===")
    print(out["route"])
    print("\n=== Answer ===")
    print(out["answer"])
    print("\n=== Sources ===")
    print(json.dumps(out["sources"], ensure_ascii=False, indent=2))
