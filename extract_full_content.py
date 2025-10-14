#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract full article content from local raw HTML files for records in a JSONL.
- Input:  JSONL with fields: url, title, preview, raw_html_path, ...
- Output: JSONL with added: content, content_len, extraction_method, status
- Also writes: failures.log for items that could not be extracted
Usage (basic):
  python extract_full_content.py --in traffic_2025_final.jsonl --out traffic_2025_full.jsonl --raw-base .
Notes:
  * raw_html_path can be absolute or relative to --raw-base
  * Supports .html and .html.gz
"""

import argparse
import json
import gzip
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup

def read_html_file(path: Path) -> Optional[str]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return None

def norm_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_boilerplate(text: str) -> str:
    # Remove common boilerplate phrases; extend as you encounter them
    patterns = [
        r"(?i)Bạn đang đọc bài viết.*$",
        r"(?i)Liên hệ quảng cáo.*$",
        r"(?i)Video.*$",
        r"(?i)Xem thêm:.*$",
        r"(?i)Ảnh:.*$",
        r"(?i)\(Theo.*\)$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text).strip()
    return text

def extract_by_selectors(soup: BeautifulSoup, selectors: List[str]) -> Tuple[str, str]:
    for sel in selectors:
        node = soup.select_one(sel)
        if not node:
            continue
        # Collect text from paragraphs and meaningful blocks inside the node
        parts = []
        # Prefer paragraphs if present
        ps = node.select("p")
        if ps:
            parts = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]
        else:
            parts = [node.get_text(" ", strip=True)]
        text = "\n".join(parts).strip()
        text = clean_boilerplate(norm_whitespace(text))
        if len(text) >= 200:  # heuristic: ensure it's not too short
            return text, f"selector:{sel}"
    return "", ""

def extract_generic(soup: BeautifulSoup) -> Tuple[str, str]:
    # Generic fallbacks
    candidates = [
        "article",
        "div#main-content",
        "div#content",
        "div.article-body",
        "div.article__body",
        "div.detail__content",
        "div.content-detail",
        "div.entry-content",
        "div.post-content",
        'div[class*="content"]',
    ]
    text, method = extract_by_selectors(soup, candidates)
    if text:
        return text, method

    # Try concatenating all <p> in the page as a last resort
    ps = soup.find_all("p")
    parts = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]
    text = clean_boilerplate(norm_whitespace("\n".join(parts)))
    if len(text) >= 200:
        return text, "fallback:all_p"
    return "", ""

def extract_for_domain(soup: BeautifulSoup, domain: str) -> Tuple[str, str]:
    d = (domain or "").lower()
    # Domain-specific selectors (extend as needed)
    if "tienphong.vn" in d:
        text, method = extract_by_selectors(soup, [
            "div#article-body",
            "div.article__body",
            "article",
            "div.detail__content",
        ])
        if text:
            return text, method

    if "baoxaydung.vn" in d:
        text, method = extract_by_selectors(soup, [
            "div.article__content",
            "div.detail-content",
            "article",
            "div#main-content",
        ])
        if text:
            return text, method

    # Add more site rules here as you see patterns...
    return extract_generic(soup)

def extract_content_from_html(html: str, domain: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    text, method = extract_for_domain(soup, domain)
    return text, method

def resolve_html_path(raw_html_path: str, raw_base: Path) -> Path:
    p = Path(raw_html_path)
    if not p.is_absolute():
        p = (raw_base / p).resolve()
    return p

def process(args):
    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    raw_base = Path(args.raw_base).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_log = out_path.with_suffix(".failures.log")
    n_in = n_out = n_fail = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout, \
         fail_log.open("w", encoding="utf-8") as flog:
        for line in fin:
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                n_fail += 1
                flog.write(f"[JSON_ERROR] line={n_in} err={e}\n")
                continue

            domain = rec.get("domain") or (rec.get("url","").split("/")[2] if "://" in rec.get("url","") else "")
            raw_html_path = rec.get("raw_html_path")

            status = "ok"
            content = ""
            method = ""

            if raw_html_path:
                html_file = resolve_html_path(raw_html_path, raw_base)
                html = read_html_file(html_file)
                if html:
                    content, method = extract_content_from_html(html, domain)
                    if not content:
                        status = "extraction_empty"
                else:
                    status = "html_not_found"
            else:
                status = "no_raw_html_path"

            # Fallback: if still empty, try to at least use preview
            if not content:
                preview = rec.get("preview") or ""
                if len(preview.strip()) >= 50:
                    content = preview.strip()
                    method = method or "fallback:preview_only"

            rec["content"] = content
            rec["content_len"] = len(content)
            rec["extraction_method"] = method
            rec["status"] = status

            if content:
                n_out += 1
            else:
                n_fail += 1
                flog.write(f"[NO_CONTENT] url={rec.get('url','')} raw={raw_html_path} status={status}\n")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Read {n_in} records; wrote {n_out} with content; failures: {n_fail}")
    print(f"Output: {out_path}")
    print(f"Failures log: {fail_log}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="outfile", required=True, help="Output JSONL path")
    ap.add_argument("--raw-base", dest="raw_base", default=".", help="Base directory for raw_html_path (if relative)")
    args = ap.parse_args()
    process(args)

if __name__ == "__main__":
    main()
