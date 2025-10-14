#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Vietnamese news JSONL:
- Clean boilerplate/HTML/URLs/zero-width
- Normalize whitespace & Unicode (NFC)
- Filter short/empty
- Deduplicate by cleaned-content MD5
- Parse pub_date -> ISO
- Extract domain/path from URL
- (Optional) Keep only HCMC-related articles via keyword regex
- Export cleaned JSONL + metrics JSON
Usage:
  python preprocess_news.py \
      --in traffic_2025_full.jsonl \
      --out traffic_2025_clean.jsonl \
      --metrics traffic_2025_clean_metrics.json \
      --min-len 150 \
      --hcmc-only
"""

import argparse, json, re, sys, hashlib, io, os, math
import unicodedata
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

# -------------------------
# Utilities
# -------------------------

def open_any(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    """Open .jsonl or .jsonl.gz with the same call."""
    p = str(path).lower()
    if p.endswith(".gz"):
        import gzip
        return gzip.open(path, mode.replace("t", ""), encoding=encoding) if "t" in mode else gzip.open(path, mode)
    return open(path, mode, encoding=encoding) if "t" in mode else open(path, mode)

def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
HTML_TAG_RE   = re.compile(r"<[^>]+>")
URL_IN_TEXT_RE= re.compile(r"https?://\S+")
MULTI_SPACE_RE= re.compile(r"[ \t]+")
MANY_NL_RE    = re.compile(r"\n{3,}")
SPACE_NL_RE   = re.compile(r"\s+\n")

# Generic boilerplate lines commonly seen in VN news
TRASH_PATTERNS = [
    r"(?im)^tin liên quan.*$",
    r"(?im)^bài viết liên quan.*$",
    r"(?im)^xem thêm:.*$",
    r"(?im)^theo .*?:.*$",
    r"(?im)video:.*$",
    r"(?im)clip:.*$",
    r"(?im)ảnh:.*$",
    r"(?im)^copyright.*$",
]
TRASH_RE_LIST = [re.compile(pat) for pat in TRASH_PATTERNS]

# Simple site tag patterns like [VOV], (PLO), ...
SITE_TAG_RE = re.compile(r"[\[(](?:VOV|PLO|VTV|VnExpress|Tuổi Trẻ|Thanh Niên|Zing|Tienphong|Dân Trí|Báo .*?)[\])]?", re.IGNORECASE)

def strip_boilerplate(text: str) -> str:
    t = HTML_TAG_RE.sub(" ", text)
    t = URL_IN_TEXT_RE.sub("", t)
    for rx in TRASH_RE_LIST:
        t = rx.sub("", t)
    t = SITE_TAG_RE.sub("", t)
    t = MULTI_SPACE_RE.sub(" ", t)
    t = MANY_NL_RE.sub("\n\n", t)
    return t.strip()

def vn_normalize(text: str) -> str:
    t = to_nfc(text)
    t = ZERO_WIDTH_RE.sub("", t)
    t = MULTI_SPACE_RE.sub(" ", t)
    t = SPACE_NL_RE.sub("\n", t)
    t = MANY_NL_RE.sub("\n\n", t)
    return t.strip()

def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Date parsing without external deps
DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]
def parse_pub_date(s):
    if not s:
        return None
    s = str(s).strip()
    # Try strict formats
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s[:26], fmt)
        except Exception:
            pass
    # Try fromisoformat (handles 'YYYY-MM-DDTHH:MM:SS[.ffffff]')
    try:
        return datetime.fromisoformat(s[:26])
    except Exception:
        return None

def extract_domain_path(url: str):
    try:
        pr = urlparse(url)
        return pr.netloc or None, pr.path or None
    except Exception:
        return None, None

# HCMC keyword set (expand as needed)
HCMC_PATTERNS = [
    r"\bTP(?:\.| |)HCM\b", r"\bTPHCM\b", r"\bHồ Chí Minh\b", r"\bSai?̀? Gòn\b", r"\bSài Gòn\b",
    r"\bQuận\s*(?:1|2|3|4|5|6|7|8|9|10|11|12)\b",
    r"\b(Bình Thạnh|Phú Nhuận|Gò Vấp|Tân Bình|Tân Phú|Bình Tân|Thủ Đức|Nhà Bè|Cần Giờ|Củ Chi|Hóc Môn|Bình Chánh)\b",
    r"\b(Xa lộ Hà Nội|Phạm Văn Đồng|Võ Văn Kiệt|Nguyễn Văn Linh|QL1|Vành đai\s*(2|3|4)|Cao tốc Long Thành)\b",
]
HCMC_RE = re.compile("|".join(HCMC_PATTERNS), re.IGNORECASE | re.UNICODE)

def is_hcmc(text: str, title: str = "") -> bool:
    blob = f"{title}\n{text}"
    return HCMC_RE.search(blob) is not None

# -------------------------
# Core processing
# -------------------------

def process_file(in_path: Path, out_path: Path, metrics_path: Path,
                 min_len: int = 150, hcmc_only: bool = False,
                 keep_prob_field: str = "_ml_pos_prob",
                 write_preview: bool = False):
    seen = set()
    kept = 0
    counters = {
        "empty_after_clean": 0,
        "too_short": 0,
        "dup_content": 0,
        "not_hcmc": 0,
        "parse_error": 0,
        "written": 0,
    }

    # Prepare IO
    out_f = open_any(out_path, "wt", encoding="utf-8")

    # Optional preview file for debugging first 50 keeps
    prev_writer = None
    if write_preview:
        prev_writer = open_any(out_path.with_suffix(out_path.suffix + ".preview.jsonl"), "wt", encoding="utf-8")
        prev_count = 0

    # Stream line-by-line
    with open_any(in_path, "rt", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                counters["parse_error"] += 1
                continue

            url    = str(row.get("url", "") or "")
            title0 = row.get("title", "")
            cont0  = row.get("content", "")

            title  = vn_normalize(title0)
            cont   = vn_normalize(strip_boilerplate(cont0))

            if not cont:
                counters["empty_after_clean"] += 1
                continue
            if len(cont) < min_len:
                counters["too_short"] += 1
                continue

            # Optional HCMC filter
            if hcmc_only and not is_hcmc(cont, title):
                counters["not_hcmc"] += 1
                continue

            h = content_hash(cont)
            if h in seen:
                counters["dup_content"] += 1
                continue
            seen.add(h)

            dt = parse_pub_date(row.get("pub_date"))
            year = dt.year if isinstance(dt, datetime) else row.get("year", None)
            if isinstance(year, float) and math.isnan(year):
                year = None
            domain, path = extract_domain_path(url)

            cleaned = {
                "url": url,
                "domain": domain,
                "path": path,
                "title": title,
                "content": cont,
                "content_len": len(cont),
                "pub_date": dt.isoformat() if isinstance(dt, datetime) else None,
                "year": int(year) if isinstance(year, int) or (isinstance(year, str) and year.isdigit()) else (year if year is None else int(year)),
                keep_prob_field: float(row.get(keep_prob_field)) if row.get(keep_prob_field) is not None else None,
                "raw_html_path": row.get("raw_html_path"),
                "extraction_method": row.get("extraction_method"),
                "status": row.get("status"),
                "_md5_clean": h,
            }

            out_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            counters["written"] += 1
            kept += 1

            if prev_writer and prev_count < 50:
                prev_writer.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                prev_count += 1

    out_f.close()
    if prev_writer:
        prev_writer.close()

    metrics = {
        "input": str(in_path),
        "output": str(out_path),
        "kept": kept,
        "dropped": counters["empty_after_clean"] + counters["too_short"] + counters["dup_content"] + counters["not_hcmc"] + counters["parse_error"],
        "drop_reasons": {
            "empty_after_clean": counters["empty_after_clean"],
            "too_short": counters["too_short"],
            "dup_content": counters["dup_content"],
            "not_hcmc": counters["not_hcmc"],
            "parse_error": counters["parse_error"],
        },
        "unique_contents": len(seen),
        "hcmc_only": hcmc_only,
        "min_len": min_len,
    }

    with open_any(metrics_path, "wt", encoding="utf-8") as mf:
        mf.write(json.dumps(metrics, ensure_ascii=False, indent=2))

    return metrics

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Preprocess Vietnamese news JSONL.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input .jsonl or .jsonl.gz")
    ap.add_argument("--out", dest="out_path", required=True, help="Output cleaned .jsonl (support .gz)")
    ap.add_argument("--metrics", dest="metrics_path", required=False, default=None, help="Output metrics .json")
    ap.add_argument("--min-len", dest="min_len", type=int, default=150, help="Min content length after clean")
    ap.add_argument("--hcmc-only", dest="hcmc_only", action="store_true", help="Keep only HCMC-related articles")
    ap.add_argument("--preview", dest="preview", action="store_true", help="Also write first 50 rows to <out>.preview.jsonl")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    metrics_path = Path(args.metrics_path) if args.metrics_path else out_path.with_suffix(out_path.suffix + ".metrics.json")

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = process_file(
        in_path=in_path,
        out_path=out_path,
        metrics_path=metrics_path,
        min_len=args.min_len,
        hcmc_only=args.hcmc_only,
        write_preview=args.preview,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
