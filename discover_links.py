import asyncio, aiohttp, logging, argparse, re
from pathlib import Path
from datetime import datetime, timezone, date
from aiolimiter import AsyncLimiter
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode
from collections import defaultdict, deque

# ---------- config ----------
OUT_LINKS = Path("data/links.txt")
LOG_FILE  = Path("data/discover.log")
CONCURRENCY = 40
RATE_WINDOW = 2
PER_DOMAIN_MAX = 3
TIMEOUT = 20

USER_AGENT = (
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Mở rộng seed nhiều báo lớn
SITEMAP_SEEDS = [
    # VnExpress
    "https://vnexpress.net/sitemap.xml",
    "https://vnexpress.net/latestnews-sitemap.xml",
    # Tuổi Trẻ
    "https://tuoitre.vn/sitemapindex.xml",
    "https://tuoitre.vn/Sitemap/GoogleNews.ashx",
    # Zing
    "https://znews.vn/sitemap/sitemap-index.xml",
    # Thanh Niên
    "https://thanhnien.vn/sitemap.xml",
    # Dân Trí
    "https://dantri.com.vn/sitemaps.xml",
    # Báo Giao Thông
    "https://www.baogiaothong.vn/sitemap.xml",
    # Vietnamnet
    "https://vietnamnet.vn/sitemaps.xml",
    # NLĐO
    "https://nld.com.vn/sitemaps.xml",
    # Tiền Phong
    "https://tienphong.vn/sitemap.xml",
    # VTV
    "https://vtv.vn/sitemaps.xml",
    # VTC News
    "https://vtc.vn/sitemap.xml",
    # VOV
    "https://vov.vn/sitemap.xml",
    # Lao Động
    "https://laodong.vn/sitemap.xml",
]

RSS_LIST_PAGES = [
    "https://vnexpress.net/rss",
    "https://tuoitre.vn/rss.htm",
    "https://thanhnien.vn/rss.html",
    "https://dantri.com.vn/rss.htm",
    "https://vietnamnet.vn/rss",
    "https://nld.com.vn/rss.htm",
    "https://tienphong.vn/rss",
    "https://vtv.vn/rss",
    "https://vtc.vn/rss",
    "https://vov.vn/rss",
    "https://laodong.vn/rss"
]

# ---------- article URL filters ----------
BLOCK_SUFFIXES = (".xml", ".rss", ".json", ".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".m3u8")
BLOCK_SUBSTRS  = ("/sitemap", "/rss", "/feed", "/video/", "/photo/", "/amp", "/tag/", "/topic/", "/chu-de/", "/page/")
ARTICLE_PATTERNS = {
    "vnexpress.net":      re.compile(r"^https?://(?:www\.)?vnexpress\.net/[^?#]+-\d+\.html$"),
    "tuoitre.vn":         re.compile(r"^https?://(?:www\.)?tuoitre\.vn/[^?#]+-\d{8}\.htm$"),
    "znews.vn":           re.compile(r"^https?://(?:www\.)?znews\.vn/[^?#]+-post\d+\.html$"),
    "zingnews.vn":        re.compile(r"^https?://(?:www\.)?zingnews\.vn/[^?#]+-post\d+\.html$"),
    "dantri.com.vn":      re.compile(r"^https?://(?:www\.)?dantri\.com\.vn/[^?#]+-\d+\.htm$"),
    "thanhnien.vn":       re.compile(r"^https?://(?:www\.)?thanhnien\.vn/(?!tag/|video/)[^?#]+\.htm$"),
    "baogiaothong.vn":    re.compile(r"^https?://(?:www\.)?baogiaothong\.vn/[^?#]+\.htm$"),
    "hcmcpv.org.vn":      re.compile(r"^https?://(?:www\.)?hcmcpv\.org\.vn/tin-tuc/[^?#]+$"),
    "vietnamnet.vn":      re.compile(r"^https?://(?:www\.)?vietnamnet\.vn/[^?#]+\.html$"),
    "nld.com.vn":         re.compile(r"^https?://(?:www\.)?nld\.com\.vn/[^?#]+\.htm$"),
    "tienphong.vn":       re.compile(r"^https?://(?:www\.)?tienphong\.vn/[^?#]+\.tpo$"),
    "vtv.vn":             re.compile(r"^https?://(?:www\.)?vtv\.vn/[^?#]+\.htm$"),
    "vtc.vn":             re.compile(r"^https?://(?:www\.)?vtc\.vn/[^?#]+\.html$"),
    "vov.vn":             re.compile(r"^https?://(?:www\.)?vov\.vn/[^?#]+\.vov$"),
    "laodong.vn":         re.compile(r"^https?://(?:www\.)?laodong\.vn/[^?#]+\.ldo$"),
}
GENERIC_HTML = re.compile(r"^https?://[^/]+/[^?#]+\.html?$")
GENERIC_DEEP = re.compile(r"^https?://[^/]+/(?:[^/]+/){2,}[^?#]+$")

def probably_article(u: str) -> bool:
    if not u.startswith(("http://","https://")):
        return False
    if u.endswith(BLOCK_SUFFIXES) or any(s in u for s in BLOCK_SUBSTRS):
        return False
    host = urlparse(u).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    pat = ARTICLE_PATTERNS.get(host)
    if pat and pat.match(u):
        return True
    if GENERIC_HTML.match(u) or GENERIC_DEEP.match(u):
        return True
    return False

DROP_QUERY_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","fbclid","gclid"}
def normalize_url(url: str) -> str:
    s = urlsplit(url)
    q = [(k,v) for k,v in parse_qsl(s.query, keep_blank_values=True) if k not in DROP_QUERY_KEYS]
    return urlunsplit((s.scheme, s.netloc, s.path, urlencode(q, doseq=True), ""))

SEEN_FILE = Path("data/seen_urls.txt")
def load_seen():
    seen = set()
    if SEEN_FILE.exists():
        for ln in SEEN_FILE.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                seen.add(ln)
    return seen
def save_seen(seen):
    SEEN_FILE.write_text("\n".join(sorted(seen)), encoding="utf-8")

# ---------- time helpers ----------
DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b"),
    re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b"),
]
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

def safe_to_date(y: int, m: int=1, d: int=1) -> date | None:
    try:
        return date(y, m, d)
    except Exception:
        return None

def parse_date_guess(s: str) -> date | None:
    s = (s or "").strip()
    if not s: return None
    try:
        if "T" in s or "-" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.date()
    except Exception:
        pass
    for pat in DATE_PATTERNS:
        m = pat.search(s)
        if m:
            g = [int(x) for x in m.groups()]
            if len(g) == 3:
                if 1900 < g[0] < 3000:   # yyyy-mm-dd
                    return safe_to_date(g[0], g[1], g[2])
                if 1900 < g[2] < 3000:   # dd-mm-yyyy
                    return safe_to_date(g[2], g[1], g[0])
    m = YEAR_PATTERN.search(s)
    if m:
        return safe_to_date(int(m.group(1)), 1, 1)
    return None

def in_range(d: date | None, start: date | None, end: date | None) -> bool:
    if d is None: return False
    if start and d < start: return False
    if end and d > end: return False
    return True

def date_from_sitemap_loc(loc: str) -> date | None:
    return parse_date_guess(loc)

def date_from_article_url(u: str) -> date | None:
    m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", u)
    if m: return safe_to_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", u)
    if m: return safe_to_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", u)
    if m: return safe_to_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = YEAR_PATTERN.search(u)
    if m: return safe_to_date(int(m.group(1)), 1, 1)
    return None

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("discover")

# ---------- helpers ----------
def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
def is_http_url(u: str) -> bool:
    return u.startswith("http://") or u.startswith("https://")

async def fetch_text(session, url, limiter, timeout=TIMEOUT):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/xml, text/xml, text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    }
    async with limiter:
        try:
            async with session.get(url, headers=headers, allow_redirects=True, timeout=timeout) as r:
                if r.status == 200:
                    return await r.text(errors="ignore")
                else:
                    logger.debug(f"{url[:80]}... -> {r.status}")
        except Exception as e:
            logger.debug(f"fetch error {url[:80]}... {type(e).__name__}")
    return None

def extract_links_from_sitemap_xml(
    xml_text: str,
    start_d: date | None,
    end_d: date | None,
    parent_hint_date: date | None = None,   # fallback theo sitemap cha
):
    urls, children = [], []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls, children

    tag = strip_ns(root.tag).lower()

    if tag == "sitemapindex":
        for sm in root:
            if strip_ns(sm.tag).lower() == "sitemap":
                loc = None
                lastmod = None
                for c in sm:
                    t = strip_ns(c.tag).lower()
                    if t == "loc": loc = (c.text or "").strip()
                    elif t == "lastmod": lastmod = (c.text or "").strip()
                if loc and is_http_url(loc):
                    sm_date = parse_date_guess(lastmod) if lastmod else date_from_sitemap_loc(loc)
                    if (sm_date is None) or in_range(sm_date, start_d, end_d):
                        children.append(loc)

    elif tag == "urlset":
        for u in root:
            if strip_ns(u.tag).lower() == "url":
                loc, lastmod = None, None
                for c in u:
                    t = strip_ns(c.tag).lower()
                    if t == "loc": loc = (c.text or "").strip()
                    elif t == "lastmod": lastmod = (c.text or "").strip()
                if loc and is_http_url(loc):
                    u_date = parse_date_guess(lastmod) if lastmod else date_from_article_url(loc)
                    if u_date is None:
                        u_date = parent_hint_date  # Fallback: ngày của sitemap cha
                    if in_range(u_date, start_d, end_d):
                        urls.append(loc)

    return urls, children

def extract_links_from_rss_xml(xml_text: str, start_d: date | None, end_d: date | None):
    urls = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls

    def item_date(elem) -> date | None:
        for child in elem:
            t = strip_ns(child.tag).lower()
            if t in ("pubdate", "updated", "published", "dc:date"):
                if child.text and child.text.strip():
                    d = parse_date_guess(child.text.strip())
                    if d: return d
        return None

    for item in root.iter():
        t = strip_ns(item.tag).lower()
        if t == "item" or t == "entry":
            link = None
            for c in item:
                tt = strip_ns(c.tag).lower()
                if tt == "link":
                    if c.text and c.text.strip():
                        link = c.text.strip()
                    elif "href" in c.attrib:
                        link = c.attrib["href"].strip()
            if not (link and is_http_url(link)):
                continue
            d = item_date(item) or date_from_article_url(link)
            if in_range(d, start_d, end_d):
                urls.append(link)
    return urls

# ---------- main logic ----------
async def discover(sitemap_seeds, rss_list_pages, start_d: date | None, end_d: date | None, limit: int | None):
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    limiter = AsyncLimiter(PER_DOMAIN_MAX, RATE_WINDOW)
    to_visit_sitemaps = list(dict.fromkeys(sitemap_seeds))
    rss_feeds = set()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        rss_href = re.compile(r'href=["\']([^"\']+\.(?:rss|xml))["\']', re.I)
        rss_alt  = re.compile(r'<link[^>]+rel=["\']alternate["\'][^>]+type=["\']application/(?:rss|atom)\+xml["\'][^>]+href=["\']([^"\']+)["\']', re.I)
        for page in rss_list_pages:
            html = await fetch_text(session, page, limiter)
            if not html: continue
            for pat in (rss_href, rss_alt):
                for m in pat.finditer(html):
                    href = m.group(1)
                    if is_http_url(href): rss_feeds.add(href)

        urls_out = []
        visited_sitemaps = set()
        while to_visit_sitemaps:
            sm = to_visit_sitemaps.pop()
            if sm in visited_sitemaps: continue
            visited_sitemaps.add(sm)

            xml = await fetch_text(session, sm, limiter)
            if not xml: continue

            # 👉 đoán ngày từ chính URL của sitemap hiện tại
            parent_hint = date_from_sitemap_loc(sm)

            urls, children = extract_links_from_sitemap_xml(
                xml, start_d, end_d, parent_hint_date=parent_hint
            )
            urls_out.extend(urls)
            for child in children:
                if child not in visited_sitemaps:
                    to_visit_sitemaps.append(child)
            if limit and len(urls_out) >= limit:
                break

        for feed in rss_feeds:
            xml = await fetch_text(session, feed, limiter)
            if not xml: continue
            urls_out.extend(extract_links_from_rss_xml(xml, start_d, end_d))

    # Dedup + normalize + chỉ giữ URL có vẻ là bài
    clean, seen_norm = [], set()
    for u in urls_out:
        if not is_http_url(u): continue
        if not probably_article(u): continue
        nu = normalize_url(u)
        if nu in seen_norm: continue
        seen_norm.add(nu); clean.append(nu)

    # Round-robin theo domain
    buckets = defaultdict(deque)
    for u in clean:
        host = urlparse(u).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        buckets[host].append(u)
    rr = []
    while buckets:
        for host in list(buckets.keys()):
            if buckets[host]:
                rr.append(buckets[host].popleft())
            if not buckets[host]:
                buckets.pop(host, None)
    if limit: rr = rr[:limit]
    return rr

def write_links_append_no_dup(new_links):
    """Append vào OUT_LINKS nhưng bỏ trùng với file cũ (sau normalize)."""
    OUT_LINKS.parent.mkdir(parents=True, exist_ok=True)

    existed = set()
    if OUT_LINKS.exists():
        for ln in OUT_LINKS.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.startswith(("http://","https://")):
                existed.add(normalize_url(ln))

    uniq_new = []
    for ln in new_links:
        nu = normalize_url(ln)
        if nu not in existed:
            existed.add(nu)
            uniq_new.append(ln)

    with OUT_LINKS.open("a", encoding="utf-8") as f:
        for ln in uniq_new:
            f.write(ln.strip() + "\n")
    logging.info(f"✅ Appended {len(uniq_new)} new links (skipped {len(new_links)-len(uniq_new)} dup) -> {OUT_LINKS}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (bao gồm)")
    parser.add_argument("--end",   type=str, default=None, help="YYYY-MM-DD (bao gồm)")
    parser.add_argument("--limit", type=int, default=50000, help="Giới hạn số link tối đa")
    parser.add_argument("--use-seen", action="store_true", help="Loại bỏ URL đã có trong data/seen_urls.txt")
    args = parser.parse_args()

    start_d = datetime.fromisoformat(args.start).date() if args.start else None
    end_d   = datetime.fromisoformat(args.end).date() if args.end else None

    rr = asyncio.run(discover(SITEMAP_SEEDS, RSS_LIST_PAGES, start_d, end_d, args.limit))

    # Dedup với seen_urls.txt (nếu bật)
    if args.use_seen:
        seen = load_seen()
        fresh = []
        for u in rr:
            nu = normalize_url(u)
            if nu in seen:
                continue
            fresh.append(u)
            seen.add(nu)
        save_seen(seen)
        rr = fresh

    write_links_append_no_dup(rr)
