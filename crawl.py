import asyncio, aiohttp, json, hashlib, logging, gzip, random, os
from pathlib import Path
from datetime import datetime
from aiolimiter import AsyncLimiter
import tldextract

# ==================== CẤU HÌNH ====================
LINKS_FILE = Path("data/links.txt")
OUT_JSONL_GZ = Path("data/news_2025.jsonl.gz")       # << ghi jsonl nén
SEEN_URLS = Path("data/seen_urls.txt")
PROGRESS_FILE = Path("data/progress.json")
LOG_FILE = Path("data/crawler.log")

USER_AGENT = "NewsCrawler/1.0 (+mailto:redwhite7769@gmail.com)"
TIMEOUT = 20
CONCURRENCY = 80                    # có thể tăng nếu CPU/IO cho phép
RATE_WINDOW = 2                     # seconds
PER_DOMAIN_MAX = 3                  # ~1.5 rps/domain
MAX_RETRIES = 3
RETRY_DELAY = 2                     # seconds
WRITER_BATCH = 200                  # ghi theo lô để giảm I/O
RESULT_QUEUE_MAXSIZE = 2000         # đệm kết quả giữa worker -> writer

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==================== HELPER ====================
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def raw_ext_from_encoding(enc: str | None) -> str:
    enc = (enc or "").lower()
    if "br" in enc: return ".html.br"
    if "gzip" in enc: return ".html.gz"
    if "deflate" in enc: return ".html.deflate"
    return ".html"  # không nén từ server

# ==================== RATE LIMITER ====================
class DomainRateLimiter:
    def __init__(self, max_rate: int, time_period: int):
        self.max_rate = max_rate
        self.time_period = time_period
        self.limiters = {}
    def get_limiter(self, domain: str) -> AsyncLimiter:
        if domain not in self.limiters:
            self.limiters[domain] = AsyncLimiter(self.max_rate, self.time_period)
        return self.limiters[domain]

# ==================== STATS ====================
class CrawlStats:
    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.duplicate = 0
        self.start_time = datetime.now()
    def print_summary(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("📊 KẾT QUẢ CRAWL (RAW, SERVER-COMPRESSED WHEN POSSIBLE)")
        logger.info(f"⏱️  Thời gian: {elapsed:.1f}s ({elapsed/60:.1f} phút)")
        logger.info(f"📝 Tổng links: {self.total}")
        logger.info(f"✅ Thành công: {self.success}")
        logger.info(f"❌ Thất bại: {self.failed}")
        logger.info(f"🔄 Trùng lặp: {self.duplicate}")
        logger.info(f"⚡ Tốc độ: {self.total/elapsed:.2f} URLs/s")
        logger.info("=" * 60)

# ==================== FETCH (GIỮ BYTES NÉN) ====================
async def fetch_raw(session, url, limiter):
    """Trả về (bytes_payload, content_encoding) giữ nguyên nén của server."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        "Accept-Encoding": "br, gzip, deflate"   # << yêu cầu máy chủ nén
    }
    for attempt in range(MAX_RETRIES):
        async with limiter:
            try:
                async with session.get(url, headers=headers, allow_redirects=True) as r:
                    if r.status == 200:
                        enc = r.headers.get("Content-Encoding", "")
                        data = await r.read()   # << bytes, không giải nén
                        return data, enc
                    elif r.status in (429, 503):
                        if attempt < MAX_RETRIES - 1:
                            wait = RETRY_DELAY * (2 ** attempt) * (1 + random.random()*0.25)
                            logger.warning(f"⏳ {url[:60]}... - {r.status}, retry {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        return None, None
                    else:
                        logger.debug(f"❌ {url[:60]}... - Status {r.status}")
                        return None, None
            except asyncio.TimeoutError:
                logger.debug(f"⏱️ Timeout: {url[:60]}...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
            except Exception as e:
                logger.debug(f"❌ Error: {url[:60]}... - {type(e).__name__}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
    return None, None

# ==================== WRITER (ĐƠN LUỒNG, THEO LÔ) ====================
async def writer_task(result_q: asyncio.Queue, jsonl_path: Path, seen, stats):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(jsonl_path, "at", encoding="utf-8") as jout:  # <-- at
        batch = []
        while True:
            item = await result_q.get()
            if item is None:
                # flush batch cuối: ghi RAW rồi JSON
                for rec in batch:
                    raw_bytes = rec.pop("_raw_bytes")
                    raw_path = Path(rec["_raw_path"])
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(raw_path, "wb") as f:
                        f.write(raw_bytes)
                    jout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                result_q.task_done()
                break

            batch.append(item)
            if len(batch) >= WRITER_BATCH:
                for rec in batch:
                    raw_bytes = rec.pop("_raw_bytes")
                    raw_path = Path(rec["_raw_path"])
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(raw_path, "wb") as f:
                        f.write(raw_bytes)
                    jout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                batch.clear()
            result_q.task_done()

# ==================== WORKER ====================
async def worker(name, q_in, result_q, session, limiter_mgr, seen, stats):
    while True:
        url = await q_in.get()
        try:
            if url is None:
                return
            h = md5(url)
            if h in seen:
                stats.duplicate += 1
                continue
            seen.add(h)

            domain = get_domain(url)
            limiter = limiter_mgr.get_limiter(domain)

            raw_bytes, enc = await fetch_raw(session, url, limiter)
            if raw_bytes is None:
                stats.failed += 1
                continue

            day_dir = Path("raw_html") / datetime.now().strftime("%Y%m%d")
            raw_path = day_dir / f"{h}{raw_ext_from_encoding(enc)}"
            rec = {
                "url": url,
                "domain": domain,
                "_raw_path": str(raw_path),   # dùng tạm, sẽ giữ lại dưới tên raw_html_path
                "_raw_bytes": raw_bytes,      # để writer ghi file
                "content_encoding": enc or "",
                "crawled_at": datetime.now().isoformat()
            }
            # đưa cho writer
            await result_q.put({
                **rec,
                # bản ghi jsonl không chứa bytes:
                "raw_html_path": str(raw_path)
            })
            stats.success += 1
            if stats.success % 50 == 0:
                logger.info(f"✅ [{stats.success}] last: {url[:80]}...")
        except Exception as e:
            logger.error(f"❌ Lỗi xử lý {str(url)[:60]}...: {e}")
            stats.failed += 1
        finally:
            q_in.task_done()

# ==================== PROGRESS SAVER ====================
async def save_progress_periodically(seen, stats, interval=30):
    while True:
        await asyncio.sleep(interval)
        try:
            SEEN_URLS.write_text("\n".join(sorted(seen)), encoding="utf-8")
            PROGRESS_FILE.write_text(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "total_seen": len(seen),
                "success": stats.success,
                "failed": stats.failed
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"💾 Progress saved")
        except Exception as e:
            logger.error(f"Lỗi lưu progress: {e}")

# ==================== MAIN ====================
async def main():
    logger.info("🚀 BẮT ĐẦU CRAWL (server-compressed raw + JSONL.GZ)")
    Path("data").mkdir(exist_ok=True)

    if not LINKS_FILE.exists():
        logger.error(f"❌ Không tìm thấy {LINKS_FILE}")
        return

    # Lọc input
    links = [
        ln.strip() for ln in LINKS_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#") and ln.strip().startswith(("http://","https://"))
    ]
    logger.info(f"📂 Loaded {len(links)} links")

    # seen theo md5(url)
    seen = set()
    if SEEN_URLS.exists():
        seen = set(SEEN_URLS.read_text(encoding="utf-8").splitlines())
        logger.info(f"🔄 Loaded {len(seen)} seen URLs")

    stats = CrawlStats()
    stats.total = len(links)

    limiter_mgr = DomainRateLimiter(PER_DOMAIN_MAX, RATE_WINDOW)

    # Lưu ý: auto_decompress=False để giữ bytes nén của server
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=10, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        auto_decompress=False,        # << RẤT QUAN TRỌNG
        trust_env=True                # tôn trọng env proxy nếu có
    ) as session:
        q_in = asyncio.Queue()
        result_q = asyncio.Queue(maxsize=RESULT_QUEUE_MAXSIZE)

        for u in links:
            q_in.put_nowait(u)

        # writer đơn
        writer = asyncio.create_task(writer_task(result_q, OUT_JSONL_GZ, seen, stats))

        # workers
        workers = [
            asyncio.create_task(worker(f"W{i}", q_in, result_q, session, limiter_mgr, seen, stats))
            for i in range(CONCURRENCY)
        ]
        saver = asyncio.create_task(save_progress_periodically(seen, stats))

        # chờ input xử lý xong
        await q_in.join()
        # gửi sentinel cho workers
        for _ in workers:
            q_in.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)

        # gửi sentinel cho writer
        await result_q.put(None)
        await result_q.join()  # đợi writer flush batch
        await writer

        saver.cancel()
        try:
            await saver
        except asyncio.CancelledError:
            pass

    # save cuối
    SEEN_URLS.write_text("\n".join(sorted(seen)), encoding="utf-8")
    stats.print_summary()
    logger.info(f"✅ Xong! JSONL.GZ: {OUT_JSONL_GZ} | Raw: raw_html/")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("🛑 Interrupted")
