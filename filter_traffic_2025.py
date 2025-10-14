import json, gzip, re, os
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm.auto import tqdm

# ==================== CẤU HÌNH ====================
IN_JSONL  = Path("data/news_giaothog_2025.jsonl.gz")
OUT_JSONL = Path("data/traffic_2025.jsonl")
LOG_BAD   = Path("data/filter_bad_lines.log")

# ==================== TỪ KHÓA ====================
TRAFFIC_KEYWORDS = [
    # --- giao thông chung ---
    "giao thông","kẹt xe","ùn tắc","ùn ứ","tắc đường","kẹt đường",
    "tai nạn","va chạm","phân luồng","thi công","sửa chữa","cấm đường",
    "cao tốc","vành đai","quốc lộ","tỉnh lộ","nút giao","đường","cầu","hầm",
    "xe buýt","metro","vận tải","bến xe","bến phà","sân bay","cảng",
    "hạ tầng","ngập nước","trạm thu phí","BOT","hầm chui",
    "xe container","xe tải","tai nạn giao thông","ùn ứ kéo dài",
    "thủ thiêm","vành đai 3","vành đai 4","ngã tư","ngã sáu",
]

URL_TRAFFIC_HINTS = [
    "giao-thong","traffic","transport","cao-toc","vanh-dai",
    "metro","duong","do-thi","ha-tang","ham","cau-"
]

HCM_HINTS = [
    # --- Tên TP.HCM ---
    "TP HCM","TP.HCM","TPHCM","HCM","Thành phố Hồ Chí Minh","Ho Chi Minh",
    "Hồ Chí Minh","Ho Chi Minh City","HCM City","Saigon","Sài Gòn",
    "Thành phố HCM","Thủ Đức",

    # --- Quận huyện ---
    "Quận 1","Quận 2","Quận 3","Quận 4","Quận 5","Quận 6","Quận 7","Quận 8",
    "Quận 9","Quận 10","Quận 11","Quận 12","Bình Thạnh","Phú Nhuận","Tân Bình",
    "Tân Phú","Gò Vấp","Bình Tân","Bình Chánh","Nhà Bè","Củ Chi","Hóc Môn","Cần Giờ",

    # --- Tuyến đường / cầu / hầm ---
    "Võ Văn Kiệt","Phạm Văn Đồng","Nguyễn Văn Linh","Trường Chinh","Điện Biên Phủ",
    "Nam Kỳ Khởi Nghĩa","Cộng Hòa","Lý Thường Kiệt","Nguyễn Hữu Thọ",
    "Cầu Phú Mỹ","Cầu Sài Gòn","Cầu Thủ Thiêm","Hầm Thủ Thiêm","Ngã tư An Sương",

    # --- Địa danh đặc trưng ---
    "Sân bay Tân Sơn Nhất","Bến xe Miền Đông","Bến xe Miền Tây",
    "KCN Tân Bình","KCN Hiệp Phước","UBND TP.HCM","Sở GTVT TP.HCM",
    "Ban Quản lý dự án giao thông TP.HCM","Vành đai 3","Vành đai 4",
]

# ==================== HÀM HỖ TRỢ ====================
def has_any(text: str, keywords) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)

def read_html_auto(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":  # gzip magic
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fin:
            return fin.read()
    else:
        with open(path, "rt", encoding="utf-8", errors="ignore") as fin:
            return fin.read()

def extract_title_and_date(soup: BeautifulSoup):
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"):
        title = ogt["content"].strip() or title

    pub_year, pub_iso = None, ""
    for m in soup.find_all("meta", attrs={"property": "article:published_time"}):
        val = m.get("content")
        if not val: continue
        try:
            dt = datetime.fromisoformat(val.replace("Z","+00:00"))
            pub_year, pub_iso = dt.year, dt.date().isoformat()
            break
        except Exception:
            pass
    return title, pub_year, pub_iso

def extract_text(soup: BeautifulSoup):
    for css in ["article p",".article p",".fck_detail p",".post-content p",".detail-content p"]:
        ps = soup.select(css)
        if ps:
            parts = [p.get_text(" ", strip=True) for p in ps if len(p.get_text(strip=True))>20]
            if parts: return " ".join(parts)
    return soup.get_text(" ", strip=True)[:5000]

def keep_article(url, title, body, pub_year):
    text_join = f"{title} {body}"

    # nới năm: bỏ nếu quá cũ
    if pub_year and pub_year < 2024:
        return False

    # phải là bài giao thông
    if not (has_any(text_join, TRAFFIC_KEYWORDS) or has_any(url, URL_TRAFFIC_HINTS)):
        return False

    # phải liên quan TP.HCM
    if not (has_any(text_join, HCM_HINTS) or has_any(url, ["tp-hcm","tphcm","ho-chi-minh","do-thi","giao-thong"])):
        return False

    return True

# ==================== MAIN ====================
def main():
    inp, outp = IN_JSONL.resolve(), OUT_JSONL.resolve()
    LOG_BAD.parent.mkdir(parents=True, exist_ok=True)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = kept = bad = 0
    examples = []

    print(f"[IN ] {inp}")
    print(f"[OUT] {outp}")
    print(f"[MODE] TP.HCM giao thông (2024–2025)\n")

    with gzip.open(inp, "rt", encoding="utf-8") as fin, \
         outp.open("w", encoding="utf-8") as fout, \
         LOG_BAD.open("w", encoding="utf-8") as fbad:

        for line in tqdm(fin, desc="Lọc bài", unit="bài", dynamic_ncols=True):
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            url = obj.get("url","")
            raw_path = obj.get("raw_html_path","")
            if not raw_path or not Path(raw_path).exists():
                bad += 1
                continue

            try:
                html = read_html_auto(Path(raw_path))
                soup = BeautifulSoup(html, "html.parser")
                title, pub_year, pub_iso = extract_title_and_date(soup)
                body = extract_text(soup)

                if keep_article(url, title, body, pub_year):
                    rec = {
                        "url": url,
                        "title": title,
                        "pub_date": pub_iso,
                        "year": pub_year,
                        "preview": body[:400],
                        "raw_html_path": raw_path
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if len(examples) < 5: examples.append(url)
            except Exception as e:
                bad += 1
                fbad.write(json.dumps({"reason": str(e), "url": url}, ensure_ascii=False)+"\n")

    print(f"\n✅ DONE. Tổng {total:,} | Giữ {kept:,} | Lỗi {bad:,}")
    if examples:
        print("\nVí dụ bài giữ lại:")
        for u in examples:
            print(" -", u)
    else:
        print("⚠️ Không có bài phù hợp — thử mở rộng keyword hoặc kiểm tra file raw_html.")

if __name__ == "__main__":
    main()
