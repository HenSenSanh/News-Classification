import json, gzip, re, argparse
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

# ==================== ĐƯỜNG DẪN MẶC ĐỊNH ====================
IN_JSONL  = Path("data/news_2025.jsonl.gz")
OUT_JSONL = Path("data/traffic_2025.jsonl")
LOG_BAD   = Path("data/filter_bad_lines.log")

# ==================== TỪ KHÓA GIAO THÔNG ====================
TRAFFIC_KEYWORDS = [
    "giao thông","kẹt xe","ùn tắc","tắc đường","tai nạn","va chạm",
    "phân luồng","cấm đường","sửa chữa đường","kết nối giao thông",
    "cao tốc","vành đai","quốc lộ","tỉnh lộ","nút giao","cầu","hầm",
    "bến xe","xe buýt","metro","đường sắt đô thị","sân bay","cảng",
    "mộc bài","long thành","vành đai 3","vành đai 4","thủ thiêm","nhơn trạch"
]
URL_TRAFFIC_HINTS = ["giao-thong", "giao-thong-24h", "/traffic", "/transport", "metro", "cao-toc", "vanh-dai"]

# ==================== TỪ KHÓA TP.HCM ====================
HCM_HINTS = [
    # --- Tên gọi chung TP.HCM ---
    "TP[ .]?HCM", "TPHCM", "Thành phố Hồ Chí Minh",
    "Hồ Chí Minh", "Sài Gòn", "Ho Chi Minh City",
    "HCM City", "Saigon", "Thành phố HCM",

    # --- Thành phố trực thuộc ---
    "Thủ Đức",

    # --- Quận nội thành ---
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6",
    "Quận 7", "Quận 8", "Quận 10", "Quận 11", "Quận 12",
    "Bình Thạnh", "Phú Nhuận", "Tân Bình", "Tân Phú", "Gò Vấp",
    "Bình Tân",

    # --- Huyện ngoại thành ---
    "Bình Chánh", "Nhà Bè", "Củ Chi", "Hóc Môn", "Cần Giờ",

    # --- Tuyến đường lớn, cầu, hầm ---
    "Xa lộ Hà Nội", "Nguyễn Văn Linh", "Phạm Văn Đồng",
    "Võ Văn Kiệt", "Kinh Dương Vương", "Nguyễn Hữu Thọ",
    "Nguyễn Văn Trỗi", "Cộng Hòa", "Lý Thường Kiệt", "Trường Chinh",
    "Điện Biên Phủ", "Hoàng Văn Thụ", "Trần Hưng Đạo",
    "Nam Kỳ Khởi Nghĩa", "Hồng Bàng", "Lê Văn Sỹ", "Nguyễn Thị Minh Khai",

    # --- Cầu / hầm / tuyến ---
    "Cầu Thủ Thiêm", "Cầu Sài Gòn", "Cầu Phú Mỹ",
    "Cầu Bình Triệu", "Cầu Ông Lãnh", "Cầu Nguyễn Tri Phương",
    "Cầu chữ Y", "Cầu Chà Và", "Cầu vượt Cộng Hòa",
    "Hầm Thủ Thiêm", "Hầm vượt sông Sài Gòn",

    # --- Khu vực đặc trưng ---
    "Ngã tư An Sương", "Ngã tư Hàng Xanh", "Ngã sáu Gò Vấp",
    "Bến xe Miền Đông", "Bến xe Miền Tây",
    "Sân bay Tân Sơn Nhất", "Ga Sài Gòn",
    "Khu công nghệ cao", "KCN Tân Bình", "KCN Hiệp Phước",

    # --- Dự án / tuyến giao thông lớn ---
    "Metro số 1", "Tuyến metro Bến Thành - Suối Tiên",
    "Vành đai 2", "Vành đai 3", "Vành đai 4",
    "Cao tốc Long Thành", "Cao tốc Mộc Bài",
    "Cao tốc TP.HCM - Trung Lương", "Cao tốc TP.HCM - Long Thành - Dầu Giây",

    # --- Cụm từ đặc trưng khác ---
    "cửa ngõ phía Đông", "cửa ngõ phía Tây", "cửa ngõ phía Nam", "cửa ngõ phía Bắc",
    "trung tâm thành phố", "nội thành", "ngoại thành",
    "UBND TP.HCM", "Sở GTVT TP.HCM", "Ban Quản lý dự án giao thông TP.HCM"
]

# ==================== HÀM TIỆN ÍCH ====================
def has_any(text: str, keywords) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)

def read_html_auto(path: Path) -> str:
    """Đọc file HTML, tự nhận diện nén hay không (gzip hoặc thường)."""
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        if head == b"\x1f\x8b":  # gzip header
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fin:
                return fin.read()
        else:
            with open(path, "rt", encoding="utf-8", errors="ignore") as fin:
                return fin.read()
    except Exception as e:
        raise e

# ==================== TRÍCH XUẤT TIÊU ĐỀ & NGÀY ====================
def extract_title_and_date(soup: BeautifulSoup):
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"):
        title = ogt["content"].strip() or title

    date_candidates = []
    for sel in [
        {"property": "article:published_time"},
        {"property": "article:modified_time"},
        {"name": "pubdate"},
        {"itemprop": "datePublished"},
        {"name": "date"},
        {"property": "og:updated_time"},
    ]:
        m = soup.find("meta", **sel)
        if m and m.get("content"):
            date_candidates.append(m["content"].strip())

    for t in soup.find_all("time"):
        if t.get("datetime"):
            date_candidates.append(t["datetime"].strip())
        elif t.get_text(strip=True):
            date_candidates.append(t.get_text(strip=True))

    pub_year = None
    pub_iso  = ""
    for raw in date_candidates:
        s = raw.replace("Z","+00:00")
        try:
            dt = datetime.fromisoformat(s) if "T" in s or "-" in s else None
        except Exception:
            dt = None
        if not dt:
            m1 = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", raw)
            m2 = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", raw)
            if m1:
                y = int(m1.group(1)); pub_year = y; pub_iso = f"{y}-{int(m1.group(2)):02d}-{int(m1.group(3)):02d}"
                break
            if m2:
                y = int(m2.group(3)); pub_year = y; pub_iso = f"{y}-{int(m2.group(2)):02d}-{int(m2.group(1)):02d}"
                break
        else:
            pub_year = dt.year
            pub_iso  = dt.date().isoformat()
            break
    return title, pub_year, pub_iso

# ==================== TRÍCH XUẤT NỘI DUNG ====================
def extract_text(soup: BeautifulSoup):
    selectors = [
        "article p",".article p",".news-content p",".fck_detail p",
        ".detail-content p",".post-content p","div[class*='content'] p","p"
    ]
    for css in selectors:
        ps = soup.select(css)
        if ps:
            parts = []
            for p in ps:
                txt = p.get_text(" ", strip=True)
                if len(txt) > 20:
                    parts.append(txt)
            if parts:
                return " ".join(parts)
    return soup.get_text(" ", strip=True)[:5000]

# ==================== ĐIỀU KIỆN GIỮ BÀI ====================
def keep_article(url, title, body, pub_year, hcm_only=False):
    if pub_year != 2025:
        if "2025" not in url:
            return False
    text_join = f"{title} {body}"
    is_traffic = has_any(text_join, TRAFFIC_KEYWORDS) or has_any(url, URL_TRAFFIC_HINTS)
    if not is_traffic:
        return False
    if hcm_only:
        is_hcm = has_any(text_join, HCM_HINTS) or has_any(url, ["tp-hcm","tphcm","ho-chi-minh"])
        if not is_hcm:
            return False
    return True

# ==================== CHƯƠNG TRÌNH CHÍNH ====================
def main(inp=IN_JSONL, outp=OUT_JSONL, hcm_only=False):
    outp.parent.mkdir(parents=True, exist_ok=True)
    kept = bad = total = 0
    with gzip.open(inp, "rt", encoding="utf-8") as fin, \
         outp.open("w", encoding="utf-8") as fout, \
         LOG_BAD.open("w", encoding="utf-8") as fbad:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                fbad.write(line + "\n")
                continue

            raw_path = obj.get("raw_html_path","")
            url      = obj.get("url","")
            if not raw_path or not Path(raw_path).exists():
                bad += 1
                fbad.write(json.dumps({"reason":"missing_raw","obj":obj}, ensure_ascii=False) + "\n")
                continue

            try:
                html = read_html_auto(Path(raw_path))
                soup = BeautifulSoup(html, "html.parser")
                title, pub_year, pub_iso = extract_title_and_date(soup)
                body = extract_text(soup)

                if keep_article(url, title, body, pub_year, hcm_only=hcm_only):
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
            except Exception as e:
                bad += 1
                fbad.write(json.dumps({"reason":str(e), "obj":obj}, ensure_ascii=False) + "\n")

    print(f"✅ Done. total={total}, kept={kept}, bad={bad}, out={outp}")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(IN_JSONL))
    ap.add_argument("--output", type=str, default=str(OUT_JSONL))
    ap.add_argument("--hcm-only", action="store_true", help="Chỉ giữ bài liên quan TP.HCM")
    args = ap.parse_args()
    main(Path(args.input), Path(args.output), hcm_only=args.hcm_only)
