import json
from pathlib import Path
from tqdm.auto import tqdm

# ==================== ĐƯỜNG DẪN ====================
IN_JSONL  = Path("data/traffic_2025.jsonl")
OUT_JSONL = Path("data/traffic_2025_clean.jsonl")

# ==================== TỪ KHÓA GIỮ & LOẠI ====================
# Tin giao thông thực địa
KEEP_KEYWORDS = [
    "kẹt xe","ùn tắc","ùn ứ","tai nạn","va chạm","phân luồng",
    "đi lại khó khăn","ngập nước","kẹt đường","hư hỏng","xe container",
    "xe tải","xe máy","xe buýt","đường ngập","tắc nghẽn","sửa chữa",
    "hầm chui","cầu","nút giao","thủ thiêm","vành đai","cao tốc","xe cứu thương",
]

# Tin quy hoạch / chính sách / kinh tế (nên loại)
DENY_KEYWORDS = [
    "dự án","khởi công","đầu tư","quy hoạch","phê duyệt","tổng mức đầu tư",
    "vốn đầu tư","gói thầu","hạ tầng","nhà đầu tư","đấu thầu","chủ đầu tư",
    "nghiên cứu khả thi","kêu gọi đầu tư","mở rộng tuyến","chính sách","nghị quyết",
    "đề án","thẩm định","bố trí vốn","bàn giao mặt bằng","giải ngân",
    "Ban quản lý dự án","BQL","quyết định","công bố kế hoạch","họp báo",
]

# ==================== HÀM PHỤ ====================
def has_any(text: str, keywords) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)

# ==================== MAIN ====================
def main(inp=IN_JSONL, outp=OUT_JSONL):
    outp.parent.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    examples = []

    with open(inp, "r", encoding="utf-8") as fin, \
         open(outp, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Lọc lớp 2", unit="bài", dynamic_ncols=True):
            total += 1
            try:
                obj = json.loads(line)
                text = f"{obj.get('title','')} {obj.get('preview','')}"
            except Exception:
                continue

            # 1️⃣ loại tin quá ngắn
            if len(text) < 300:
                continue

            # 2️⃣ loại nếu chứa nhiều keyword quy hoạch / kinh tế mà không có keyword giao thông thực địa
            if has_any(text, DENY_KEYWORDS) and not has_any(text, KEEP_KEYWORDS):
                continue

            # 3️⃣ ưu tiên bài có dấu hiệu sự kiện thật
            if not has_any(text, KEEP_KEYWORDS):
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            if len(examples) < 5:
                examples.append(obj.get("url",""))

    print(f"\n✅ DONE. Tổng {total:,} | Giữ lại {kept:,}")
    if examples:
        print("Ví dụ bài giữ lại:")
        for u in examples:
            print(" -", u)
    else:
        print("⚠️ Không có bài phù hợp — thử nới keyword KEEP hoặc giảm DENY.")

if __name__ == "__main__":
    main()
