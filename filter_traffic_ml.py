import json, argparse, os
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ============== ĐƯỜNG DẪN ==============
DEFAULT_IN  = Path("data/traffic_2025_clean.jsonl")   # lớp 2
ALT_IN      = Path("data/traffic_2025.jsonl")         # lớp 1 (fallback)
OUT_JSONL   = Path("data/traffic_2025_final.jsonl")
OUT_REPORT  = Path("data/traffic_ml_report.txt")
OUT_MODEL   = Path("data/traffic_ml_model.joblib")
OUT_VECT    = Path("data/traffic_ml_vectorizer.joblib")

# ============== TỪ KHÓA GÁN NHÃN YẾU (WEAK LABELS) ==============
# Dấu hiệu "sự kiện giao thông" (positive)
POS_KEYS = [
    "kẹt xe","kẹt cứng","kẹt đường","ùn tắc","ùn ứ","tắc đường",
    "tai nạn","tai nạn giao thông","va chạm","lật xe","lật container","đâm vào",
    "phân luồng","phong tỏa","cấm đường tạm thời","đi lại khó khăn",
    "ngập nước","ngập sâu","lụt","ngập đường","mưa lớn gây ngập",
    "hư hỏng mặt đường","sập hố ga","sụt lún",
    "xe container","xe tải","xe buýt","xe máy","ô tô",
    "kẹt xe kéo dài","tắc nghẽn kéo dài","ùn ứ kéo dài",
    "kẹt xe giờ cao điểm","giải cứu kẹt xe","xử lý ùn tắc",
    "va chạm liên hoàn","tự té xe","băng qua đường","đâm vào dải phân cách"
]

# Dấu hiệu "quy hoạch/chính sách/kinh tế" (negative)
NEG_KEYS = [
    "dự án","khởi công","động thổ","đầu tư","quy hoạch","điều chỉnh quy hoạch",
    "phê duyệt","đề án","đề xuất","chủ trương","nghị quyết","chính sách",
    "tổng mức đầu tư","nguồn vốn","giải ngân","bố trí vốn","gói thầu","đấu thầu",
    "nhà đầu tư","báo cáo nghiên cứu khả thi","FS","PPP",
    "ban quản lý dự án","BQL dự án","ban QLDA","tư vấn thiết kế",
    "mở rộng tuyến","cao tốc (dự án)","tuyến metro (dự án)","bàn giao mặt bằng",
    "giải phóng mặt bằng","quỹ đất","quy trình","thủ tục"
]

# ============== HÀM PHỤ ==============
def norm(s: str) -> str:
    return (s or "").lower()

def has_any(text: str, keys) -> bool:
    t = norm(text)
    return any(k in t for k in keys)

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def compose_text(obj):
    return f"{obj.get('title','')} {obj.get('preview','')}"

# ============== GÁN NHÃN YẾU (WEAK LABELING) ==============
def weak_label(text: str):
    """
    Trả về:
      1: tin sự kiện giao thông rõ (positive)
      0: tin quy hoạch/chính sách rõ (negative)
     -1: không chắc (bỏ qua khi huấn luyện)
    """
    pos = has_any(text, POS_KEYS)
    neg = has_any(text, NEG_KEYS)

    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    # nếu cả hai cùng xuất hiện: xem ưu tiên theo mật độ từ khóa
    if pos and neg:
        # đếm mật độ để phân xử
        t = norm(text)
        pc = sum(t.count(k) for k in POS_KEYS)
        nc = sum(t.count(k) for k in NEG_KEYS)
        if pc >= nc + 1:
            return 1
        if nc >= pc + 1:
            return 0
        return -1
    return -1

# ============== MAIN ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(DEFAULT_IN), help="Đường dẫn input JSONL (mặc định lớp 2)")
    ap.add_argument("--output", type=str, default=str(OUT_JSONL))
    ap.add_argument("--threshold", type=float, default=0.55, help="Ngưỡng xác suất giữ lại (0..1)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[!] Không tìm thấy {inp}. Thử fallback {ALT_IN} ...")
        inp = ALT_IN
    assert inp.exists(), f"Không có input: {inp}"

    out_jsonl = Path(args.output)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    # 1) Nạp dữ liệu
    print(f"[IN ] {inp.resolve()}")
    print(f"[OUT] {out_jsonl.resolve()}")
    print(f"[CFG] threshold = {args.threshold}\n")

    records = list(load_jsonl(inp))
    texts   = [compose_text(o) for o in records]
    urls    = [o.get("url","") for o in records]

    # 2) Tạo nhãn yếu
    weak_y = [weak_label(t) for t in texts]
    labeled_idx = [i for i,y in enumerate(weak_y) if y in (0,1)]
    if len(labeled_idx) < 200:
        print(f"[!] Quá ít mẫu gán nhãn yếu (n={len(labeled_idx)}). Hãy nới POS/NEG_KEYS hoặc dùng input rộng hơn.")
        # vẫn tiếp tục nhưng kết quả có thể kém
    X_lab = [texts[i] for i in labeled_idx]
    y_lab = np.array([weak_y[i] for i in labeled_idx])

    # 3) Vector hoá: char-ngrams giúp tiếng Việt không cần tokenizer
    vect = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,6),
        min_df=1, max_df=1.0, sublinear_tf=True
    )

    # 4) Chia train/val để xem chất lượng trên nhãn yếu
    if len(X_lab) >= 100:
        Xtr, Xva, ytr, yva = train_test_split(X_lab, y_lab, test_size=0.2, random_state=42, stratify=y_lab)
    else:
        Xtr, Xva, ytr, yva = X_lab, [], y_lab, []

    Xtr_vec = vect.fit_transform(Xtr)
    clf = LogisticRegression(
        max_iter=2000, n_jobs=1, class_weight="balanced",
        C=3.0, solver="liblinear"
    )
    clf.fit(Xtr_vec, ytr)

    # 5) Báo cáo trên tập val (nếu có)
    report_txt = ""
    if len(Xva) > 0:
        Xva_vec = vect.transform(Xva)
        ypred = clf.predict(Xva_vec)
        report_txt = classification_report(yva, ypred, digits=3, target_names=["NEG (policy)","POS (event)"])
        print("\n[VAL] Báo cáo trên nhãn yếu (chỉ để tham khảo):\n")
        print(report_txt)

    # Lưu model + vectorizer
    joblib.dump(clf, OUT_MODEL)
    joblib.dump(vect, OUT_VECT)
    if report_txt:
        OUT_REPORT.write_text(report_txt, encoding="utf-8")

    # 6) Train lại trên toàn bộ nhãn yếu để tận dụng dữ liệu
    Xall_vec = vect.fit_transform(X_lab)
    clf.fit(Xall_vec, y_lab)

    # 7) Dự đoán cho TOÀN BỘ BẢN GHI
    Xfull_vec = vect.transform(texts)
    if hasattr(clf, "predict_proba"):
        pos_prob = clf.predict_proba(Xfull_vec)[:,1]
    else:
        # Fallback cho model không có proba (LinearSVC). Ở đây dùng LR nên thường không vào nhánh này.
        scores = clf.decision_function(Xfull_vec)
        # scale về 0..1 tương đối
        pos_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    thr = float(args.threshold)
    keep_mask = pos_prob >= thr

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for i, keep in enumerate(keep_mask):
            if keep:
                rec = records[i].copy()
                rec["_ml_pos_prob"] = float(pos_prob[i])
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"\n✅ DONE. Tổng {len(records):,} | Giữ {kept:,} (threshold={thr})")
    print(f"Model: {OUT_MODEL} | Vectorizer: {OUT_VECT}")
    if OUT_REPORT.exists():
        print(f"Report (nhãn yếu): {OUT_REPORT}")
    # Gợi ý: tăng/giảm --threshold để siết/lỏng
    if kept < 50:
        print("⚠️ Ít mẫu được giữ. Hãy giảm --threshold (ví dụ 0.5) hoặc mở rộng POS_KEYS.")
    elif kept > 0.9 * len(records):
        print("⚠️ Giữ quá nhiều. Hãy tăng --threshold (ví dụ 0.65) hoặc mở rộng NEG_KEYS.")

if __name__ == "__main__":
    # đảm bảo stdout utf-8 trên Windows
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
