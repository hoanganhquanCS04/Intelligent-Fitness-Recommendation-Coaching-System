"""
src/rag/text_cleaner.py
=======================
Step 1.3 — Text Cleaning & Normalization

Pipeline:
    clean_document(doc)          ← dispatcher chọn pipeline theo doc_type
        ├── clean_book_text()    ← Tier 1: xử lý Scribd artifacts + OCR noise
        ├── clean_blog_text()    ← Tier 3: xử lý nav menu, sidebar, footer
        └── (pubmed: bỏ qua — đã sạch từ XML parser)
    mọi pipeline đều chạy clean_text_universal() ở cuối

Chuẩn tham chiếu:
    - Unicode NFC normalization (chuẩn VinAI / HuggingFace datasets)
    - Hugging Face preprocessing pipeline whitespace normalization
"""

import re
import unicodedata
import json
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# LAYER 0 — Universal cleaner (mọi nguồn)
# ─────────────────────────────────────────────

def clean_text_universal(text: str) -> str:
    """
    Làm sạch text cơ bản — áp dụng cho MỌI loại nguồn.

    Thứ tự xử lý:
    1. Unicode NFC normalize  → bắt buộc cho tiếng Việt
    2. Normalize line endings → \r\n → \n
    3. Xóa ký tự control     → OCR/encoding artifacts
    4. Fix hyphenation        → word-\\nbreak → wordbreak
    5. Xóa dòng chỉ có số    → số trang PDF
    6. Normalize spaces       → multiple spaces → 1 space
    7. Collapse blank lines   → ≤ 2 dòng trống liên tiếp
    8. Strip toàn bộ
    """
    # 1. NFC normalize — BẮT BUỘC cho tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Xóa ký tự control (giữ lại \n và \t để xử lý sau)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # 4. Fix hyphenation: "per-\nformance" → "performance"
    text = re.sub(r"-[ \t]*\n[ \t]*", "", text)

    # 5. Xóa dòng chỉ có số (số trang PDF/Scribd)
    text = re.sub(r"(?m)^[ \t]*\d+[ \t]*$", "", text)

    # 6. Normalize horizontal whitespace: multiple spaces/tabs → 1 space
    text = re.sub(r"[ \t]+", " ", text)

    # 7. Trim whitespace cuối mỗi dòng
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 8. Collapse nhiều dòng trống → tối đa 2 dòng trống (= 1 paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# LAYER 1A — Book cleaner (Tier 1: PDF/OCR → txt)
# ─────────────────────────────────────────────

# Các pattern đặc thù của Scribd web reader
_SCRIBD_TIMESTAMP = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s+[AP]M\s+Scribd",
    re.IGNORECASE
)
_SCRIBD_PAGE_COUNTER = re.compile(
    r"(?m)^\s*\d+/\d+\s*$"       # "42/423" trên dòng riêng
)
_SCRIBD_URL_FRAGMENT = re.compile(
    r"https?:/[A-Za-z]\s+\d+/\d+"   # "https:/A 52/423"
)
_SCRIBD_COUNTER_INLINE = re.compile(
    r"\b\d{1,4}/\d{3,4}\b"       # inline page counter như "42/423"
)



def _is_ocr_garbage_line(line: str) -> bool:
    """
    Phát hiện dòng OCR thất bại hoàn toàn.

    Heuristic: dòng được coi là garbage nếu ≥ 1 trong các điều kiện:
    1. Tỷ lệ ký tự hữu ích < 40% (noise chars chiếm đa số)
    2. Chứa ≥ 2 ký tự pipe/backtick/bracket liên tiếp (box-drawing OCR artifact)
    3. Chứa ≥ 4 chữ cái đơn lẻ cách nhau bằng space (word-split OCR artifact)

    Ví dụ garbage:
        "| # Startin ` Strengt B Sic raining B arb Ahh x Rippeto e"  → rule 2+3
        "ÿ ĐH TT008010LÄ01303141300014088303304G1400188"             → rule 1
        "ae :  =  ¬ \" _ MS .w a wa _ sư"                            → rule 1
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 5:
        return False          # dòng rỗng / quá ngắn → để universal xử lý

    if len(stripped) > 150:
        return False          # dòng rất dài → khả năng cao là văn bản thực

    # Rule 1: tỷ lệ ký tự hữu ích thấp
    useful = sum(
        1 for c in stripped
        if c.isalnum()
        or c in ".,!?;:()\"' -"
        or "\u00C0" <= c <= "\u024F"   # Latin Extended
        or "\u1E00" <= c <= "\u1EFF"   # Latin Extended Additional (tiếng Việt)
    )
    ratio = useful / len(stripped)
    if ratio < 0.40:
        return True

    # Rule 2: box-drawing / pipe / scattered special characters
    special_chars = sum(1 for c in stripped if not c.isalnum() and c not in ".,!?;:()\"' -")
    # Nếu dòng ngắn (< 100 ký tự) và chứa > 5% ký tự đặc biệt (VD: | # ` & Ÿ)
    if len(stripped) < 100 and (special_chars / max(1, len(stripped))) > 0.05:
        return True

    return False


def clean_book_text(text: str) -> str:
    """
    Cleaning đặc thù cho sách PDF (Tier 1) — xử lý Scribd artifacts + OCR noise.

    Áp dụng cho: processed/books/*.txt

    Các bước:
    1. Xóa Scribd timestamp header ("3/28/26, 5:05 PM Scribd")
    2. Xóa Scribd page counter dạng "42/423" (standalone line)
    3. Xóa Scribd URL fragment ("https:/A 52/423")
    4. Xóa inline page counter còn sót ("... 42/423 ...")
    5. Lọc dòng OCR garbage (tỷ lệ ký tự hữu ích < 40%)
    6. Chạy clean_text_universal()
    """
    # Bước 1: Xóa Scribd timestamp
    text = _SCRIBD_TIMESTAMP.sub("", text)

    # Bước 2: Xóa page counter dạng standalone line "42/423"
    text = _SCRIBD_PAGE_COUNTER.sub("", text)

    # Bước 3: Xóa URL fragment Scribd
    text = _SCRIBD_URL_FRAGMENT.sub("", text)

    # Bước 4: Xóa inline page counter còn sót trong câu
    # (chỉ áp dụng pattern N/NNN có vẻ là page counter, không phải fraction)
    text = _SCRIBD_COUNTER_INLINE.sub("", text)

    # Bước 5: Lọc từng dòng — bỏ OCR garbage
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if _is_ocr_garbage_line(line):
            clean_lines.append("")      # giữ vị trí dòng (universal sẽ collapse)
        else:
            clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Bước 6: Universal cleaning để hoàn thiện
    return clean_text_universal(text)


# ─────────────────────────────────────────────
# LAYER 1B — Blog cleaner (Tier 3: HTML → txt)
# ─────────────────────────────────────────────

# Các pattern đặc trưng của navigation menu / sidebar / footer trong blog
_NAV_MENU_INDICATORS = re.compile(
    r"(Đăng nhập|Tìm Phòng Gym|Trang chủ|Mua ngay|Hotline|iFitness)"
)
_PRODUCT_PRICE = re.compile(r"\d[\d\.]+\s*vnđ", re.IGNORECASE)
_SOCIAL_SHARE = re.compile(r"(?i)(facebook|twitter|pinterest|whatsapp|zalo)")
_FOOTER_CONTACT = re.compile(
    r"(?i)(hotline|địa chỉ|click vào đây|click truy cập|tư vấn kĩ hơn)"
)
_RATING_LINE = re.compile(r"^\s*\d+\.\d+\s*$")   # "9.8", "8.5" — product ratings


def _detect_nav_block(lines: list[str], start: int,
                      min_block: int = 5, max_line_len: int = 60) -> int:
    """
    Từ vị trí `start`, phát hiện block navigation menu.

    Heuristic: navigation menu = ≥ `min_block` dòng LIÊN TIẾP mà:
    - Mỗi dòng có độ dài ≤ `max_line_len` ký tự (sau strip)
    - Dòng không chứa dấu câu kết câu (., !, ?)  — trừ dấu trong tên viết tắt
    - Không phải dòng trống

    Trả về: chỉ số dòng kết thúc của block (exclusive), hoặc start nếu không tìm thấy.
    """
    i = start
    n = len(lines)
    consecutive = 0

    while i < n:
        line = lines[i].strip()
        # Dòng trống: reset counter
        if not line:
            i += 1
            consecutive = 0
            continue

        is_short = len(line) <= max_line_len
        has_sentence_end = bool(re.search(r"[.!?]{1}\s+[A-ZĐÀÁẢÃẠĂẮẶẲẴẦẤẬẨẪ]", line))
        # Dòng ngắn và không phải câu hoàn chỉnh → nav item
        if is_short and not has_sentence_end:
            consecutive += 1
        else:
            break
        i += 1

    if consecutive >= min_block:
        return i       # trả về endpoint của block
    return start       # không phải block nav





def clean_blog_text(text: str) -> str:
    """
    Cleaning đặc thù cho blog HTML (Tier 3).

    Áp dụng cho: processed/blogs/*.txt

    Thứ tự (QUAN TRỌNG: nav block phải xử lý TRƯỚC universal):
    1. Xoá toàn bộ Navigation Menu/Sidebar bằng heuristic block detection
    2. Xóa dòng chứa giá sản phẩm ("2.125.000vnđ") và rating ("9.8")
    3. Xóa dòng social share (Facebook, Twitter, Pinterest, WhatsApp)
    4. Xóa footer contact block (Hotline, địa chỉ, Click vào đây)
    5. Xóa dòng "Mua ngay"
    6. Clean_text_universal()
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    n = len(lines)
    i = 0
    skip_footer = False

    while i < n:
        # 1. Phát hiện khối Menu điều hướng dài (>= 8 dòng)
        nav_end = _detect_nav_block(lines, i, min_block=8)
        if nav_end > i:
            i = nav_end  # Bỏ qua toàn bộ khối menu
            continue
            
        line = lines[i]
        stripped = line.strip()

        # Dừng skip khi gặp dòng trống sau footer
        if skip_footer:
            if not stripped:
                skip_footer = False
            i += 1
            continue

        # Phát hiện bắt đầu footer contact → skip đến dòng trống tiếp theo
        if _FOOTER_CONTACT.search(stripped):
            skip_footer = True
            i += 1
            continue

        # Xóa dòng rating đơn thuần
        if _RATING_LINE.match(stripped):
            i += 1
            continue

        # Xóa dòng giá sản phẩm
        if _PRODUCT_PRICE.search(stripped):
            i += 1
            continue

        # Xóa dòng social share
        if _SOCIAL_SHARE.search(stripped) and len(stripped) <= 30:
            i += 1
            continue

        # Xóa "Mua ngay"
        if stripped.lower() == "mua ngay":
            i += 1
            continue

        cleaned_lines.append(line)
        i += 1

    text = "\n".join(cleaned_lines)

    # Bước 7: Universal cleaning để hoàn thiện
    return clean_text_universal(text)


# ─────────────────────────────────────────────
# DISPATCHER — Chọn pipeline theo doc_type
# ─────────────────────────────────────────────

def clean_document(doc: dict) -> dict:
    """
    Dispatches cleaning pipeline dựa vào doc["metadata"]["doc_type"].

    Args:
        doc: LangChain Document dict {page_content, metadata}

    Returns:
        doc mới với page_content đã được làm sạch
        (metadata giữ nguyên, chỉ thêm "cleaned": True)
    """
    doc_type = doc.get("metadata", {}).get("doc_type", "unknown")
    text = doc.get("page_content", "")

    if doc_type == "book":
        cleaned = clean_book_text(text)
    elif doc_type == "blog":
        cleaned = clean_blog_text(text)
    elif doc_type == "pubmed":
        # PubMed đã sạch từ XML parser — chỉ chạy universal để đảm bảo NFC
        cleaned = clean_text_universal(text)
    else:
        # Fallback: universal only
        cleaned = clean_text_universal(text)

    return {
        "page_content": cleaned,
        "metadata": {**doc["metadata"], "cleaned": True}
    }


# ─────────────────────────────────────────────
# FILE I/O — Đọc / Ghi processed .txt files
# ─────────────────────────────────────────────

def _parse_txt_header(text: str) -> dict:
    """
    Đọc metadata header từ file .txt (do document_loader.py tạo ra).

    Format:
        # SOURCE: filename.pdf
        # TIER: tier1
        # LANGUAGE: en
        # DOC_TYPE: book
        # ─────...

    Returns:
        {"source": ..., "tier": ..., "language": ..., "doc_type": ...,
         "body_start": <index của dòng đầu tiên sau header>}
    """
    lines = text.split("\n")
    meta = {}
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("# SOURCE:"):
            meta["source"] = line.split(":", 1)[1].strip()
        elif line.startswith("# TIER:"):
            meta["tier"] = line.split(":", 1)[1].strip()
        elif line.startswith("# LANGUAGE:"):
            meta["language"] = line.split(":", 1)[1].strip()
        elif line.startswith("# DOC_TYPE:"):
            meta["doc_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("# BLOG_SOURCE:"):
            meta["blog_source"] = line.split(":", 1)[1].strip()
        elif line.startswith("# PARSER:"):
            meta["parser"] = line.split(":", 1)[1].strip()
        elif line.startswith("# CLEANED:"):
            meta["cleaned_step"] = line.split(":", 1)[1].strip()
        elif line.startswith("# ─") or line.startswith("# --"):
            body_start = i + 1
            break

    meta["body_start"] = body_start
    return meta


def _build_txt_header(meta: dict) -> str:
    """Tái tạo header TXT từ metadata dict."""
    lines = []
    lines.append(f"# SOURCE: {meta.get('source', 'unknown')}")
    lines.append(f"# TIER: {meta.get('tier', 'unknown')}")
    lines.append(f"# LANGUAGE: {meta.get('language', 'unknown')}")
    lines.append(f"# DOC_TYPE: {meta.get('doc_type', 'unknown')}")
    if "blog_source" in meta:
        lines.append(f"# BLOG_SOURCE: {meta['blog_source']}")
    if "parser" in meta:
        lines.append(f"# PARSER: {meta['parser']}")
    lines.append("# CLEANED: step1.3")
    lines.append("# " + "─" * 60)
    return "\n".join(lines)


def clean_txt_file(txt_path: Path, overwrite: bool = True) -> dict:
    """
    Đọc 1 file .txt đã parse → clean → ghi lại (overwrite hoặc file mới).

    Args:
        txt_path : Path tới file .txt trong processed/
        overwrite: True = ghi đè file gốc, False = lưu thành *_cleaned.txt

    Returns:
        dict thống kê: {file, chars_before, chars_after, reduction_pct}
    """
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    meta = _parse_txt_header(raw)

    # Fallback doc_type theo subfolder khi metadata bị 'unknown' hoặc thiếu
    if meta.get("doc_type", "unknown") in ("unknown", ""):
        folder = txt_path.parent.name
        meta["doc_type"] = {"books": "book", "blogs": "blog", "pubmed": "pubmed"}.get(folder, "unknown")

    # Tách phần header ra khỏi body
    lines = raw.split("\n")
    body = "\n".join(lines[meta.get("body_start", 0):])
    chars_before = len(body)

    # Tạo document dict để dispatch
    doc = {
        "page_content": body,
        "metadata": {k: v for k, v in meta.items() if k != "body_start"}
    }

    # Clean
    result = clean_document(doc)
    cleaned_body = result["page_content"]
    chars_after = len(cleaned_body)

    # Ghi lại
    header = _build_txt_header(result["metadata"])
    output = header + "\n\n" + cleaned_body

    out_path = txt_path if overwrite else txt_path.with_suffix("_cleaned.txt")
    out_path.write_text(output, encoding="utf-8", errors="replace")

    reduction = (1 - chars_after / chars_before) * 100 if chars_before > 0 else 0
    return {
        "file": txt_path.name,
        "chars_before": chars_before,
        "chars_after": chars_after,
        "reduction_pct": round(reduction, 1),
    }


# ─────────────────────────────────────────────
# BATCH RUNNER — Chạy toàn bộ processed/ folder
# ─────────────────────────────────────────────

def run_batch_clean(processed_dir: str, overwrite: bool = True) -> list[dict]:
    """
    Clean toàn bộ .txt files trong processed/books/ và processed/blogs/.

    PubMed JSONL không cần xử lý — bỏ qua.

    Args:
        processed_dir: đường dẫn thư mục processed/ (chứa books/, blogs/, pubmed/)
        overwrite    : True = ghi đè file gốc

    Returns:
        List[dict] thống kê từng file đã xử lý
    """
    base = Path(processed_dir)
    stats = []

    subdirs = {
        "books": "book",
        "blogs": "blog",
    }

    for folder_name, doc_type in subdirs.items():
        folder = base / folder_name
        if not folder.exists():
            print(f"  ⚠️  Không tìm thấy thư mục: {folder}")
            continue

        txt_files = sorted(folder.glob("*.txt"))
        print(f"\n📂 [{folder_name.upper()}] Tìm thấy {len(txt_files)} file .txt")

        for f in txt_files:
            try:
                stat = clean_txt_file(f, overwrite=overwrite)
                stats.append(stat)
                print(
                    f"  ✅ {stat['file']:<55} "
                    f"{stat['chars_before']:>8,} → {stat['chars_after']:>8,} ký tự "
                    f"(-{stat['reduction_pct']}%)"
                )
            except Exception as e:
                print(f"  ❌ Lỗi {f.name}: {e}")
                stats.append({"file": f.name, "error": str(e)})

    return stats


# ─────────────────────────────────────────────
# VERIFICATION — Kiểm tra chất lượng sau clean
# ─────────────────────────────────────────────

def verify_clean_quality(processed_dir: str) -> None:
    """
    Chạy các assertion để xác minh output đạt tiêu chí Step 1.3.

    Tiêu chí (theo M4_Phase1_Action_Plan.md):
    1. NFC check: mọi file đều là NFC normalized
    2. Zero artifact check: không còn Scribd timestamp, không còn \\x00
    3. Blog nav check: không còn block nav menu (ít nhất với gym_vn)
    """
    base = Path(processed_dir)
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("  VERIFICATION — Step 1.3 Quality Check")
    print("=" * 60)

    for txt_file in sorted(base.rglob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        issues = []

        # Check 1: NFC
        if not unicodedata.is_normalized("NFC", text):
            issues.append("NOT NFC normalized")

        # Check 2: Control characters
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
            issues.append("control chars found")

        # Check 3: Scribd timestamps còn sót
        if _SCRIBD_TIMESTAMP.search(text):
            issues.append("Scribd timestamp found")

        # Check 4: Quá nhiều dòng trống liên tiếp
        if re.search(r"\n{4,}", text):
            issues.append("excessive blank lines (≥4)")

        if issues:
            print(f"  ❌ {txt_file.name}: {', '.join(issues)}")
            failed += 1
        else:
            passed += 1

    print(f"\n  ✅ Passed: {passed} files")
    print(f"  ❌ Failed: {failed} files")
    print("=" * 60)


# ─────────────────────────────────────────────
# MAIN — Chạy thử nghiệm Step 1.3
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Force UTF-8 output trên Windows (tránh lỗi charmap encode)
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    BASE_DIR = Path(__file__).parent.parent.parent   # project root
    PROCESSED_DIR = BASE_DIR / "GYM data" / "processed"

    print("=" * 60)
    print("  Step 1.3 — Text Cleaning & Normalization")
    print("=" * 60)

    # ── Chạy batch clean ──
    stats = run_batch_clean(str(PROCESSED_DIR), overwrite=True)

    # ── Tổng kết ──
    successful = [s for s in stats if "error" not in s]
    total_before = sum(s["chars_before"] for s in successful)
    total_after  = sum(s["chars_after"]  for s in successful)
    avg_reduction = (1 - total_after / total_before) * 100 if total_before else 0

    print("\n" + "=" * 60)
    print("  CLEANING REPORT")
    print("=" * 60)
    print(f"  📄 Số file đã xử lý : {len(successful)}")
    print(f"  📊 Tổng ký tự trước : {total_before:,}")
    print(f"  📊 Tổng ký tự sau   : {total_after:,}")
    print(f"  🧹 Giảm trung bình  : {avg_reduction:.1f}%")
    print("=" * 60)

    print("\n  Top 5 file giảm nhiều nhất:")
    top5 = sorted(successful, key=lambda x: x["reduction_pct"], reverse=True)[:5]
    for s in top5:
        print(f"    {s['file']}: -{s['reduction_pct']}%")

    # ── Verification ──
    verify_clean_quality(str(PROCESSED_DIR))

    # ── Sample output ──
    sample = PROCESSED_DIR / "books" / "Starting Strength — Mark Rippetoe.txt"
    if sample.exists():
        content = sample.read_text(encoding="utf-8")
        # Tìm body (sau header)
        body_start = content.find("\n\n", content.find("# ──")) + 2
        preview = content[body_start:body_start + 500]
        print("\n📖 SAMPLE — Starting Strength (500 ký tự đầu sau clean):")
        print("-" * 60)
        print(preview)
        print("-" * 60)
