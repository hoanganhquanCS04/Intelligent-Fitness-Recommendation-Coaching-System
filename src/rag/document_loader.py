"""
src/rag/document_loader.py
==========================
Step 1.2 — Document Parsing: PDF + HTML + PubMed XML → plain text

⚠️ LƯU Ý DÀNH CHO BÁO CÁO:
Tuy module này chứa code để parse PDF sang text, nhưng trên thực tế với lượng dữ liệu sách PDF lớn, 
toàn bộ quá trình parse PDF -> Text đã được thực hiện và tối ưu trên Google Colab. 
Các notebook xử lý được đính kèm trong dự án, và kết quả đã được lưu sẵn trong thư mục processed/.
Code parse PDF cục bộ dưới đây phục vụ minh họa pipeline và chạy test với các mẫu dữ liệu nhỏ.

Chuẩn LangChain Document schema:
    {
        "page_content": "<plain text>",
        "metadata": {
            "source": "filename.pdf",
            "tier": "tier1" | "tier2" | "tier3",
            "language": "en" | "vi",
            "doc_type": "book" | "pubmed" | "blog"
        }
    }

Thư viện cần cài:
    pip install pymupdf beautifulsoup4 langdetect
"""

import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from pathlib import Path
from langdetect import detect, LangDetectException
import json
import time


# ─────────────────────────────────────────────
# PARSER A — PDF → plain text (dùng PyMuPDF)
# ─────────────────────────────────────────────

def parse_pdf(pdf_path: str, min_line_length: int = 20) -> str:
    """
    Parse PDF → plain text. Lọc bỏ header/footer ngắn theo heuristic.

    Args:
        pdf_path      : đường dẫn tới file PDF
        min_line_length: ngưỡng tối thiểu ký tự/dòng (lọc số trang, header ngắn)

    Returns:
        Chuỗi văn bản đã clean sơ bộ, page break = double newline
    """
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # "text" mode: tốt nhất cho prose

        # Heuristic: bỏ dòng quá ngắn (thường là số trang hoặc header)
        lines = [
            ln.strip()
            for ln in text.split("\n")
            if len(ln.strip()) >= min_line_length
        ]
        page_content = "\n".join(lines)
        if page_content.strip():
            pages_text.append(page_content)

    doc.close()
    return "\n\n".join(pages_text)  # double newline = page break


def load_pdf_document(pdf_path: str, tier: str = "tier1") -> dict:
    """
    Load 1 PDF file → LangChain Document dict.

    Args:
        pdf_path: đường dẫn file PDF
        tier    : "tier1" (sách học thuật)

    Returns:
        {"page_content": str, "metadata": dict}
    """
    path = Path(pdf_path)
    print(f"  [PDF] Parsing: {path.name} ...", end=" ", flush=True)
    t0 = time.time()

    text = parse_pdf(pdf_path)
    lang = detect_language(text[:2000])  # dùng 2000 ký tự đầu để detect

    elapsed = time.time() - t0
    chars = len(text)
    print(f"✅ {chars:,} ký tự | ngôn ngữ={lang} | {elapsed:.1f}s")

    return {
        "page_content": text,
        "metadata": {
            "source": path.name,
            "tier": tier,
            "language": lang,
            "doc_type": "book",
        }
    }


# ─────────────────────────────────────────────
# PARSER B — HTML → plain text (BeautifulSoup)
# ─────────────────────────────────────────────

# Các thẻ HTML không chứa nội dung cần parse
_NOISE_TAGS = ["script", "style", "nav", "footer", "aside", "header",
               "noscript", "iframe", "form", "button", "svg", "meta"]


def parse_html(html_path: str) -> str:
    """
    Parse HTML file → plain text sạch.

    Quy trình:
    1. Xóa các thẻ noise (nav, footer, script, style...)
    2. get_text() với separator newline để bảo toàn paragraph
    3. Trả về chuỗi text thô (chưa clean — sẽ clean ở Step 1.3)
    """
    with open(str(html_path), "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Xóa các thẻ noise
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def load_html_document(html_path: str, tier: str = "tier3", blog_source: str = "") -> dict:
    """
    Load 1 HTML file → LangChain Document dict.

    Args:
        html_path   : đường dẫn file HTML (raw blog article)
        tier        : "tier3" (blog/website)
        blog_source : tên subfolder nguồn blog (ví dụ: "gym_vn_articles") — dùng để tránh tên file trùng

    Returns:
        {"page_content": str, "metadata": dict}
    """
    path = Path(html_path)
    # Lấy tên subfolder nếu không truyền vào
    if not blog_source:
        blog_source = path.parent.name
    print(f"  [HTML] Parsing: {blog_source}/{path.name} ...", end=" ", flush=True)

    text = parse_html(html_path)
    lang = detect_language(text[:2000])

    chars = len(text)
    print(f"✅ {chars:,} ký tự | ngôn ngữ={lang}")

    return {
        "page_content": text,
        "metadata": {
            "source": path.name,
            "tier": tier,
            "language": lang,
            "doc_type": "blog",
            "blog_source": blog_source,
        }
    }


# ─────────────────────────────────────────────
# PARSER C — PubMed XML → list of dicts
# ─────────────────────────────────────────────

def parse_pubmed_xml(xml_path: str) -> list[dict]:
    """
    Parse PubMed XML → list of Document dicts.

    Hỗ trợ 2 schema:
    - Schema chuẩn PubMed E-utilities: PubmedArticleSet > PubmedArticle > ...
    - Schema tùy chỉnh (custom crawler): PubmedData > Article > {PMID, Title, Abstract, Query}

    Returns:
        List of {"page_content": str, "metadata": dict}
    """
    print(f"  [XML] Parsing PubMed: {Path(xml_path).name} ...", end=" ", flush=True)

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    results = []
    skipped = 0

    # ── Phát hiện schema ──
    root_tag = root.tag  # "PubmedArticleSet" hoặc "PubmedData"

    # Schema chuẩn PubMed E-utilities
    standard_articles = root.findall(".//PubmedArticle")
    # Schema tùy chỉnh (custom crawler output)
    custom_articles = root.findall(".//Article") if not standard_articles else []

    if standard_articles:
        # ── Parse chuẩn PubMed ──
        for article in standard_articles:
            pmid     = article.findtext(".//PMID") or ""
            title    = article.findtext(".//ArticleTitle") or ""
            abstract = article.findtext(".//AbstractText") or ""
            keywords = [kw.text for kw in article.findall(".//Keyword") if kw.text]
            journal  = article.findtext(".//Title") or ""
            year     = article.findtext(".//PubDate/Year") or ""

            if not abstract.strip():
                skipped += 1
                continue

            content_parts = [f"Title: {title}", f"\nAbstract: {abstract}"]
            if keywords:
                content_parts.append(f"\nKeywords: {', '.join(keywords)}")

            results.append({
                "page_content": "\n".join(content_parts),
                "metadata": {
                    "source": f"pubmed_{pmid}",
                    "tier": "tier2",
                    "language": "en",
                    "doc_type": "pubmed",
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                }
            })

    elif custom_articles:
        # ── Parse schema tùy chỉnh: PubmedData > Article > {PMID, Title, Abstract, Query} ──
        for article in custom_articles:
            pmid     = article.findtext("PMID") or ""
            title    = article.findtext("Title") or ""
            abstract = article.findtext("Abstract") or ""
            query    = article.findtext("Query") or ""  # query dùng để crawl

            if not abstract.strip():
                skipped += 1
                continue

            content_parts = [f"Title: {title}", f"\nAbstract: {abstract}"]
            if query:
                content_parts.append(f"\nTopic: {query}")

            results.append({
                "page_content": "\n".join(content_parts),
                "metadata": {
                    "source": f"pubmed_{pmid}",
                    "tier": "tier2",
                    "language": "en",
                    "doc_type": "pubmed",
                    "pmid": pmid,
                    "title": title,
                    "query": query,
                }
            })
    else:
        print(f"\n⚠️  Không nhận dạng được schema XML. Root tag: {root_tag}")

    print(f"✅ {len(results)} abstracts | bỏ qua {skipped} (không có abstract)")
    return results


def save_pubmed_to_jsonl(documents: list[dict], output_path: str) -> None:
    """
    Lưu PubMed documents ra file JSONL (JSON Lines).
    Mỗi dòng = 1 JSON object = 1 abstract.

    Args:
        documents  : list từ parse_pubmed_xml()
        output_path: đường dẫn file .jsonl output
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  → Đã lưu {len(documents)} abstracts → {output_path}")


# ─────────────────────────────────────────────
# UTILITY — Language Detection
# ─────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect ngôn ngữ của text. Trả về "vi", "en", hoặc "unknown".

    Dùng langdetect (port của Google language-detection library).
    Fallback: "en" nếu không detect được.
    """
    try:
        lang = detect(text)
        # Map về vi/en cho đơn giản; langdetect trả về ISO 639-1 codes
        if lang in ("vi",):
            return "vi"
        return "en"
    except LangDetectException:
        return "en"


# ─────────────────────────────────────────────
# BATCH LOADERS — Chạy toàn bộ corpus
# ─────────────────────────────────────────────

def load_all_books(books_dir: str) -> list[dict]:
    """
    Load toàn bộ PDF sách từ thư mục.

    [GHI CHÚ]: Theo quy trình thực tế của nhóm, các sách lớn đã được xử lý bằng Google Colab 
    và lưu trực tiếp thành file txt. Hàm này dùng để load các file sách bổ sung cục bộ.

    Args:
        books_dir: đường dẫn thư mục chứa PDF (ví dụ: "GYM data/book")

    Returns:
        List of Document dicts
    """
    import time
    
    print("\n" + "!" * 75)
    print(" CẢNH BÁO TỪ HỆ THỐNG: XỬ LÝ SÁCH PDF (TIER 1) ".center(75, "!"))
    print("!" * 75)
    print(" Bạn đang khởi chạy block lệnh chuyển đổi PDF sang Text ở local.")
    print(" Lưu ý rằng do khối lượng PDF lớn và tốn rất nhiều tài nguyên RAM/CPU,")
    print(" toàn bộ quá trình TỐI ƯU bóc tách sách đã được nhóm cấu hình để")
    print(" CHẠY TỰ ĐỘNG TRÊN GOOGLE COLAB.")
    print("\n 👉 VUI LÒNG SỬ DỤNG NOTEBOOK SAU TRỂ THỰC THI THAY THẾ:")
    print("    [ notebooks/rag_parse_books_optimized.ipynb ]\n")
    print(" Kết quả của notebook trên đã nằm sẵn ở thư mục 'processed/'.")
    print("!" * 75)
    print("\n...Hệ thống sẽ tạm dừng 5 giây để bạn đọc thông báo.")
    print("...Sau đó sẽ tiếp tục chạy demo quét sơ các file PDF nhỏ (nếu có).\n")
    time.sleep(5)

    books_path = Path(books_dir)
    pdf_files = sorted(books_path.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  Không tìm thấy file PDF trong: {books_dir}")
        return []

    print(f"\n📚 Load sách PDF từ: {books_dir}")
    print(f"   Tìm thấy {len(pdf_files)} file PDF\n")

    documents = []
    for pdf_file in pdf_files:
        try:
            doc = load_pdf_document(str(pdf_file), tier="tier1")
            documents.append(doc)
        except Exception as e:
            print(f"❌ Lỗi khi parse {pdf_file.name}: {e}")

    print(f"\n✅ Đã parse xong {len(documents)}/{len(pdf_files)} sách PDF")
    return documents


def load_all_blogs(blogs_dir: str) -> list[dict]:
    """
    Load toàn bộ HTML blog từ tất cả sub-thư mục trong blogs_dir.

    Cấu trúc mong đợi:
        blogs_dir/
            gym_vn_articles/article_001.html, ...
            rp_strength_articles/article_001.html, ...
            sbs_articles/article_001.html, ...

    Returns:
        List of Document dicts
    """
    blogs_path = Path(blogs_dir)
    html_files = sorted(blogs_path.rglob("*.html"))

    if not html_files:
        print(f"⚠️  Không tìm thấy file HTML trong: {blogs_dir}")
        return []

    print(f"\n📝 Load blog HTML từ: {blogs_dir}")
    print(f"   Tìm thấy {len(html_files)} file HTML\n")

    documents = []
    for html_file in html_files:
        try:
            doc = load_html_document(str(html_file), tier="tier3")
            documents.append(doc)
        except Exception as e:
            print(f"❌ Lỗi khi parse {html_file.name}: {e}")

    print(f"\n✅ Đã parse xong {len(documents)}/{len(html_files)} bài blog")
    return documents


def load_pubmed(xml_path: str, output_jsonl: str) -> list[dict]:
    """
    Parse PubMed XML và lưu ra JSONL.

    Args:
        xml_path    : đường dẫn file XML PubMed
        output_jsonl: đường dẫn output file JSONL

    Returns:
        List of Document dicts
    """
    print(f"\n🔬 Load PubMed abstracts từ: {xml_path}")
    documents = parse_pubmed_xml(xml_path)
    save_pubmed_to_jsonl(documents, output_jsonl)
    return documents


# ─────────────────────────────────────────────
# SAVE UTILITIES
# ─────────────────────────────────────────────

def save_text_document(document: dict, output_dir: str) -> str:
    """
    Lưu 1 Document dict ra file .txt.

    Tên file output:
    - Blog: [blog_source]_[filename].txt  (tránh trùng tên giữa các nguồn)
    - Khác: [filename].txt

    Trả về đường dẫn file đã lưu.
    """
    metadata     = document["metadata"]
    stem         = Path(metadata["source"]).stem
    blog_source  = metadata.get("blog_source", "")

    # Thêm prefix tên nguồn blog để tránh ghi đè
    filename     = f"{blog_source}_{stem}.txt" if blog_source else f"{stem}.txt"
    output_path  = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write(f"# SOURCE: {metadata['source']}\n")
        f.write(f"# TIER: {metadata['tier']}\n")
        f.write(f"# LANGUAGE: {metadata['language']}\n")
        f.write(f"# DOC_TYPE: {metadata['doc_type']}\n")
        if blog_source:
            f.write(f"# BLOG_SOURCE: {blog_source}\n")
        f.write("# " + "─" * 60 + "\n\n")
        f.write(document["page_content"])

    return str(output_path)


def save_all_text_documents(documents: list[dict], output_dir: str) -> list[str]:
    """
    Lưu toàn bộ documents ra thư mục output.

    Returns:
        List đường dẫn các file đã lưu
    """
    saved_paths = []
    for doc in documents:
        try:
            path = save_text_document(doc, output_dir)
            saved_paths.append(path)
        except Exception as e:
            print(f"❌ Lỗi khi lưu {doc['metadata']['source']}: {e}")

    print(f"  → Đã lưu {len(saved_paths)} file .txt → {output_dir}")
    return saved_paths


# ─────────────────────────────────────────────
# MAIN — Chạy thử nghiệm
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Đường dẫn (thay đổi nếu cần) ──
    BASE_DIR    = Path(__file__).parent.parent.parent  # Project root
    RAW_DIR     = BASE_DIR / "GYM data"
    BOOKS_DIR   = RAW_DIR / "book"
    BLOGS_DIR   = RAW_DIR / "blogs"
    PUBMED_XML  = RAW_DIR / "pubmed" / "pubmed_abstracts_large.xml"

    PROCESSED_DIR      = RAW_DIR / "processed"
    BOOKS_OUT          = PROCESSED_DIR / "books"
    BLOGS_OUT          = PROCESSED_DIR / "blogs"
    PUBMED_JSONL_OUT   = PROCESSED_DIR / "pubmed" / "all_abstracts.jsonl"

    print("=" * 60)
    print("  Step 1.2 — Document Parsing")
    print("=" * 60)

    # ── 1. Parse sách PDF ──
    # NOTE: Dữ liệu PDF lớn đã được xử lý triệt để trên Google Colab. 
    # Vòng lặp dưới đây minh họa quá trình với các file PDF mới/nhỏ.
    print("[NOTE] Bước xử lý PDF thực tế (tier1) chủ yếu được chạy trên Google Colab.")
    print("       Đang quét thư mục cục bộ cho các file test/sample...\n")
    all_books = load_all_books(str(BOOKS_DIR))
    if all_books:
        save_all_text_documents(all_books, str(BOOKS_OUT))
        # Quality check: in 300 ký tự đầu của sách đầu tiên
        print(f"\n📖 SAMPLE — {all_books[0]['metadata']['source']}")
        print(all_books[0]["page_content"][:400])
        print("...")

    # ── 2. Parse blog HTML ──
    all_blogs = load_all_blogs(str(BLOGS_DIR))
    if all_blogs:
        save_all_text_documents(all_blogs, str(BLOGS_OUT))

    # ── 3. Parse PubMed XML ──
    pubmed_docs = []
    if PUBMED_XML.exists():
        pubmed_docs = load_pubmed(str(PUBMED_XML), str(PUBMED_JSONL_OUT))
    else:
        print(f"\n⚠️  Không tìm thấy file PubMed XML: {PUBMED_XML}")

    # ── 4. Báo cáo tổng hợp ──
    total_docs  = len(all_books) + len(all_blogs) + len(pubmed_docs)
    total_chars = (
        sum(len(d["page_content"]) for d in all_books)
        + sum(len(d["page_content"]) for d in all_blogs)
        + sum(len(d["page_content"]) for d in pubmed_docs)
    )

    print("\n" + "=" * 60)
    print("  PARSING REPORT")
    print("=" * 60)
    print(f"  📚 Sách PDF (Tier 1)  : {len(all_books):>4} file")
    print(f"  🔬 PubMed (Tier 2)    : {len(pubmed_docs):>4} abstracts")
    print(f"  📝 Blog HTML (Tier 3) : {len(all_blogs):>4} bài")
    print(f"  ─────────────────────────────")
    print(f"  📦 TỔNG               : {total_docs:>4} documents")
    print(f"  💬 Tổng ký tự         : {total_chars:,}")
    print(f"  📁 Output dir         : {PROCESSED_DIR}")
    print("=" * 60)

    # ── 5. Quality check ──
    print("\n🔍 QUALITY CHECK:")
    if all_blogs:
        sample_blog = all_blogs[0]
        lang = sample_blog["metadata"]["language"]
        src  = sample_blog["metadata"]["source"]
        chars = len(sample_blog["page_content"])
        print(f"  Blog mẫu: {src} | ngôn ngữ={lang} | {chars:,} ký tự")
        # Kiểm tra không có thẻ HTML còn sót
        has_html = "<html>" in sample_blog["page_content"].lower()
        print(f"  HTML tag còn sót: {'❌ CÓ' if has_html else '✅ KHÔNG'}")
