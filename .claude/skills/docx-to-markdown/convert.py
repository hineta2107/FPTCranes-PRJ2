#!/usr/bin/env python3
"""Convert .docx files to Markdown with images, attachments, and auto-splitting."""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text.strip()).strip("_").lower()


def extract_images(docx_path: str, images_dir: Path) -> dict[str, str]:
    """Extract all images from the docx and return {rId: relative_path} mapping."""
    images_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    with zipfile.ZipFile(docx_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("word/media/"):
                fname = os.path.basename(name)
                target = images_dir / fname
                target.write_bytes(zf.read(name))
                mapping[fname] = f"images/{fname}"
    return mapping


def extract_attachments(docx_path: str, refs_dir: Path) -> dict[str, str]:
    """Extract non-image embedded objects to references/."""
    refs_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    with zipfile.ZipFile(docx_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("word/embeddings/"):
                fname = os.path.basename(name)
                target = refs_dir / fname
                target.write_bytes(zf.read(name))
                mapping[fname] = f"references/{fname}"
    return mapping


# ── Image resolution from paragraph XML ──────────────────────────────────────

def _get_image_filenames_from_paragraph(paragraph: Paragraph, doc: Document) -> list[str]:
    """Return list of image filenames referenced in this paragraph's XML."""
    filenames: list[str] = []
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    }
    # inline images
    for blip in paragraph._element.findall(".//a:blip", ns):
        r_embed = blip.get(qn("r:embed"))
        if r_embed:
            try:
                rel = doc.part.rels[r_embed]
                filenames.append(os.path.basename(rel.target_ref))
            except KeyError:
                pass
    # also check drawing > blipFill
    for blip in paragraph._element.findall(".//pic:blipFill/a:blip", ns):
        r_embed = blip.get(qn("r:embed"))
        if r_embed:
            try:
                rel = doc.part.rels[r_embed]
                fname = os.path.basename(rel.target_ref)
                if fname not in filenames:
                    filenames.append(fname)
            except KeyError:
                pass
    return filenames


# ── Paragraph to Markdown ────────────────────────────────────────────────────

HEADING_MAP = {
    "Title": 1,
    "Subtitle": 2,
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
    "Heading 6": 6,
}


def _run_to_md(run) -> str:
    """Convert a single run to markdown text with inline formatting."""
    text = run.text or ""
    if not text:
        return ""
    if run.bold:
        text = f"**{text}**"
    if run.italic:
        text = f"*{text}*"
    if run.underline:
        text = f"<u>{text}</u>"
    if run.font and run.font.strike:
        text = f"~~{text}~~"
    return text


def _paragraph_to_md(
    para: Paragraph,
    doc: Document,
    image_map: dict[str, str],
) -> str:
    """Convert a single paragraph to markdown."""
    style_name = para.style.name if para.style else "Normal"

    # Check for images in this paragraph
    img_lines: list[str] = []
    for fname in _get_image_filenames_from_paragraph(para, doc):
        rel_path = image_map.get(fname, f"images/{fname}")
        img_lines.append(f"![{fname}]({rel_path})")

    # Build text from runs
    text = "".join(_run_to_md(r) for r in para.runs).strip()

    # Heading
    heading_level = HEADING_MAP.get(style_name)
    if heading_level and text:
        line = f"{'#' * heading_level} {text}"
    elif style_name.startswith("List"):
        # Detect ordered vs unordered from numbering
        numPr = para._element.find(qn("w:pPr/w:numPr"))
        if numPr is not None:
            ilvl_el = numPr.find(qn("w:ilvl"))
            indent = int(ilvl_el.get(qn("w:val"), "0")) if ilvl_el is not None else 0
            prefix = "  " * indent
            # Check if ordered
            numId_el = numPr.find(qn("w:numId"))
            numId = numId_el.get(qn("w:val"), "0") if numId_el is not None else "0"
            if int(numId) % 2 == 0:
                line = f"{prefix}1. {text}"
            else:
                line = f"{prefix}- {text}"
        else:
            line = f"- {text}"
    else:
        line = text

    # Append images after text
    if img_lines:
        parts = [line] if line else []
        parts.extend(img_lines)
        return "\n".join(parts)

    return line


def _table_to_md(table: Table) -> str:
    """Convert a docx Table to a markdown table."""
    rows = table.rows
    if not rows:
        return ""

    md_rows: list[str] = []
    for i, row in enumerate(rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

    return "\n".join(md_rows)


# ── Main conversion ─────────────────────────────────────────────────────────

def _iter_block_items(doc: Document):
    """Yield paragraphs and tables in document order."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def convert_docx_to_md(
    docx_path: str,
    output_dir: Path,
    max_lines: int = 200,
) -> list[Path]:
    """Convert a docx to markdown file(s). Returns list of created files."""
    doc = Document(docx_path)

    images_dir = output_dir / "images"
    refs_dir = output_dir / "references"

    image_map = extract_images(docx_path, images_dir)
    attachment_map = extract_attachments(docx_path, refs_dir)

    # Clean up empty dirs
    if refs_dir.exists() and not any(refs_dir.iterdir()):
        refs_dir.rmdir()
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()

    # Build full markdown lines
    all_lines: list[str] = []
    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            md = _paragraph_to_md(block, doc, image_map)
            all_lines.append(md)
        elif isinstance(block, Table):
            all_lines.append("")
            all_lines.append(_table_to_md(block))
            all_lines.append("")

    # Remove excessive blank lines
    cleaned: list[str] = []
    prev_blank = False
    for line in all_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    # Add references section if attachments exist
    if attachment_map:
        cleaned.append("")
        cleaned.append("## References / Attachments")
        cleaned.append("")
        for fname, rel_path in sorted(attachment_map.items()):
            cleaned.append(f"- [{fname}]({rel_path})")

    # Split into parts if needed
    base_name = Path(docx_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(cleaned) <= max_lines:
        out_file = output_dir / f"{slugify(base_name)}.md"
        out_file.write_text("\n".join(cleaned), encoding="utf-8")
        print(f"✅ Created: {out_file} ({len(cleaned)} lines)")
        return [out_file]

    # Split at heading boundaries
    parts: list[list[str]] = []
    current_part: list[str] = []

    for line in cleaned:
        is_heading = line.startswith("#")
        if is_heading and len(current_part) >= max_lines:
            parts.append(current_part)
            current_part = []
        current_part.append(line)

    if current_part:
        parts.append(current_part)

    created_files: list[Path] = []
    index_entries: list[str] = [f"# {base_name}", "", "## Parts", ""]

    for i, part in enumerate(parts, 1):
        part_file = output_dir / f"{slugify(base_name)}_part{i:02d}.md"
        # Add navigation header
        nav = f"*Part {i} of {len(parts)}*\n"
        content = nav + "\n".join(part)
        part_file.write_text(content, encoding="utf-8")
        created_files.append(part_file)

        # Extract first heading for index
        first_heading = next((l for l in part if l.startswith("#")), f"Part {i}")
        heading_text = first_heading.lstrip("#").strip()
        index_entries.append(
            f"{i}. [{heading_text}]({part_file.name}) ({len(part)} lines)"
        )
        print(f"✅ Created: {part_file} ({len(part)} lines)")

    # Write index
    index_file = output_dir / "index.md"
    index_file.write_text("\n".join(index_entries) + "\n", encoding="utf-8")
    created_files.insert(0, index_file)
    print(f"✅ Created: {index_file} (index)")

    return created_files


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert .docx to Markdown")
    parser.add_argument("input", help="Path to .docx file")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="Max lines per markdown file (default: 200)",
    )
    args = parser.parse_args()

    docx_path = args.input
    if not os.path.isfile(docx_path):
        print(f"❌ File not found: {docx_path}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stem = Path(docx_path).stem
        output_dir = Path("docs/converted") / slugify(stem)

    files = convert_docx_to_md(docx_path, output_dir, args.max_lines)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📄 Total files created: {len(files)}")


if __name__ == "__main__":
    main()
