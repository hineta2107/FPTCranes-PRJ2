# docx-to-markdown

Convert `.docx` files to well-structured Markdown with image extraction, attachment handling, and automatic splitting for long documents.

## When to Use

Use this skill when the user wants to convert a Word document (`.docx`) to Markdown format. Triggers: "convert docx", "docx to markdown", "convert document", "word to markdown".

## How to Use

Run the conversion script:

```bash
cd $PROJECT_ROOT
python3 .claude/skills/docx-to-markdown/convert.py "<path_to_docx>" [--output-dir <dir>] [--max-lines 200]
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `input` | (required) | Path to the `.docx` file |
| `--output-dir` | `./docs/converted/<filename>/` | Output directory for markdown, images, and references |
| `--max-lines` | `200` | Maximum lines per markdown file before splitting |

### Output Structure

```
<output-dir>/
├── images/          # Extracted images (png, jpg, etc.)
├── references/      # Non-image attachments (pdf, xlsx, etc.)
├── <name>_part01.md # First part (or <name>.md if no split needed)
├── <name>_part02.md # Second part
├── ...
└── index.md         # Table of contents linking all parts
```

### Features

- **Images**: Extracted to `images/` folder and referenced with relative paths in markdown
- **Tables**: Converted to proper markdown tables
- **Headings**: Preserves heading hierarchy (H1-H6)
- **Text formatting**: Bold, italic, underline, strikethrough
- **Lists**: Ordered and unordered lists
- **Attachments**: Non-image embedded objects saved to `references/`
- **Auto-split**: Long documents split at heading boundaries respecting `--max-lines`
- **Index**: Generates `index.md` linking all parts when split occurs

### Dependencies

- `python-docx` (pip install python-docx)
- `Pillow` (pip install Pillow) — optional, for image format detection
