#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT / "paper"
MANUSCRIPT_PATH = PAPER_ROOT / "manuscript.md"
RECONCILE_ROOT = PAPER_ROOT / "reconcile"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
IMAGE_LINE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]+\)(?:\{[^}]*\})?")
BOLD_LINE_RE = re.compile(r"^\*\*(?P<text>.+?)\*\*$")
TOP_LEVEL_EXPORT_HEADINGS = {
    "ABSTRACT": "Abstract",
    "INTRODUCTION": "Introduction",
    "METHODS": "Methods",
    "RESULTS": "Results",
    "DISCUSSION": "Discussion",
    "CONCLUSIONS": "Conclusions",
    "KEYWORDS": "Keywords",
    "CONFLICT OF INTEREST": "Conflict of Interest",
    "FUNDING": "Funding",
    "AUTHOR CONTRIBUTIONS (CREDIT)": "Author Contributions (CRediT)",
    "DATA AVAILABILITY": "Data Availability",
    "ETHICAL CONSIDERATIONS": "Ethical Considerations",
    "APPENDIX": "Appendix",
    "REFERENCES": "References",
}
NON_HEADING_BOLD_PREFIXES = ("Figure ", "Table ")
SMART_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


@dataclass(frozen=True)
class SectionDiffSummary:
    changed: list[str]
    unchanged: list[str]
    export_only: list[str]
    canonical_only: list[str]


def require_tool(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"Required tool not found on PATH: {tool_name}")


def run(cmd: list[str | Path], *, cwd: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def slugify(value: str) -> str:
    lowered = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "peer-edit"


def default_output_dir(docx_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RECONCILE_ROOT / f"{slugify(docx_path.stem)}-{timestamp}"


def normalize_markdown_for_diff(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(SMART_PUNCTUATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200b", "")

    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("!["):
            image_match = IMAGE_LINE_RE.fullmatch(stripped)
            if image_match is not None:
                alt_text = image_match.group("alt").strip()
                line = f"![{alt_text}]"
        cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized + "\n"


def build_section_map(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    heading_stack: dict[int, str] = {}
    current_key = "Preamble"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        sections[current_key] = f"{body}\n" if body else ""

    def parse_heading(line: str) -> tuple[int, str] | None:
        heading_match = HEADING_RE.match(line)
        if heading_match is not None:
            return len(heading_match.group(1)), heading_match.group(2).strip()

        bold_match = BOLD_LINE_RE.fullmatch(line.strip())
        if bold_match is None:
            return None

        if line.strip().count("**") != 2:
            return None

        text_value = bold_match.group("text").strip()
        if not text_value or text_value.startswith(NON_HEADING_BOLD_PREFIXES):
            return None

        normalized_export_heading = text_value.upper()
        if normalized_export_heading in TOP_LEVEL_EXPORT_HEADINGS:
            return 1, TOP_LEVEL_EXPORT_HEADINGS[normalized_export_heading]

        if heading_stack.get(1) is not None:
            return 2, text_value
        return None

    for line in text.splitlines():
        parsed_heading = parse_heading(line)
        if parsed_heading is None:
            current_lines.append(line)
            continue

        flush()
        level, heading_text = parsed_heading
        heading_stack[level] = heading_text
        for stale_level in [key for key in heading_stack if key > level]:
            del heading_stack[stale_level]
        current_key = " > ".join(heading_stack[idx] for idx in sorted(heading_stack))
        current_lines = [line]

    flush()
    return sections


def summarize_section_changes(
    canonical_sections: dict[str, str],
    exported_sections: dict[str, str],
) -> SectionDiffSummary:
    ordered_keys: list[str] = []
    for section_key in [*canonical_sections.keys(), *exported_sections.keys()]:
        if section_key not in ordered_keys:
            ordered_keys.append(section_key)

    changed: list[str] = []
    unchanged: list[str] = []
    export_only: list[str] = []
    canonical_only: list[str] = []

    for section_key in ordered_keys:
        canonical_body = canonical_sections.get(section_key)
        exported_body = exported_sections.get(section_key)
        if canonical_body is None:
            export_only.append(section_key)
        elif exported_body is None:
            canonical_only.append(section_key)
        elif canonical_body == exported_body:
            unchanged.append(section_key)
        else:
            changed.append(section_key)

    return SectionDiffSummary(
        changed=changed,
        unchanged=unchanged,
        export_only=export_only,
        canonical_only=canonical_only,
    )


def write_summary(
    *,
    summary: SectionDiffSummary,
    docx_path: Path,
    output_dir: Path,
    raw_export_path: Path,
    exported_normalized_path: Path,
    canonical_normalized_path: Path,
    diff_path: Path,
) -> Path:
    summary_path = output_dir / "summary.md"
    lines = [
        "# Peer Edit Reconciliation Summary",
        "",
        f"- Source DOCX: `{docx_path}`",
        f"- Canonical manuscript: `{MANUSCRIPT_PATH}`",
        f"- Raw export markdown: `{raw_export_path}`",
        f"- Exported normalized markdown: `{exported_normalized_path}`",
        f"- Canonical normalized markdown: `{canonical_normalized_path}`",
        f"- Unified diff: `{diff_path}`",
        "",
        "## Changed Sections",
    ]
    if summary.changed:
        lines.extend(f"- {section}" for section in summary.changed)
    else:
        lines.append("- None")

    lines.extend(["", "## Export-Only Sections"])
    if summary.export_only:
        lines.extend(f"- {section}" for section in summary.export_only)
    else:
        lines.append("- None")

    lines.extend(["", "## Canonical-Only Sections"])
    if summary.canonical_only:
        lines.extend(f"- {section}" for section in summary.canonical_only)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Unchanged Sections",
            f"- {len(summary.unchanged)} section(s)",
            "",
            "## Next Steps",
            "1. Review the summary and unified diff.",
            "2. Apply accepted prose/table edits manually to `paper/manuscript.md`.",
            "3. Rebuild the submission outputs with `just paper-build`.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def convert_docx_to_markdown(docx_path: Path, output_dir: Path) -> Path:
    require_tool("pandoc")
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    raw_export_path = output_dir / "exported.raw.md"
    run(
        [
            "pandoc",
            docx_path,
            "--from=docx",
            "--to=markdown+pipe_tables+grid_tables+raw_html",
            "--wrap=preserve",
            f"--extract-media={media_dir}",
            "--output",
            raw_export_path,
        ],
        cwd=PROJECT_ROOT,
    )
    return raw_export_path


def reconcile_peer_edit_docx(
    *,
    docx_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")
    if not MANUSCRIPT_PATH.exists():
        raise FileNotFoundError(f"Canonical manuscript missing: {MANUSCRIPT_PATH}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite or choose a different output directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_export_path = convert_docx_to_markdown(docx_path, output_dir)
    exported_raw = raw_export_path.read_text(encoding="utf-8")
    canonical_raw = MANUSCRIPT_PATH.read_text(encoding="utf-8")

    exported_normalized = normalize_markdown_for_diff(exported_raw)
    canonical_normalized = normalize_markdown_for_diff(canonical_raw)

    exported_normalized_path = output_dir / "exported.normalized.md"
    canonical_normalized_path = output_dir / "canonical.normalized.md"
    exported_normalized_path.write_text(exported_normalized, encoding="utf-8")
    canonical_normalized_path.write_text(canonical_normalized, encoding="utf-8")

    diff_text = "".join(
        difflib.unified_diff(
            canonical_normalized.splitlines(keepends=True),
            exported_normalized.splitlines(keepends=True),
            fromfile="canonical.normalized.md",
            tofile="exported.normalized.md",
        )
    )
    diff_path = output_dir / "canonical_vs_exported.diff"
    diff_path.write_text(diff_text or "No differences after normalization.\n", encoding="utf-8")

    summary = summarize_section_changes(
        build_section_map(canonical_normalized),
        build_section_map(exported_normalized),
    )
    summary_path = write_summary(
        summary=summary,
        docx_path=docx_path,
        output_dir=output_dir,
        raw_export_path=raw_export_path,
        exported_normalized_path=exported_normalized_path,
        canonical_normalized_path=canonical_normalized_path,
        diff_path=diff_path,
    )
    return summary_path, diff_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an exported peer-edit DOCX into a normalized diff against paper/manuscript.md."
    )
    parser.add_argument(
        "--docx",
        required=True,
        type=Path,
        help="Path to the exported peer-edited DOCX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory for reconciliation artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.docx)
    summary_path, diff_path = reconcile_peer_edit_docx(
        docx_path=args.docx.resolve(),
        output_dir=output_dir.resolve(),
        overwrite=args.overwrite,
    )
    print(f"Summary: {summary_path}")
    print(f"Diff: {diff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
