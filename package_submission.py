"""
Automated packaging script for the Amazon ML Challenge submission.
Creates the exact required submission ZIP structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches
│   └── candidate_pairs.tsv         # blocking candidates
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code files
│       ├── README.md               # run instructions
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # methodology write-up
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def create_submission_package(team_name: str = "team", output_zip: str = None) -> Path:
    project_dir = Path(__file__).resolve().parent
    zip_filename = output_zip or f"{team_name}_submission.zip"
    zip_path = project_dir / zip_filename

    print(f"Creating submission package: {zip_path}...")

    # Required output files. Both are mandatory per the challenge criteria
    # (candidate_pairs.tsv is the exact candidate set matching_results.tsv was
    # produced from) so a missing file must hard-fail the packaging run
    # instead of silently shipping an incomplete zip.
    out_dir = project_dir / "output"
    matching_tsv = out_dir / "matching_results.tsv"
    candidate_tsv = out_dir / "candidate_pairs.tsv"

    missing = [str(p) for p in (matching_tsv, candidate_tsv) if not p.exists()]
    if missing:
        raise SystemExit(
            "ERROR: required output file(s) missing, refusing to package an incomplete submission:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nRun the inference pipeline (entity_resolution.predict) and make_candidate_pairs.py first."
        )

    src_dir = project_dir / "entity_resolution"
    extra_src_files = [project_dir / "make_candidate_pairs.py"]
    readme_file = project_dir / "README.md"
    req_file = project_dir / "requirements.txt"
    doc_file = project_dir / "Documentation_template.md"

    if not src_dir.exists():
        raise SystemExit(f"ERROR: source package not found: {src_dir}")
    if not readme_file.exists():
        raise SystemExit(f"ERROR: README.md not found: {readme_file}")
    if not req_file.exists():
        raise SystemExit(f"ERROR: requirements.txt not found: {req_file}")

    # Fallback to student_resource/Documentation_template.md if needed
    if not doc_file.exists():
        alt_doc = project_dir / "student_resource" / "Documentation_template.md"
        if alt_doc.exists():
            doc_file = alt_doc
    if not doc_file.exists():
        raise SystemExit(f"ERROR: Documentation_template.md not found: {doc_file}")

    code_root = "code/business_entity_resolution"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output folder
        zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. Code folder: all source under entity_resolution/, recursively,
        # excluding __pycache__ / compiled artifacts, plus the root-level
        # blocking-only reconstruction script used to (re)build candidate_pairs.tsv.
        for py_file in sorted(src_dir.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel = py_file.relative_to(src_dir)
            zipf.write(py_file, arcname=f"{code_root}/src/entity_resolution/{rel.as_posix()}")

        for extra_file in extra_src_files:
            if extra_file.exists():
                zipf.write(extra_file, arcname=f"{code_root}/src/{extra_file.name}")

        zipf.write(readme_file, arcname=f"{code_root}/README.md")
        zipf.write(req_file, arcname=f"{code_root}/requirements.txt")

        # 3. Documentation
        zipf.write(doc_file, arcname="Documentation_template.md")

    print(f"Successfully packaged submission into: {zip_path.resolve()}")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ER Submission ZIP")
    parser.add_argument("--team-name", type=str, default="my_team", help="Your team name for the ZIP archive")
    args = parser.parse_args()
    create_submission_package(team_name=args.team_name)
