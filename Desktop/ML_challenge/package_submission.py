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

    # Required output files
    out_dir = project_dir / "output"
    matching_tsv = out_dir / "matching_results.tsv"
    candidate_tsv = out_dir / "candidate_pairs.tsv"

    if not matching_tsv.exists():
        print(f"WARNING: {matching_tsv} not found! Run 'python -m entity_resolution.predict' first.")
    if not candidate_tsv.exists():
        print(f"WARNING: {candidate_tsv} not found! Run 'python -m entity_resolution.predict' first.")

    src_dir = project_dir / "entity_resolution"
    readme_file = project_dir / "README.md"
    req_file = project_dir / "requirements.txt"
    doc_file = project_dir / "Documentation_template.md"

    # Fallback to student_resource/Documentation_template.md if needed
    if not doc_file.exists():
        alt_doc = project_dir / "student_resource" / "Documentation_template.md"
        if alt_doc.exists():
            doc_file = alt_doc

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output folder
        if matching_tsv.exists():
            zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        if candidate_tsv.exists():
            zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. Code folder
        for py_file in src_dir.glob("*.py"):
            zipf.write(py_file, arcname=f"code/business_entity_resolution/src/{py_file.name}")

        if readme_file.exists():
            zipf.write(readme_file, arcname="code/business_entity_resolution/README.md")
        if req_file.exists():
            zipf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")

        # 3. Documentation
        if doc_file.exists():
            zipf.write(doc_file, arcname="Documentation_template.md")

    print(f"Successfully packaged submission into: {zip_path.resolve()}")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ER Submission ZIP")
    parser.add_argument("--team-name", type=str, default="my_team", help="Your team name for the ZIP archive")
    args = parser.parse_args()
    create_submission_package(team_name=args.team_name)
