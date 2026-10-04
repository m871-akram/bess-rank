"""Command line entry point: python -m bessrank.run {data,qa,smoke,features,validate,test,explore,report}."""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from bessrank import config

PACKAGES = ["pandas", "numpy", "pyarrow", "requests", "holidays", "scipy", "ortools", "xgboost",
            "scikit-learn", "torch", "matplotlib", "mlflow", "pytest"]


def git_commit():
    """Current commit hash, with '-dirty' if there are uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def update_provenance(section, info):
    """Record a stage's details in results/provenance.json with versions and the git commit."""
    path = config.PROVENANCE_JSON
    prov = json.loads(path.read_text()) if path.exists() else {}
    prov[section] = {
        **info,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2) + "\n")


def cmd_data(args):
    from bessrank import data
    meta = data.build_data(refresh=args.refresh)
    keep = ["source", "smard_download_utc_first", "smard_download_utc_last", "first_day", "last_day", "rows"]
    update_provenance("data", {k: meta[k] for k in keep})


def cmd_qa(args):
    from bessrank import data
    data.write_qa_report()


def cmd_smoke(args):
    from bessrank import databricks
    ok = databricks.smoke_test()
    sys.exit(0 if ok else 1)


def not_yet(args):
    sys.exit(f"`{args.command}` is not implemented yet (PLAN.md §11, later sessions).")


def main():
    parser = argparse.ArgumentParser(prog="python -m bessrank.run")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="rebuild data/ from the Databricks volume or SMARD")
    p.add_argument("--refresh", action="store_true", help="skip the volume and re-download from SMARD")
    p.set_defaults(func=cmd_data)
    sub.add_parser("qa", help="write results/qa_data.md").set_defaults(func=cmd_qa)
    sub.add_parser("smoke", help="Databricks smoke test").set_defaults(func=cmd_smoke)
    for name in ["features", "validate", "test", "explore", "report"]:
        sub.add_parser(name).set_defaults(func=not_yet)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
