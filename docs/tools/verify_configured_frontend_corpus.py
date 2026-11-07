#!/usr/bin/env python3
"""Run pinned frontend expectations using current SwiftPM objects, not stale globs."""
import argparse
import json
from pathlib import Path
import subprocess
from configured_frontend_corpus import ROOT, plan, summarize


def objects(binary, modules):
    result = []
    for module in modules:
        mapping = json.loads((binary / (module + ".build") / "output-file-map.json").read_text())
        result.extend(value["object"] for source, value in mapping.items() if source and "object" in value)
    if not result or len(result) != len(set(result)):
        raise ValueError("missing or duplicate compiler objects")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    configured = plan(args.reference.resolve())
    (out / "plan.json").write_text(json.dumps(configured, indent=2, allow_nan=False) + "\n")
    build = ["/usr/bin/swift", "build", "--scratch-path", ".build-validation",
             "-Xswiftc", "-warnings-as-errors", "-Xswiftc", "-strict-concurrency=complete"]
    with (out / "runner-build.log").open("w") as log:
        subprocess.run(build, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = Path(subprocess.check_output(build[:4] + ["--show-bin-path"], cwd=ROOT, text=True).strip())
        subprocess.run(["/usr/bin/swiftc", "-parse-as-library", "-warnings-as-errors", "-strict-concurrency=complete",
                        "-I", str(binary / "Modules"), str(ROOT / "docs/tools/run_configured_frontend_corpus.swift"),
                        *objects(binary, ["SAGAFrontends", "SAGAShaderIR", "SAGAValidation"]),
                        "-o", str(out / "configured-corpus")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "runner.log").open("w") as log:
        subprocess.run([str(out / "configured-corpus"), str(out / "plan.json"), str(out / "actual.json")],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = summarize(configured, json.loads((out / "actual.json").read_text()))
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(denominator=result["denominator"], counts=result["counts"], full_saga5_complete=False)))


if __name__ == "__main__":
    main()
