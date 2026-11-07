#!/usr/bin/env python3
"""Pinned frontend snapshot expectations, not backend/diagnostic/CTS conformance."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tomllib

PIN = "a55352911bb26a1b33bc8d5fc03bdebb0a967f5a"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPS = (1 << 12) | (1 << 15)
TOP_KEYS = {"targets", "capabilities", "wgsl-in", "spv-in", "spv", "msl", "msl_pipeline",
            "hlsl", "glsl", "wgsl", "bounds_check_policies", "pipeline_constants", "glsl_multiview",
            "glsl_exclude_list", "spv_flow_dump_prefix", "fragment_module", "mesh_output_validation", "task_limits"}
TARGETS = {"IR", "ANALYSIS", "SPIRV", "METAL", "GLSL", "DOT", "HLSL", "WGSL", "NO_VALIDATION"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def flags(text, names):
    if not isinstance(text, str):
        raise ValueError("bitflags must be a string")
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if any(p not in names for p in parts):
        raise ValueError("unknown bitflag: " + text)
    return parts


def json_metadata(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"toml_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {k:json_metadata(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_metadata(v) for v in value]
    return value


def configuration(data, fmt, cap_names):
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise ValueError("unmapped configuration fields: " + str(sorted(unknown)))
    default_targets = "HLSL | SPIRV | GLSL | METAL | WGSL" if fmt == "wgsl" else "WGSL"
    targets = flags(data.get("targets", default_targets), TARGETS)
    cap = DEFAULT_CAPS
    if "capabilities" in data:
        cap = 0
        for name in flags(data["capabilities"], cap_names):
            cap |= 1 << cap_names[name]
    wgsl = data.get("wgsl-in", {})
    spv = data.get("spv-in", {})
    if set(wgsl) - {"parse_doc_comments"} or set(spv) - {"adjust_coordinate_space"}:
        raise ValueError("unmapped frontend options")
    doc = wgsl.get("parse_doc_comments", False)
    adjust = spv.get("adjust_coordinate_space", False)
    if type(doc) is not bool or type(adjust) is not bool:
        raise ValueError("frontend option must be bool")
    return dict(capabilities=cap, parse_doc_comments=doc, adjust_coordinate_space=adjust,
                strict_capabilities=True, validation="NO_VALIDATION" not in targets, targets=targets,
                backend_configuration=json_metadata(data),
                boundary="frontend parse + pre-compaction validation only; no output or post-compaction parity")


def pinned_files(reference):
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(reference), *args])
    if git("rev-parse", "HEAD").decode().strip() != PIN or git("status", "--porcelain"):
        raise ValueError("reference pin/worktree drift")
    paths = git("ls-tree", "-r", "--name-only", PIN).decode().splitlines()
    selected = [p for p in paths if p.startswith("naga/tests/in/") or p in {
        "naga-test/src/lib.rs", "naga/tests/naga/snapshots.rs", "naga/tests/naga/wgsl_errors.rs",
        "naga/src/valid/mod.rs", "naga/tests/naga/deep-expression.wgsl", "naga/tests/naga/deep-template.wgsl"}]
    blobs = {}
    for name in selected:
        # Compare with immutable Git blob, never bless current bytes by hashing them alone.
        data = git("show", PIN + ":" + name)
        path = reference / name
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("reference bytes drift: " + name)
        blobs[name] = data
    return blobs


def plan(reference):
    blobs = pinned_files(reference)
    names = dict((name, int(bit)) for name, bit in re.findall(
        r"const ([A-Z][A-Z0-9_]*) = 1 << (\d+);", re.search(r"pub struct Capabilities: u64 \{(.*?)^    \}", blobs["naga/src/valid/mod.rs"].decode(), re.S | re.M).group(1)))
    if len(names) != 46 or names.get("SHADER_FLOAT16") != 26:
        raise ValueError("pinned capability map changed")
    rows = []
    for source in sorted(blobs):
        suffix = Path(source).suffix
        fmt = {".wgsl": "wgsl", ".vert": "glsl", ".frag": "glsl", ".comp": "glsl", ".spvasm": "spirv-assembly"}.get(suffix)
        if not fmt or "/tests/in/" not in source:
            continue
        config_source = str(Path(source).with_suffix(".toml"))
        raw = blobs.get(config_source)
        data = tomllib.loads(raw.decode()) if raw else {}
        config = configuration(data, fmt, names)
        row = dict(source=source, path=str(reference/source), sha256=sha(blobs[source]), format=fmt,
                   stage={".vert":"vertex", ".frag":"fragment", ".comp":"compute"}.get(suffix,"compute"),
                   expected_parse="accepted", expected_semantic="accepted" if config["validation"] else "not-run",
                   expectation_source="naga/tests/naga/snapshots.rs:546-643",
                   config_source=config_source if raw else None, config_sha256=sha(raw) if raw else None, **config)
        if fmt == "spirv-assembly":
            row["unavailable_reason"] = "SPIR-V assembler/binary provenance not yet mapped; not a pass"
        rows.append(row)
    for stem in ["deep-expression", "deep-template"]:
        source = "naga/tests/naga/" + stem + ".wgsl"
        rows.append(dict(source=source,path=str(reference/source),sha256=sha(blobs[source]),format="wgsl",stage="compute",
                         expected_parse="rejected",expected_semantic="not-run",expectation_source="naga/tests/naga/wgsl_errors.rs",
                         **configuration({},"wgsl",names)))
    return dict(schema=1,reference_revision=PIN,scope="frontend snapshot acceptance only, not full parity",
                oracle_hashes={p:sha(blobs[p]) for p in ["naga-test/src/lib.rs","naga/tests/naga/snapshots.rs","naga/tests/naga/wgsl_errors.rs","naga/src/valid/mod.rs"]},
                copied_shader_bytes=0, rows=rows)


def summarize(plan, actual):
    rows = plan["rows"]
    ids = [r["source"] for r in rows]
    observed = [r["source"] for r in actual]
    if len(set(ids)) != len(ids) or len(set(observed)) != len(observed) or set(ids) != set(observed):
        raise ValueError("result membership/duplicates differ from pinned denominator")
    by_id = {r["source"]:r for r in actual}
    results = []
    for row in rows:
        result = by_id[row["source"]]
        if result.get("sha256") != row["sha256"]:
            raise ValueError("result source hash mismatch")
        for field in ["capabilities", "parse_doc_comments", "strict_capabilities", "adjust_coordinate_space",
                      "stage", "format", "config_source", "config_sha256", "validation"]:
            if field in row and (field not in result or type(result[field]) is not type(row[field]) or result[field] != row[field]):
                raise ValueError("result configuration mismatch: " + field)
        if result.get("unavailable_reason"):
            outcome = "unsupported"
        elif result.get("parse") != row["expected_parse"]:
            outcome = "fail"
        elif row["expected_semantic"] == "accepted" and result.get("semantic") != "accepted":
            outcome = "fail"
        elif row["expected_parse"] == "rejected":
            # Rejection alone does not verify diagnostic text/range or why it was rejected.
            outcome = "rejection-only"
        elif not row["validation"]:
            outcome = "parse-only"
        else:
            outcome = "pass"
        results.append(dict(row,actual=result,outcome=outcome))
    return dict(schema=1,reference_revision=PIN,denominator=len(rows),counts=dict(collections.Counter(r["outcome"] for r in results)),
                scope=plan["scope"],full_saga5_complete=False,rows=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--actual",type=Path)
    args = parser.parse_args()
    p = plan(args.reference.resolve())
    result = summarize(p,json.loads(args.actual.read_text())) if args.actual else p
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result.get("counts",dict(inputs=len(result["rows"]))),sort_keys=True))


if __name__ == "__main__":
    main()
