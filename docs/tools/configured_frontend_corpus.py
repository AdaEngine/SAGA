#!/usr/bin/env python3
"""Pinned frontend snapshot expectations, not backend/diagnostic/CTS conformance."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import tomllib
try:
    import resource
except ImportError:
    resource = None

PIN = "a55352911bb26a1b33bc8d5fc03bdebb0a967f5a"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPS = (1 << 12) | (1 << 15)
TOP_KEYS = {"targets", "capabilities", "wgsl-in", "spv-in", "spv", "msl", "msl_pipeline",
            "hlsl", "glsl", "wgsl", "bounds_check_policies", "pipeline_constants", "glsl_multiview",
            "glsl_exclude_list", "spv_flow_dump_prefix", "fragment_module", "mesh_output_validation", "task_limits"}
TARGETS = {"IR", "ANALYSIS", "SPIRV", "METAL", "GLSL", "DOT", "HLSL", "WGSL", "NO_VALIDATION"}
MAX_ASSEMBLY_BYTES = 4_194_304  # Default SAGA source byte / SPIR-V word limits.
MAX_ASSEMBLER_BYTES = 64 * 1024 * 1024
UNMAPPED_ASSEMBLY = "SPIR-V assembler/binary provenance not yet mapped; not a pass"


def profile_options(profile):
    if profile not in ("strict", "naga-native"):
        raise ValueError("unknown frontend corpus profile: " + profile)
    enabled = profile == "naga-native"
    return dict(allow_vertex_writable_storage=enabled, allow_1d_texture_sample_level_offset=enabled,
                naga_float_rounding=enabled, naga_integer_concretization=enabled,
                allow_mixed_workgroup_dimension_types=enabled,
                allow_legacy_naga_subgroups=enabled)


def profile_options_match(actual, expected):
    return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
        type(actual[key]) is bool and actual[key] == value for key, value in expected.items()
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


class AssemblyUnavailable(Exception):
    pass


def bounded_read(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise AssemblyUnavailable("file exceeds the configured byte limit")
    return data


def run_assembler(arguments, limit):
    if resource is None:
        raise AssemblyUnavailable("bounded assembler execution is unavailable on this platform")

    def limit_output():
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit + 1, limit + 1))

    # Bound child output before reading it into the parent. This CLI is single-threaded.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(arguments, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    timeout=10, preexec_fn=limit_output, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            raise AssemblyUnavailable(f"assembler could not run: {error}") from error
        if result.returncode:
            stderr.seek(0)
            detail = stderr.read(512).decode("utf-8", errors="replace").strip()
            reason = f"assembler exited with status {result.returncode}"
            raise AssemblyUnavailable(f"{reason}: {detail}" if detail else reason)
        stdout.seek(0)
        binary = stdout.read(limit + 1)
        if len(binary) > limit:
            raise AssemblyUnavailable("assembler output exceeds the configured byte limit")
        return binary


def spirv_version(binary):
    if len(binary) < 20 or len(binary) % 4:
        raise AssemblyUnavailable("assembler returned a truncated or unaligned SPIR-V binary")
    endian = {b"\x03\x02\x23\x07": "<", b"\x07\x23\x02\x03": ">"}.get(binary[:4])
    if endian is None:
        raise AssemblyUnavailable("assembler returned an invalid SPIR-V magic word")
    _, version, _, bound, schema = struct.unpack_from(endian + "5I", binary)
    if not 0x00010000 <= version <= 0x00010600 or version & 0xff0000ff or bound == 0 or schema != 0:
        raise AssemblyUnavailable("assembler returned an unsupported SPIR-V header")
    return version


def map_spirv_assembly(rows, assembler, output):
    assembly_rows = [row for row in rows if row["format"] == "spirv-assembly"]
    if not assembly_rows:
        return
    if assembler is None:
        for row in assembly_rows:
            row["unavailable_reason"] = UNMAPPED_ASSEMBLY
        return
    if output is None:
        raise ValueError("an assembly output directory is required")

    try:
        tool = assembler.resolve(strict=True)
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise AssemblyUnavailable("configured assembler is not an executable file")
        tool_hash = sha(bounded_read(tool, MAX_ASSEMBLER_BYTES))
        version_text = run_assembler([str(tool), "--version"], 4096).decode("utf-8").strip()
        if not version_text:
            raise AssemblyUnavailable("assembler did not report a version")
    except (AssemblyUnavailable, OSError, UnicodeError) as error:
        for row in assembly_rows:
            row["unavailable_reason"] = f"SPIR-V assembler unavailable: {error}"
        return

    if output.is_symlink():
        raise ValueError("assembly output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    for row in assembly_rows:
        source = Path(row["path"])
        arguments = [str(source), "-o", "-"]  # Pinned convert_snapshots_spv invocation.
        try:
            if source.is_symlink() or not source.is_file() or sha(bounded_read(source, MAX_ASSEMBLY_BYTES)) != row["sha256"]:
                raise ValueError("reference bytes drift: " + row["source"])
            binary = run_assembler([str(tool), *arguments], MAX_ASSEMBLY_BYTES)
            version = spirv_version(binary)
        except AssemblyUnavailable as error:
            row["unavailable_reason"] = f"SPIR-V assembly unavailable: {error}"
            continue
        if source.is_symlink() or sha(bounded_read(source, MAX_ASSEMBLY_BYTES)) != row["sha256"]:
            raise ValueError("reference bytes drift during assembly: " + row["source"])
        binary_hash = sha(binary)
        target = output / (binary_hash + ".spv")
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(binary)
                stream.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        row["stage"] = "module"  # Entry stages come from the binary, not a CLI hint.
        row["assembly"] = dict(source_sha256=row["sha256"], binary_path=str(target), binary_sha256=binary_hash,
                               binary_bytes=len(binary), spirv_version=version, assembler_path=str(tool),
                               assembler_sha256=tool_hash, assembler_version=version_text, assembler_arguments=arguments)
    if sha(bounded_read(tool, MAX_ASSEMBLER_BYTES)) != tool_hash:
        raise ValueError("assembler executable drift during corpus assembly")


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


def plan(reference, assembler=None, assembly_output=None, profile="strict"):
    options = profile_options(profile)
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
        rows.append(row)
    map_spirv_assembly(rows, assembler, assembly_output)
    for stem in ["deep-expression", "deep-template"]:
        source = "naga/tests/naga/" + stem + ".wgsl"
        rows.append(dict(source=source,path=str(reference/source),sha256=sha(blobs[source]),format="wgsl",stage="compute",
                         expected_parse="rejected",expected_semantic="not-run",expectation_source="naga/tests/naga/wgsl_errors.rs",
                         **configuration({},"wgsl",names)))
    for row in rows:
        row.update(profile=profile, profile_options=options.copy())
    return dict(schema=1,reference_revision=PIN,scope="frontend snapshot acceptance only, not full parity",
                profile=profile, profile_options=options,
                oracle_hashes={p:sha(blobs[p]) for p in ["naga-test/src/lib.rs","naga/tests/naga/snapshots.rs","naga/tests/naga/wgsl_errors.rs","naga/src/valid/mod.rs"]},
                copied_shader_bytes=0, rows=rows)


def summarize(plan, actual):
    rows = plan["rows"]
    if "profile" in plan:
        expected = profile_options(plan["profile"])
        if not profile_options_match(plan.get("profile_options"), expected) or any(
            row.get("profile") != plan["profile"] or not profile_options_match(row.get("profile_options"), expected) for row in rows
        ):
            raise ValueError("plan profile provenance mismatch")
    ids = [r["source"] for r in rows]
    observed = [r["source"] for r in actual]
    if len(set(ids)) != len(ids) or len(set(observed)) != len(observed) or set(ids) != set(observed):
        raise ValueError("result membership/duplicates differ from pinned denominator")
    by_id = {r["source"]:r for r in actual}
    results = []
    for row in rows:
        result = by_id[row["source"]]
        if "profile_options" in row and not profile_options_match(result.get("profile_options"), row["profile_options"]):
            raise ValueError("result configuration mismatch: profile_options")
        if result.get("sha256") != row["sha256"]:
            raise ValueError("result source hash mismatch")
        for field in ["capabilities", "parse_doc_comments", "strict_capabilities", "adjust_coordinate_space",
                      "stage", "format", "config_source", "config_sha256", "validation", "profile", "profile_options"]:
            if field in row and (field not in result or type(result[field]) is not type(row[field]) or result[field] != row[field]):
                raise ValueError("result configuration mismatch: " + field)
        if row.get("format") == "spirv-assembly":
            assembly = row.get("assembly")
            unavailable = row.get("unavailable_reason")
            if bool(assembly) == bool(unavailable) or result.get("unavailable_reason") != unavailable:
                raise ValueError("result assembly availability mismatch")
            if json.dumps(result.get("assembly"), sort_keys=True) != json.dumps(assembly, sort_keys=True):
                raise ValueError("result assembly provenance mismatch")
            if unavailable and (result.get("parse") != "not-run" or result.get("semantic") != "not-run"):
                raise ValueError("unavailable assembly cannot have a frontend result")
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
    report = dict(schema=1,reference_revision=PIN,denominator=len(rows),counts=dict(collections.Counter(r["outcome"] for r in results)),
                  scope=plan["scope"],full_saga5_complete=False,rows=results)
    if "profile" in plan:
        report.update(profile=plan["profile"], profile_options=plan["profile_options"])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--actual",type=Path)
    parser.add_argument("--spirv-as", type=Path, help="opt in using this SPIRV-Tools assembler executable")
    parser.add_argument("--profile", choices=("strict", "naga-native"), default="strict")
    args = parser.parse_args()
    p = plan(args.reference.resolve(), args.spirv_as, args.output.resolve().parent / "spirv-binaries", args.profile)
    result = summarize(p,json.loads(args.actual.read_text())) if args.actual else p
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result.get("counts",dict(inputs=len(result["rows"]))),sort_keys=True))


if __name__ == "__main__":
    main()
