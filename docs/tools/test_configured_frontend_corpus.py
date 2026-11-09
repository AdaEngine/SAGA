import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from configured_frontend_corpus import configuration, summarize, map_spirv_assembly, DEFAULT_CAPS


def assembly_row(root):
    path = root / "shader.spvasm"
    source = b"; pinned source fixture\n"
    path.write_bytes(source)
    return dict(source="shader.spvasm", path=str(path), sha256=hashlib.sha256(source).hexdigest(),
                format="spirv-assembly", stage="compute", expected_parse="accepted",
                expected_semantic="accepted", validation=True)


def fake_assembler(root, output, status=0):
    path = root / "spirv-as"
    escaped = "".join(f"\\{byte:03o}" for byte in output)
    path.write_text("#!/bin/sh\n"
                    'if [ "$1" = "--version" ]; then echo "SPIRV-Tools test version"; exit 0; fi\n'
                    'if [ "$#" -ne 3 ] || [ "$2" != "-o" ] || [ "$3" != "-" ]; then exit 77; fi\n'
                    f"printf '{escaped}'\nexit {status}\n")
    path.chmod(0o755)
    return path

class ConfiguredCorpusTests(unittest.TestCase):
    def test_pinned_defaults_differ_from_frontend_api_defaults(self):
        c = configuration({},"wgsl",{})
        self.assertEqual(c["capabilities"],DEFAULT_CAPS)
        self.assertFalse(c["parse_doc_comments"])
        self.assertFalse(c["adjust_coordinate_space"])
        self.assertIn("METAL",c["targets"])
        self.assertEqual(configuration({},"glsl",{})["targets"],["WGSL"])

    def test_options_and_capability_masks_are_not_silently_dropped(self):
        c = configuration({"capabilities":"A | B","wgsl-in":{"parse_doc_comments":True},"spv-in":{"adjust_coordinate_space":True}},"wgsl",{"A":1,"B":26})
        self.assertEqual(c["capabilities"],(1<<1)|(1<<26))
        self.assertTrue(c["parse_doc_comments"])
        self.assertTrue(c["adjust_coordinate_space"])
        self.assertFalse(configuration({"targets":"IR | NO_VALIDATION"},"glsl",{})["validation"])

    def test_unknown_configs_and_wrong_option_types_fail_closed(self):
        for d in [{"capabilities":"FAKE"},{"targets":"FAKE"},{"wgsl-in":{"typo":True}},
                  {"spv-in":{"adjust_coordinate_space":1}},{"wgsl-in":{"parse_doc_comments":"false"}},{"typo":1}]:
            with self.assertRaises(ValueError):configuration(d,"wgsl",{})

    def test_expectation_mismatches_and_unavailable_never_count_as_pass(self):
        rows = [dict(source=str(i),sha256="hash",expected_parse="accepted",expected_semantic="accepted",validation=True) for i in range(4)]
        actual = [dict(source="0",sha256="hash",validation=True,parse="accepted",semantic="accepted"),
                  dict(source="1",sha256="hash",validation=True,parse="rejected",semantic="not-run"),
                  dict(source="2",sha256="hash",validation=True,parse="accepted",semantic="rejected"),
                  dict(source="3",sha256="hash",validation=True,unavailable_reason="no assembler")]
        report=summarize(dict(rows=rows,scope="test"),actual)
        self.assertEqual(report["counts"],{"pass":1,"fail":2,"unsupported":1})
        self.assertFalse(report["full_saga5_complete"])

    def test_missing_duplicate_extra_and_hash_drift_results_reject(self):
        row=dict(source="a",sha256="hash",expected_parse="accepted",expected_semantic="accepted",validation=True)
        p=dict(rows=[row],scope="test")
        for actual in [[],[row,row],[dict(row,source="b")],[dict(row,sha256="drift")]]:
            with self.assertRaises(ValueError):summarize(p,actual)

    def test_report_rejects_option_or_config_provenance_drift(self):
        row=dict(source="a",sha256="hash",expected_parse="accepted",expected_semantic="accepted",validation=True,
                 capabilities=3,parse_doc_comments=False,strict_capabilities=True,adjust_coordinate_space=False,
                 stage="compute",format="wgsl",config_source="a.toml",config_sha256="config-hash")
        actual=dict(row,parse="accepted",semantic="accepted")
        for field in ["capabilities","parse_doc_comments","strict_capabilities","adjust_coordinate_space",
                      "stage","format","config_source","config_sha256","validation"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(dict(rows=[row],scope="test"),[dict(actual,**{field:"drift"})])
            absent=dict(actual);del absent[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                summarize(dict(rows=[row],scope="test"),[absent])

    def test_no_validation_and_error_rejection_are_separate_obligations(self):
        a=dict(source="a",sha256="hash",expected_parse="accepted",expected_semantic="not-run",validation=False)
        b=dict(source="b",sha256="hash",expected_parse="rejected",expected_semantic="not-run",validation=True)
        report=summarize(dict(rows=[a,b],scope="test"),[dict(a,parse="accepted"),dict(b,parse="rejected")])
        self.assertEqual(report["counts"],{"parse-only":1,"rejection-only":1})

    def test_assembly_mapping_binds_binary_to_pinned_source_and_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            row = assembly_row(root)
            binary = struct.pack("<5I", 0x07230203, 0x00010600, 0, 1, 0)
            tool = fake_assembler(root, binary)
            map_spirv_assembly([row], tool, root / "binaries")
            self.assertNotIn("unavailable_reason", row)
            self.assertEqual(row["stage"], "module")
            record = row["assembly"]
            self.assertEqual(record["source_sha256"], row["sha256"])
            self.assertEqual(record["binary_sha256"], hashlib.sha256(binary).hexdigest())
            self.assertEqual(record["binary_bytes"], len(binary))
            self.assertEqual(record["spirv_version"], 0x00010600)
            self.assertEqual(Path(record["binary_path"]).read_bytes(), binary)
            self.assertEqual(record["assembler_path"], str(tool))
            self.assertEqual(record["assembler_sha256"], hashlib.sha256(tool.read_bytes()).hexdigest())
            self.assertEqual(record["assembler_version"], "SPIRV-Tools test version")
            self.assertEqual(record["assembler_arguments"], [row["path"], "-o", "-"])

            configured = dict(rows=[row], scope="test")
            actual = dict(row, parse="accepted", semantic="accepted")
            self.assertEqual(summarize(configured, [actual])["counts"], {"pass": 1})
            for field in record:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    summarize(configured, [dict(actual, assembly=dict(record, **{field: "drift"}))])
            with self.assertRaises(ValueError):
                summarize(configured, [dict(actual, assembly=None)])

    def test_assembly_unavailable_is_not_a_pass_without_a_verified_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for tool in [None, root / "missing", fake_assembler(root, b"not SPIR-V")]:
                row = assembly_row(root)
                map_spirv_assembly([row], tool, root / "binaries")
                self.assertNotIn("assembly", row)
                self.assertIn("unavailable_reason", row)
                configured = dict(rows=[row], scope="test")
                observed = dict(row, parse="not-run", semantic="not-run")
                self.assertEqual(summarize(configured, [observed])["counts"], {"unsupported": 1})
                with self.assertRaises(ValueError):
                    summarize(configured, [dict(observed, unavailable_reason=None, parse="accepted", semantic="accepted")])

    def test_assembly_tool_failure_and_source_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            row = assembly_row(root)
            binary = struct.pack("<5I", 0x07230203, 0x00010300, 0, 1, 0)
            tool = fake_assembler(root, binary, status=1)
            map_spirv_assembly([row], tool, root / "binaries")
            self.assertIn("unavailable_reason", row)
            self.assertFalse(list((root / "binaries").glob("*.spv")))

            row = assembly_row(root)
            Path(row["path"]).write_bytes(b"drift")
            with self.assertRaises(ValueError):
                map_spirv_assembly([row], fake_assembler(root, binary), root / "binaries")

    def test_assembly_output_size_and_version_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            invalid = [
                struct.pack("<5I", 0x07230203, 0x00010300, 0, 1, 0) + bytes(16),
                struct.pack("<5I", 0x07230203, 0x00010700, 0, 1, 0),
            ]
            for binary in invalid:
                row = assembly_row(root)
                with patch("configured_frontend_corpus.MAX_ASSEMBLY_BYTES", 32):
                    map_spirv_assembly([row], fake_assembler(root, binary), root / "binaries")
                self.assertIn("unavailable_reason", row)
                self.assertNotIn("assembly", row)

if __name__ == "__main__":unittest.main()
