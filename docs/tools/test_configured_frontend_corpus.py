import unittest
from configured_frontend_corpus import configuration, summarize, DEFAULT_CAPS

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

if __name__ == "__main__":unittest.main()
