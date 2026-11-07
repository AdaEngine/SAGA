"""Positive/negative controls for the audit, not shader conformance tests."""
import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

import frontend_coverage as fc


class CoverageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.expected = {
            'schema': 1, 'reference_revision': fc.PIN,
            'input_fingerprints': {'inventory': 'a' * 64},
            'metadata': {'options': {'wgsl': ['capabilities']}, 'capabilities': ['FLOAT64']},
            'records': [
                {'id': 'fixture/a.wgsl', 'kind': 'fixture', 'source': 'a.wgsl',
                 'sha256': 'b' * 64, 'role': 'input', 'format': 'wgsl'},
                {'id': 'fixture/a.toml', 'kind': 'fixture', 'source': 'a.toml',
                 'sha256': 'c' * 64, 'role': 'config', 'format': 'configuration'},
                {'id': 'fixture/out.wgsl', 'kind': 'fixture', 'source': 'out.wgsl',
                 'sha256': 'd' * 64, 'role': 'output', 'format': 'wgsl'},
                {'id': 'requirement/type', 'kind': 'requirement', 'source': 'valid.rs',
                 'sha256': 'e' * 64, 'role': 'rule', 'format': 'shared',
                 'requirement': 'type', 'polarities': ['positive', 'negative']},
            ]}
        self.manifest, self.provenance = fc.initial_documents(self.expected)

    def audit(self):
        return fc.audit_documents(self.manifest, self.provenance, self.expected, self.root)

    def rejected(self, text):
        with self.assertRaisesRegex(fc.AuditError, text):
            self.audit()

    def add_result(self, record, polarity='positive'):
        case = 'Suite.case_' + polarity
        log = 'Test ' + case + ' passed; 1 test passed\n'
        log_path = 'docs/evidence/saga-5/coverage/' + polarity + '.log'
        path = self.root / log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(log)
        production = self.root / 'Sources/SAGAFrontends/Example.swift'
        production.parent.mkdir(parents=True, exist_ok=True)
        production.write_text('// synthetic TEST CONTROL, not a real compiler\n')
        result_path = 'docs/evidence/saga-5/coverage/' + polarity + '.json'
        result = {'schema': 1, 'reference_revision': fc.PIN, 'kind': 'native-frontend-semantic',
                  'command': ['/usr/bin/swift', 'test', '--filter', case], 'exit_code': 0,
                  'tests_executed': 1, 'case': case, 'record_id': record['id'],
                  'reference_sha256': record['sha256'], 'polarity': polarity,
                  'format': record['format'], 'stage': 'compute',
                  'diagnostic': {'code': 'type-mismatch', 'location_unit': 'utf8', 'start': 0, 'end': 1} if polarity == 'negative' else None,
                  'expected': 'accept' if polarity == 'positive' else 'reject',
                  'actual': 'accept' if polarity == 'positive' else 'reject',
                  'options': {'capabilities': []} if record['format'] == 'wgsl' else {}, 'capabilities': [], 'assertions': ['typed result checked'],
                  'log': log_path, 'log_sha256': hashlib.sha256(log.encode()).hexdigest(),
                  'implementation_hashes': {'Sources/SAGAFrontends/Example.swift':
                      hashlib.sha256(production.read_bytes()).hexdigest()}}
        (self.root / result_path).write_text(json.dumps(result))
        record['status'] = 'tested'
        record['result_files'].append(result_path)
        return result, self.root / result_path

    def test_honest_unimplemented_is_audit_success_not_parity(self):
        result = self.audit()
        self.assertFalse(result['compiler_parity'])
        self.assertEqual(result['tested'], 0)
        self.assertEqual(result['records'], 4)

    def test_positive_execution_control(self):
        self.add_result(self.manifest['records'][0])
        self.assertEqual(self.audit()['tested'], 1)

    def test_rule_requires_both_polarities(self):
        self.add_result(self.manifest['records'][3])
        self.rejected('polarities')
        self.add_result(self.manifest['records'][3], 'negative')
        self.assertEqual(self.audit()['tested'], 1)

    def test_dropped_denominator(self):
        self.manifest['records'].pop()
        self.rejected('denominator')

    def test_duplicate_denominator(self):
        self.manifest['records'].append(copy.deepcopy(self.manifest['records'][0]))
        self.rejected('duplicate')

    def test_unknown_extra_item(self):
        self.manifest['records'][0]['id'] = 'invented'
        self.rejected('denominator')

    def test_unknown_status(self):
        self.manifest['records'][0]['status'] = 'covered'
        self.rejected('status')

    def test_fake_tested_without_result(self):
        self.manifest['records'][0]['status'] = 'tested'
        self.rejected('result')

    def test_unrun_cannot_have_results(self):
        self.manifest['records'][0]['result_files'] = ['invented.json']
        self.rejected('unexecuted')

    def test_unsupported_requires_reason(self):
        self.manifest['records'][0]['status'] = 'unsupported'
        self.manifest['records'][0]['reason'] = ''
        self.rejected('reason')
        self.manifest['records'][0]['reason'] = 'feature not implemented'
        self.assertEqual(self.audit()['tested'], 0)

    def test_hash_drift(self):
        self.manifest['records'][0]['sha256'] = 'f' * 64
        self.rejected('metadata drift')

    def test_pin_drift(self):
        self.manifest['reference_revision'] = '0' * 40
        self.rejected('header')

    def test_options_cannot_be_dropped(self):
        self.manifest['metadata']['options'] = {}
        self.rejected('header')

    def test_demo_cannot_claim_parity(self):
        self.manifest['compiler_parity'] = True
        self.rejected('parity')

    def test_false_summary_cannot_claim_parity(self):
        self.manifest['summary'] = 'all Naga passed'
        self.rejected('fields')

    def test_classification_is_immutable(self):
        self.manifest['records'][1]['role'] = 'input'
        self.rejected('metadata drift')

    def test_config_is_not_shader_execution(self):
        self.add_result(self.manifest['records'][1])
        self.rejected('non-executable')

    def test_output_is_not_shader_execution(self):
        self.add_result(self.manifest['records'][2])
        self.rejected('non-executable')

    def test_provenance_dropped(self):
        self.provenance['fixtures'].pop()
        self.rejected('provenance')

    def test_root_license_is_not_fixture_clearance(self):
        self.provenance['fixtures'][0]['license_status'] = 'cleared-under-root-MIT'
        self.rejected('provenance')

    def test_no_unaudited_copies(self):
        self.provenance['fixtures'][0]['copied_to'] = ['Tests/copied.wgsl']
        self.rejected('provenance')

    def test_result_must_have_actual_success(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['actual'] = 'reject'
        path.write_text(json.dumps(result))
        self.rejected('actual')

    def test_zero_tests_not_success(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['tests_executed'] = 0
        path.write_text(json.dumps(result))
        self.rejected('tests')

    def test_log_hash_drift(self):
        result, path = self.add_result(self.manifest['records'][0])
        (self.root / result['log']).write_text('changed')
        self.rejected('log')

    def test_fake_nonzero_count_with_zero_test_log(self):
        result, path = self.add_result(self.manifest['records'][0])
        log = 'Test ' + result['case'] + ' passed; 0 tests passed\n'
        (self.root / result['log']).write_text(log)
        result['log_sha256'] = hashlib.sha256(log.encode()).hexdigest()
        path.write_text(json.dumps(result))
        self.rejected('count')

    def test_source_hash_drift(self):
        self.add_result(self.manifest['records'][0])
        (self.root / 'Sources/SAGAFrontends/Example.swift').write_text('changed')
        self.rejected('implementation')

    def test_wrong_result_identity(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['record_id'] = 'fixture/another.wgsl'
        path.write_text(json.dumps(result))
        self.rejected('identity')

    def test_parser_demo_result_not_semantic(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['kind'] = 'lexer-demo'
        path.write_text(json.dumps(result))
        self.rejected('kind')

    def test_result_format_mismatch(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['format'] = 'glsl'
        path.write_text(json.dumps(result))
        self.rejected('format')

    def test_unknown_result_capability(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['capabilities'] = ['INVENTED']
        path.write_text(json.dumps(result))
        self.rejected('capabilities')

    def test_result_must_record_option_defaults(self):
        result, path = self.add_result(self.manifest['records'][0])
        result['options'] = {}
        path.write_text(json.dumps(result))
        self.rejected('option fields')

    def test_negative_requires_typed_diagnostic(self):
        result, path = self.add_result(self.manifest['records'][0], 'negative')
        result.pop('diagnostic')
        path.write_text(json.dumps(result))
        self.rejected('diagnostic')

    def test_traversal_result_path(self):
        self.manifest['records'][0].update(status='tested', result_files=['../outside.json'])
        self.rejected('path')

    def test_classify_formats_and_deep_negative(self):
        self.assertEqual(fc.classify('naga/tests/in/a.toml'), ('config', 'configuration'))
        self.assertEqual(fc.classify('naga/tests/out/wgsl/a.wgsl'), ('output', 'wgsl'))
        self.assertEqual(fc.classify('naga/tests/naga/deep-template.wgsl'), ('negative-input', 'wgsl'))
        self.assertEqual(fc.classify('naga/tests/in/spv/a.spvasm'), ('input', 'spirv-assembly'))
        self.assertEqual(fc.classify('naga/tests/in/a.slang'), ('upstream-auxiliary', 'slang'))


class PinnedDiscoveryTests(unittest.TestCase):
    def test_full_pinned_tree_and_real_manifest(self):
        expected = fc.authoritative(fc.ROOT.parent / 'wgpu')
        ids = [r['id'] for r in expected['records']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(r['kind'] == 'lexical-feature' for r in expected['records']), 557)
        self.assertEqual(sum(r['kind'] == 'fixture' for r in expected['records']), 1425)
        self.assertTrue(any('wgsl_errors.rs' in r['source'] and r['role'] == 'inline-case' for r in expected['records']))
        self.assertTrue(any('validation.rs' in r['source'] and r['role'] == 'inline-case' for r in expected['records']))
        self.assertEqual(len(expected['metadata']['capabilities']), 46)
        self.assertEqual(expected['metadata']['options']['spirv']['fields'], ['adjust_coordinate_space', 'strict_capabilities', 'block_ctx_dump_prefix'])
        manifest = json.loads((fc.ROOT / 'docs/frontend-coverage.json').read_text())
        provenance = json.loads((fc.ROOT / 'docs/frontend-fixture-provenance.json').read_text())
        self.assertFalse(fc.audit_documents(manifest, provenance, expected)['compiler_parity'])


if __name__ == '__main__':
    unittest.main()
