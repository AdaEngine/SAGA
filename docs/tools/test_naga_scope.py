import os, json, pathlib, subprocess, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
REF=pathlib.Path(os.environ.get('SAGA_REFERENCE', str(ROOT.parent.parent/'wgpu')))
class NagaScopeTests(unittest.TestCase):
    def setUp(self):
        self.d=json.loads((ROOT/'naga-scope.json').read_text())
    def test_user_choice_and_pin(self):
        self.assertEqual(self.d['scope'],'full-naga')
        self.assertEqual(self.d['reference_revision'],'a55352911bb26a1b33bc8d5fc03bdebb0a967f5a')
        self.assertIn('Status: accepted',(ROOT/'adr/port-full-naga-on-swift.md').read_text())
    @unittest.skipUnless(os.environ.get("SAGA_REFERENCE"), "set SAGA_REFERENCE for format directory audit")
    def test_reference_format_inventory(self):
        self.assertEqual(set(self.d['frontends']),{'wgsl','glsl','spv'})
        self.assertEqual(set(self.d['backends']),{'wgsl','glsl','spv','msl','hlsl','dot'})
        for kind,folder in [('frontends','front'),('backends','back')]:
            actual={p.name for p in (REF/'naga/src'/folder).iterdir() if p.is_dir()}
            self.assertEqual(set(self.d[kind]),actual)
    def test_all_components_unimplemented_and_owned(self):
        self.assertEqual(len(self.d['items']),16)
        self.assertEqual(len({r['id'] for r in self.d['items']}),16)
        for r in self.d['items']:
            self.assertEqual(r['implementation'],'not-started')
            self.assertEqual(r['verification'],'not-run')
            if os.environ.get('SAGA_REFERENCE'): self.assertTrue((REF/r['source']).exists(),r['source'])
            self.assertIn(r['owner'],['SAGA-3','SAGA-5','SAGA-6'])
    def test_capability_not_all_to_all(self):
        self.assertFalse(self.d['assume_all_to_all'])
        self.assertFalse(self.d['runtime_backend_expansion'])
        self.assertTrue(self.d['binary_ir_interchange_in_scope'])
        self.assertFalse(self.d['runtime_implemented'])
