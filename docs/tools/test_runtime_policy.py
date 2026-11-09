import copy, hashlib, importlib.util, json, os, pathlib, subprocess, unittest
from runtime_policy import ROOT, POLICY, audit, audit_package, audit_source, unexpected_project_files
from generate_compiler_inventory import generate, PIN

class RuntimePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package=json.loads(subprocess.check_output(['swift','package','--package-path',str(ROOT),'dump-package'],text=True))
    def test_current_package_and_sources(self):
        self.assertEqual(audit()['external_dependencies'],0)
    def test_external_runtime_and_transitive_dependency_rejected(self):
        for name in ['Dawn','wgpu','Naga','innocent-wrapper']:
            package=copy.deepcopy(self.package); package['dependencies']=[{'url':'https://example.invalid/'+name}]
            self.assertIn('external dependencies forbidden',audit_package(package))
    def test_nonlocal_binary_system_and_unsafe_settings_rejected(self):
        for kind in ['binary','system','plugin','executable']:
            package=copy.deepcopy(self.package); next(t for t in package['targets'] if t['name']=='SAGATypes')['type']=kind
            self.assertTrue(audit_package(package))
        for field,value in [('settings',[{'unsafeFlags':['-lswiftNaga']}]),('path','../wgpu'),('sources',['bridge.c']),('resources',[{'path':'engine.wasm'}])]:
            package=copy.deepcopy(self.package); package['targets'][0][field]=value
            self.assertTrue(audit_package(package))
        package=copy.deepcopy(self.package); package['targets'][0]['dependencies']=[{'product':['wgpu','wgpu',None,None]}]
        self.assertTrue(audit_package(package))
    def test_only_named_diagnostic_executables_are_allowed(self):
        for name in ['SAGA', 'SAGACompiler', 'SAGAMetal', 'SAGAWebGPU']:
            package=copy.deepcopy(self.package)
            next(p for p in package['products'] if p['name']==name)['type']={'executable':None}
            self.assertTrue(audit_package(package))
        package=copy.deepcopy(self.package)
        next(p for p in package['products'] if p['name']=='SAGAMetalDiagnostic')['type']={'library':['automatic']}
        self.assertTrue(audit_package(package))
        package=copy.deepcopy(self.package)
        next(p for p in package['products'] if p['name']=='SAGAWebDiagnostic')['type']={'library':['automatic']}
        self.assertTrue(audit_package(package))
        package=copy.deepcopy(self.package)
        package['targets'].append({'name':'UnexpectedCLI','type':'executable','dependencies':[]})
        self.assertTrue(audit_package(package))

    def test_wrong_tools_language_and_graph_rejected(self):
        for key,value in [('toolsVersion',{'_version':'6.3.0'}),('swiftLanguageVersions',['5'])]:
            package=copy.deepcopy(self.package);package[key]=value;self.assertTrue(audit_package(package))
        package=copy.deepcopy(self.package);package['targets'][0]['dependencies']=[{'byName':['SAGAMetal',None]}]
        self.assertTrue(audit_package(package))
    def test_import_and_safety_escape_rejected(self):
        for text in ['import Metal','import Vulkan','import Naga','import struct Metal.MTLSize','@unchecked Sendable','@_silgen_name("naga")','@preconcurrency import SAGABackend','nonisolated(unsafe) var x=0','unsafeBitCast(x,to:Int.self)']:
            self.assertTrue(audit_source('Sources/SAGACore/Bad.swift',text),text)
        self.assertEqual(audit_source('Sources/SAGAMetal/Good.swift','#if canImport(Metal)\nimport Metal\n#endif'),[])
    def test_web_abi_is_confined_to_the_reactor(self):
        path='Sources/SAGAWebDiagnostic/WebReactor.swift'
        self.assertEqual(audit_source(path,(ROOT/path).read_text()),[])
        for escape in ['@_extern(c, "naga")', '@_extern(wasm, module: "other", name: "request")',
                       '@_expose(wasm, "unreviewed")', '@_silgen_name("anything")']:
            self.assertTrue(audit_source(path, f'#if os(WASI) && compiler(>=6.3)\n{escape}\n#endif'))
        self.assertTrue(audit_source('Sources/SAGAWebDiagnostic/WebGPUHost.swift',
                                     '@_extern(wasm, module: "saga_host", name: "request")'))
        self.assertEqual(unexpected_project_files(['web/index.html','web/wasi.mjs','web/webgpu.mjs',
                                                   'web/reactor.mjs','scripts/test-web-diagnostic.mjs']),[])
        self.assertEqual(unexpected_project_files(['web/foreign.wasm','web/foreign.mjs']),
                         ['web/foreign.wasm','web/foreign.mjs'])
    def test_project_scope_rejects_foreign_sources_and_cargo(self):
        for path in ['Cargo.toml','Package.resolved','Sources/SAGATypes/bridge.c','Sources/SAGATypes/module.modulemap','Sources/Evil/a.swift','.sloppy/runtime.swift','libNaga.dylib','scripts/naga.wasm','docs/hidden/Cargo.toml','docs/foreign.c']:
            self.assertEqual(unexpected_project_files([path]),[path])
        self.assertEqual(unexpected_project_files(['Package.swift','README.md','Sources/SAGATypes/Types.swift']),[])
    def test_documentation_scope(self):
        self.assertEqual(unexpected_project_files([
            'AGENTS.md', '.spi.yml', '.agents/skills/saga-development/SKILL.md',
            'Sources/SAGA/SAGA.docc/SAGA.md',
        ]), [])
        unexpected = [
            '.agents/skills/saga-development/tool.sh', 'Sources/SAGA/Notes.md',
            'Sources/SAGA/SAGA.docc/foreign.js', 'Sources/Unknown/Unknown.docc/Unknown.md',
        ]
        self.assertEqual(unexpected_project_files(unexpected), unexpected)
    def test_historical_fixture_hashes_and_original_guard(self):
        baseline=ROOT/'docs/baseline/saga-2'
        for path,digest in json.loads((baseline/'hashes.json').read_text()).items():
            self.assertEqual(hashlib.sha256((baseline/path).read_bytes()).hexdigest(),digest,path)
        self.assertEqual((baseline/'verification-results.json').read_bytes(),(ROOT/'docs/verification-results.json').read_bytes())
        original=(baseline/'test_docs.py').read_text()
        self.assertIn("self.assertFalse((REPO/'Package.swift').exists())",original)
        spec=importlib.util.spec_from_file_location('frozen',baseline/'verify_docs.py'); frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        self.assertEqual(frozen.unexpected_project_files(['Package.swift']),['Package.swift'])
    def test_offline_compiler_inventory_denominator(self):
        inventory=json.loads((ROOT/'docs/compiler-feature-inventory.json').read_text())
        self.assertEqual(inventory['reference_revision'],PIN)
        self.assertEqual(inventory['counts']['items'],len(inventory['items']))
        self.assertEqual(inventory['counts']['fixtures'],len(inventory['fixtures']))
        self.assertEqual(len({r['id'] for r in inventory['items']}),len(inventory['items']))
        self.assertEqual(set(inventory['counts']['by_component']),{c['id'] for c in inventory['components']})
        for row in inventory['items']:
            self.assertIn(row['source'],inventory['source_hashes']);self.assertGreater(row['line'],0)
            self.assertEqual(row['implementation'],'not-started');self.assertEqual(row['verification'],'not-run')
        for row in inventory['fixtures']:
            self.assertRegex(row['sha256'],r'^[a-f0-9]{64}$'); self.assertEqual(row['verification'],'not-run')
        self.assertTrue(inventory['limitations'])
    @unittest.skipUnless(os.environ.get('SAGA_REFERENCE'),'set SAGA_REFERENCE for pinned inventory reproducibility')
    def test_reference_inventory_reproducibility(self):
        self.assertEqual(generate(os.environ['SAGA_REFERENCE']),json.loads((ROOT/'docs/compiler-feature-inventory.json').read_text()))

class BuildEvidenceTests(unittest.TestCase):
    def test_swift_test_log_requires_nonzero_testing_summary(self):
        spec=importlib.util.spec_from_file_location('test_log_guard',ROOT/'scripts/check-swift-test-log.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        for text in ['', 'Executed 0 tests, with 0 failures', 'Test run with 0 tests passed after 0.001 seconds.', 'Test run with 2 tests failed after 1 second.', 'recorded an issue\nTest run with 1 test passed after 1 second.']:
            with self.assertRaises(ValueError):guard.require_nonzero_success(text)
        self.assertEqual(guard.require_nonzero_success('Executed 0 tests\nTest run with 25 tests in 0 suites passed after 0.001 seconds.'),25)
    def test_sdk_checksum_matches_archived_official_metadata(self):
        pin=json.loads((ROOT/'scripts/swift-web-sdk.json').read_text())
        self.assertEqual(pin['release'],'6.3.2')
        wasm=next(p for p in pin['official_release_record']['platforms'] if p['platform']=='wasm-sdk')
        self.assertEqual(pin['wasm_sdk']['checksum'],wasm['checksum'])
        self.assertEqual(wasm['checksum'],'a61f0584c93283589f8b2f42db05c1f9a182b506c2957271402992655591dd7c')
        self.assertEqual(pin['wasm_sdk']['url'],'https://download.swift.org/swift-6.3.2-release/wasm-sdk/swift-6.3.2-RELEASE/swift-6.3.2-RELEASE_wasm.artifactbundle.tar.gz')
