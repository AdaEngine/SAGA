import collections, hashlib, json, pathlib, re, subprocess, unittest
from urllib.parse import unquote, urlsplit
from generate_inventory import extract, SPEC_REV, CTS_REV, WGPU_REV
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO=ROOT.parent
REF=REPO.parent/'wgpu'
load=lambda p: json.loads((ROOT/p).read_text())

class DocumentationTests(unittest.TestCase):
    def test_source_hashes_and_pins(self):
        for name,digest in load('source-hashes.json').items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        self.assertEqual(load('sources/gpuweb-revision.json')['sha'],SPEC_REV)
        self.assertEqual(load('sources/cts-revision.json')['sha'],CTS_REV)
        self.assertFalse(load('sources/cts-tree.json')['truncated'])
        self.assertEqual(load('sources/cts-tree.json')['sha'],load('sources/cts-revision.json')['tree'])
        for row in load('sources/fetch.json'):
            self.assertEqual(row['status'],200)
            self.assertEqual(hashlib.sha256((ROOT/'sources'/row['file']).read_bytes()).hexdigest(),row['sha256'])

    def test_inventory_unique_reproducible_and_sources(self):
        rows=load('inventory.json')['items']; self.assertEqual(rows,extract())
        self.assertEqual(len(rows),750)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        paths={n['path'] for n in load('sources/cts-tree.json')['tree']}
        register=(ROOT/'inventory.md').read_text()
        for r in rows:
            source=(ROOT/r['source']).read_text()
            if 'anchor' in r:self.assertIn('{#'+r['anchor']+'}',source)
            else:self.assertRegex(source,r'\b'+r['token']+r'\b')
            self.assertIn('id="'+r['id']+'"',register)
            self.assertIn(r['test_source'],paths)
            self.assertEqual(r['implementation'],'not-started')
            self.assertEqual(r['verification'],'not-run')
            self.assertEqual(r['evidence'],[])

    def test_reference_readonly_and_native_inventory(self):
        self.assertEqual(subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip(),WGPU_REV)
        self.assertEqual(subprocess.check_output(['git','--no-optional-locks','-C',str(REF),'status','--porcelain'],text=True),'')
        for p in ['tests/tests/wgpu-gpu','tests/tests/wgpu-validation','tests/tests/wgpu-compile','tests/tests/wgpu-dependency','tests/tests/wgpu_surface.rs','naga/tests/in','naga/tests/out','cts_runner','deno_webgpu','wgpu-types/src/features.rs']:
            self.assertTrue((REF/p).exists(),p)
        native=load('native-inventory.json')['items']
        self.assertEqual(len(native),59); self.assertEqual(len({r['id'] for r in native}),59)
        text=(REF/'wgpu-types/src/features.rs').read_text()
        actual=re.findall(r'^\s*const ([A-Z][A-Z0-9_]+) = 1 << (\d+);',text,re.M)
        self.assertEqual([(r['reference_name'],str(r['bit'])) for r in native],actual)
        for r in native:
            self.assertTrue((REF/r['test_source']).exists()); self.assertEqual(r['parity'],'proposed')

    def test_dag_and_mermaid(self):
        nodes=load('roadmap.json')['nodes']; graph={n['id']:n['depends_on'] for n in nodes}
        self.assertEqual(set(graph),{f'SAGA-{i}' for i in range(2,12)})
        expected={2:[],3:[2],4:[3],5:[2,3],6:[5,4],7:[4],8:[6,7],9:[7,8],10:[2,3],11:[8,9,10]}
        self.assertEqual(graph,{f'SAGA-{i}':[f'SAGA-{j}' for j in d] for i,d in expected.items()})
        visited=set(); active=set()
        def visit(n):
            self.assertNotIn(n,active,'cycle')
            if n in visited:return
            active.add(n)
            for d in graph[n]:self.assertIn(d,graph);visit(d)
            active.remove(n);visited.add(n)
        for n in graph:visit(n)
        mermaid=(ROOT/'roadmap.md').read_text().split('```mermaid')[1].split('```')[0]
        edges={(f'SAGA-{a}',f'SAGA-{b}') for a,b in re.findall(r'SAGA(\d+)(?:\[[^\]]*\])? --> SAGA(\d+)',mermaid)}
        self.assertEqual(edges,{(d,n) for n,deps in graph.items() for d in deps})

    def test_authored_markdown_links_and_ids(self):
        # Third-party immutable source notices are not rewritten to fix upstream relative links.
        files=[p for p in ROOT.rglob('*.md') if 'sources' not in p.relative_to(ROOT).parts]
        count=0
        for path in files:
            text=path.read_text()
            for target in re.findall(r'\]\(([^\s)]+)\)',text):
                parts=urlsplit(target)
                if parts.scheme or target.startswith('//'):continue
                dest=(path.parent/unquote(parts.path)).resolve() if parts.path else path
                self.assertTrue(dest.is_relative_to(ROOT),target)
                self.assertTrue(dest.is_file(),f'{path.name}: {target}')
                if parts.fragment:
                    self.assertIn('id="'+unquote(parts.fragment)+'"',dest.read_text(),target)
                count+=1
        self.assertGreater(count,40)

    def test_project_scope_guard_preserves_plans_but_rejects_runtime(self):
        from verify_docs import unexpected_project_files
        self.assertEqual(unexpected_project_files([
            'IDEA.md', 'docs/tools/test_docs.py', '.sloppy/plans/wgpu-rust/index.html'
        ]), [])
        for path in ['Package.swift', 'Package.resolved', 'Sources/SAGA.swift',
                     'Cargo.toml', '.sloppy/runtime.swift']:
            self.assertEqual(unexpected_project_files([path]), [path])

    def test_report_distinguishes_platform_choice_from_open_scope(self):
        report=load('verification-results.json')
        self.assertEqual(report['decisions']['native_backend'], 'Apple/Metal')
        self.assertEqual(report['decisions']['web_backend'], 'browser WebGPU')
        self.assertEqual(report['decisions']['additional_shader_formats'], 'awaiting-user')
        self.assertEqual(report['decisions']['native_extension_parity'], 'awaiting-user')
        self.assertEqual(report['executed'], dict(swift_build=False, swift_tests=False,
                                                rust_tests=False, gpu=False, cts=False))
        self.assertNotIn('Platform and native parity remain proposed', report['limitations'])

    def test_scope_and_required_outputs(self):
        for name in ['roadmap.md','conformance.md','inventory.md','inventory.json','native-inventory.json','provenance.md','adr/implement-independent-webgpu.md','verification.md']:
            self.assertTrue((ROOT/name).is_file(),name)
        self.assertFalse((REPO/'Package.swift').exists())
        self.assertEqual((REPO/'IDEA.md').read_text(),'rewrite wgpu to swift\n')
        from verify_docs import unexpected_project_files
        paths=subprocess.check_output(['git','-C',str(REPO),'ls-files','--cached',
                                       '--others','--exclude-standard','-z'],text=True)
        self.assertEqual(unexpected_project_files(paths.rstrip('\0').split('\0')), [])
        self.assertIn('not a certified clean-room',(ROOT/'adr/implement-independent-webgpu.md').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
