#!/usr/bin/env python3
"""Documentation baseline checks only; no Swift, Rust, GPU or CTS execution."""
import collections, hashlib, importlib.util, json, pathlib, re, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = pathlib.Path('/Users/vlad-prusakov/Developer/wgpu')
checks = []
def check(value, message):
    if not value:
        raise AssertionError(message)
    checks.append(message)
def load(name): return json.loads((ROOT/name).read_text())
def unexpected_project_files(paths):
    # Existing planning artifacts are not runtime files or evidence of this turn's edits.
    return [p for p in paths if p and p != 'IDEA.md'
            and not p.startswith(('docs/', '.sloppy/plans/'))]

def main():
    (ROOT/'verification-results.json').write_text(json.dumps({'status':'running','scope':'documentation only'})+'\n')
    inv = load('inventory.json'); native = load('native-inventory.json')
    ids = [r['id'] for r in inv['items'] + native['items']]
    check(len(ids)==len(set(ids)), '809 unique inventory IDs across normative-review and native registers')
    check(len(inv['items'])==750 and len(native['items'])==59, 'Frozen coarse denominators: 750 review units; 59 native flags')
    for r in inv['items']:
        text=(ROOT/r['source']).read_text()
        check(('{#'+r['anchor']+'}') in text if r['kind']=='section' else bool(re.search(r'\b'+re.escape(r['token'])+r'\b',text)), 'Source resolves: '+r['id'])
        check(r['implementation']=='not-started' and r['verification']=='not-run' and not r['evidence'], 'Honest baseline state: '+r['id'])
    tree=load('sources/cts-tree.json')
    check(tree['truncated'] is False and tree['sha']==load('sources/cts-revision.json')['tree'] and load('sources/cts-revision.json')['sha']==inv['cts_revision'], 'CTS recursive tree complete and revision matched')
    paths={r['path'] for r in tree['tree']}
    for r in inv['items']:
        check(r['test_source'] in paths,'Planned CTS source directory exists: '+r['id'])
    for r in native['items']:
        check((REF/r['source']).is_file() and r['reference_name'] in (REF/r['source']).read_text(), 'Native token resolves: '+r['id'])
        check(r['parity']=='proposed' and r['implementation']=='not-started' and r['verification']=='not-run', 'Native parity not silently accepted: '+r['id'])
    spec=importlib.util.spec_from_file_location('generate_inventory', ROOT/'tools/generate_inventory.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    check(mod.extract()==inv['items'], 'Generated inventory equals deterministic extraction without rewriting')
    for path,digest in load('source-hashes.json').items():
        check(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,'SHA256: '+path)
    check(subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip()==inv['reference_revision'],'Read-only wgpu HEAD matches pinned reference')
    for path in ['tests/tests/wgpu-gpu','tests/tests/wgpu-validation','tests/tests/wgpu-compile','tests/tests/wgpu-dependency','tests/tests/wgpu_surface.rs','naga/tests/in','naga/tests/out','cts_runner','deno_webgpu','wgpu-types/src/features.rs']:
        check((REF/path).exists(),'Reference path exists: '+path)
    nodes=load('roadmap.json')['nodes']; dag={n['id']:n['depends_on'] for n in nodes}
    check(set(dag)=={f'SAGA-{i}' for i in range(2,12)},'DAG contains exactly SAGA-2..11')
    expected={2:[],3:[2],4:[3],5:[2,3],6:[5,4],7:[4],8:[6,7],9:[7,8],10:[2,3],11:[8,9,10]}
    check(dag=={f'SAGA-{k}':[f'SAGA-{v}' for v in vs] for k,vs in expected.items()},'DAG matches task-tool prerequisites')
    seen=set(); active=set()
    def visit(n):
        assert n in dag and n not in active, 'Unknown dependency or cycle: '+n
        if n in seen:return
        active.add(n)
        for dep in dag[n]:visit(dep)
        active.remove(n);seen.add(n)
    for n in dag:visit(n)
    check(len(seen)==10,'DAG acyclic')
    md=list(ROOT.glob('*.md'))+list((ROOT/'adr').glob('*.md'))
    link_count=0
    def anchors(text):
        explicit=set(re.findall(r'<a\s+id="([^"]+)"',text))
        for heading in re.findall(r'^#{1,6}\s+(.+)$',text,re.M):
            explicit.add(re.sub(r'[^\w\- ]','',heading.lower()).replace(' ','-'))
        return explicit
    for path in md:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if re.match(r'https?://',target):continue
            target=target.strip('<>'); dest,_,anchor=target.partition('#'); file=(path.parent/dest).resolve() if dest else path
            check(file.is_file(),'Local link exists: '+path.name+' -> '+target)
            if anchor:check(anchor in anchors(file.read_text()),'Anchor exists: '+target)
            link_count+=1
    register=(ROOT/'inventory.md').read_text()
    check(set(re.findall(r'<a id="([^"]+)"',register))=={r['id'] for r in inv['items']},'Every baseline ID has exactly its register anchor')
    check(not (ROOT.parent/'Package.swift').exists(),'No Package.swift/runtime introduced')
    paths=subprocess.check_output(['git','-C',str(ROOT.parent),'ls-files','--cached',
                                   '--others','--exclude-standard','-z'],text=True)
    check(not unexpected_project_files(paths.rstrip('\0').split('\0')),
          'Project files contain documentation and existing plans, not runtime/dependencies')
    report={'status':'pass','scope':'documentation only','checks':len(checks),'local_links':link_count,'inventory_units':750,'native_flags':59,'dag_nodes':10,'dag_edges':sum(map(len,dag.values())),'decisions':{'native_backend':'Apple/Metal','web_backend':'browser WebGPU','additional_shader_formats':'awaiting-user','native_extension_parity':'awaiting-user'},'executed':{'swift_build':False,'swift_tests':False,'rust_tests':False,'gpu':False,'cts':False},'limitations':['Normative assertion count unknown; coarse inventory not exhaustive','Expanded CTS case count unknown; no CTS executed','Apple/Metal and browser WebGPU selected; exact matrix, additional shader formats and native parity remain open','External living anchors not exhaustively requested; pinned source anchors checked locally'],'checks_detail':checks}
    (ROOT/'verification-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks_detail'},indent=2))
if __name__=='__main__':main()
