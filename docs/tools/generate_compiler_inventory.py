#!/usr/bin/env python3
"""Deterministic lexical/reference-file denominator, NOT semantic feature completeness."""
import argparse, collections, hashlib, json, pathlib, re, subprocess
PIN='a55352911bb26a1b33bc8d5fc03bdebb0a967f5a'
ROOT=pathlib.Path(__file__).resolve().parents[2]

def generate(reference):
    reference=pathlib.Path(reference)
    assert subprocess.check_output(['git','-C',str(reference),'rev-parse','HEAD'],text=True).strip()==PIN
    assert subprocess.check_output(['git','--no-optional-locks','-C',str(reference),'status','--porcelain'],text=True)==''
    scope=json.loads((ROOT/'docs/naga-scope.json').read_text())
    files={}; items=[]; fixtures=[]
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    # Scope component anchors intentionally overlap shared source; IDs are component/path/symbol based.
    for component in scope['items']:
        base=reference/component['source']
        paths=sorted(base.rglob('*.rs')) if base.is_dir() else [base]
        # CLI owns Rust serialization declarations in the library and naga-types too.
        if component['id']=='cli-and-ir-interchange': paths+=sorted((reference/'naga-types/src').rglob('*.rs'))
        for p in paths:
            path=p.relative_to(reference).as_posix(); files[path]=digest(p)
            items.append({'id':component['id']+'/'+path,'component':component['id'],'stage':('frontend' if component['id'].startswith('front-') else 'backend' if component['id'].startswith('back-') else 'shared/tooling'),'kind':'source-unit','source':path,'line':1,'owner':component['owner'],'implementation':'not-started','verification':'not-run'})
            occurrence=collections.Counter()
            # Public enum/struct/type declarations, public named fields, constants (including bitflags).
            patterns=[('declaration',r'^\s*pub(?:\([^)]*\))?\s+(?:enum|struct|type)\s+(\w+)'),('field',r'^\s*pub\s+(\w+)\s*:'),('constant',r'^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+(\w+)\s*[:=]')]
            cli_attribute = False
            for line,text in enumerate(p.read_text().splitlines(),1):
                line_patterns = list(patterns)
                if path == 'naga-cli/src/bin/naga.rs':
                    if cli_attribute: line_patterns.append(('cli-argument',r'^\s*(\w+)\s*:'))
                    cli_attribute = '#[argh(' in text
                for kind,pattern in line_patterns:
                    match=re.search(pattern,text)
                    if match:
                        symbol=match[1]; occurrence[(kind,symbol)]+=1
                        items.append({'id':f"{component['id']}/{path}/{kind}/{symbol}/{occurrence[(kind,symbol)]}",'component':component['id'],'stage':('frontend' if component['id'].startswith('front-') else 'backend' if component['id'].startswith('back-') else 'shared/tooling'),'kind':kind,'symbol':symbol,'source':path,'line':line,'owner':component['owner'],'implementation':'not-started','verification':'not-run'})
    # Pin all compiler source/manifest inputs, including auxiliary IR/arena/types/serialization code.
    for folder in ['naga/src','naga-types/src','naga-cli/src']:
        for p in sorted((reference/folder).rglob('*.rs')): files[p.relative_to(reference).as_posix()]=digest(p)
    for name in ['naga/Cargo.toml','naga-cli/Cargo.toml','naga-types/Cargo.toml']: files[name]=digest(reference/name)
    for folder in ['naga/tests/in','naga/tests/out']:
        for p in sorted((reference/folder).rglob('*')):
            if p.is_file(): fixtures.append({'path':p.relative_to(reference).as_posix(),'sha256':digest(p),'role':'input' if folder.endswith('/in') else 'expected-output','implementation':'not-started','verification':'not-run','owner':'SAGA-5' if folder.endswith('/in') else 'SAGA-6'})
    return {'schema':1,'reference_revision':PIN,'denominator_kind':'lexical declarations, public fields, constants, source files and fixture files; not semantic assertions or expanded cases','limitations':['No parser-level exhaustive enum variant extraction or capability combination proof','Options can occur outside public fields/argh attributes and require manual semantic expansion','Fixture files are not expanded test cases; generated outputs are not executed','No shader implementation, translation or upstream binary compatibility follows from inventory'], 'components':[{k:v for k,v in c.items() if k in ['id','source','owner']} for c in scope['items']], 'source_hashes':dict(sorted(files.items())),'counts':{'items':len(items),'fixtures':len(fixtures),'sources':len(files),'by_component':dict(sorted(collections.Counter(i['component'] for i in items).items())),'by_kind':dict(sorted(collections.Counter(i['kind'] for i in items).items()))},'items':items,'fixtures':fixtures}
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('reference'); parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    output=ROOT/'docs/compiler-feature-inventory.json'; result=generate(args.reference)
    if args.check: assert json.loads(output.read_text())==result, 'inventory not reproducible'
    else: output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts'],indent=2))
