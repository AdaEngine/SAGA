"""Fail-closed foundation dependency/module audit. Not a malicious-code sandbox."""
import hashlib, json, pathlib, re, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT/'docs/runtime-policy.json').read_text())
WEB_FILES = {'web/index.html','web/reactor.mjs','web/wasi.mjs','web/webgpu.mjs',
             'scripts/test-web-diagnostic.mjs','scripts/test-web-error-scopes.mjs',
             'scripts/test-web-host-close.mjs','scripts/test-web-texture-layout.mjs',
                                                   'scripts/test-web-resource-limits.mjs',
                                                   'scripts/test-web-core-limits.mjs',
                                                   'scripts/test-web-sparse-vertex-slots.mjs',
                                                   'scripts/test-web-resource-limits-browser.mjs',
                                                   'scripts/test-web-immediate-limit.mjs',
                                                   'scripts/test-web-immediate-limit-browser.mjs',
             'scripts/test-web-empty-query-resolve.mjs', 'scripts/test-web-indexed-indirect-capacity.mjs',
             'scripts/test-web-volume-self-copy.mjs',
             'scripts/test-web-indexed-indirect-capacity-browser.mjs',
             'scripts/test-web-wgsl-concretization-browser.mjs',
             'scripts/test-web-sampler-address.mjs', 'scripts/test-web-concurrent-buffer.mjs',
             'scripts/test-web-sampler-lod-browser.mjs', 'scripts/test-web-map-release.mjs',
             'scripts/test-web-buffer-release-allocation.mjs',
             'scripts/test-web-resource-release-allocation.mjs',
             'scripts/test-web-texture-view-release-race.mjs',
             'scripts/test-web-external-texture-closed-submission.mjs',
             'scripts/test-web-external-texture-closed-submission-browser.mjs',
             'scripts/test-web-texture-view-release-race-browser.mjs',
             'scripts/test-web-present-release-race-browser.mjs',
             'scripts/test-web-submission-release-race-browser.mjs', 'scripts/test-web-submission-read-release.mjs',
             'scripts/test-web-submission-read-release-browser.mjs',
             'scripts/test-web-external-texture-release-race-browser.mjs',
             'scripts/test-web-map-unmap.mjs', 'scripts/test-web-map-unmap-browser.mjs'}
WEB_REACTOR = 'Sources/SAGAWebDiagnostic/WebReactor.swift'

def is_docc_markdown(path):
    parts = path.parts
    return (len(parts) >= 4 and parts[0] == 'Sources' and parts[1] in POLICY['graph']
            and parts[2] == parts[1] + '.docc' and path.suffix == '.md')

def unexpected_project_files(paths):
    def allowed(p):
        path=pathlib.PurePosixPath(p)
        if p in POLICY['standalone_benchmark_files'] or p in POLICY['archived_shader_artifacts']: return True
        if path.name in {'Cargo.toml','Cargo.lock','Package.resolved','package.json'} or path.suffix in {'.rs','.c','.h','.cpp','.m','.mm','.a','.dylib','.so','.wasm','.modulemap'}: return False
        if p in {'IDEA.md','README.md','AGENTS.md','Package.swift','.gitignore','.spi.yml'} | WEB_FILES: return True
        if p.startswith(('docs/','.sloppy/plans/')): return True
        if p.startswith('.github/workflows/'): return p.endswith(('.yml','.yaml'))
        if p.startswith('scripts/'): return p.endswith(('.py','.sh','.json'))
        parts=path.parts
        if len(parts) >= 4 and parts[:2] == ('.agents', 'skills'): return path.suffix == '.md'
        if is_docc_markdown(path): return True
        return len(parts)>=3 and parts[0] in {'Sources','Tests'} and parts[1] in POLICY['graph'] and p.endswith('.swift')
    return [p for p in paths if p and not allowed(p)]

def audit_package(package):
    errors=[]
    if package.get('toolsVersion',{}).get('_version') != POLICY['tools_version']+'.0': errors.append('tools version must be 6.2.0')
    if package.get('swiftLanguageVersions') != ['6']: errors.append('language mode must be 6')
    if package.get('dependencies'): errors.append('external dependencies forbidden')
    targets=package.get('targets',[])
    if {t['name'] for t in targets} != set(POLICY['graph']): errors.append('target graph names differ')
    for t in targets:
        name=t['name']; deps=[]
        for d in t.get('dependencies',[]):
            if set(d) != {'byName'} or len(d['byName']) != 2 or d['byName'][1] is not None: errors.append('nonlocal/conditional dependency '+name)
            else: deps.append(d['byName'][0])
        if deps != POLICY['graph'].get(name): errors.append('dependency edges differ '+name)
        if t.get('type') != POLICY['target_types'].get(name): errors.append('non-Swift target '+name)
        if t.get('settings', []) != POLICY.get('target_settings', {}).get(name, []): errors.append('unexpected target settings '+name)
        for k in ['resources','exclude','providers','pluginUsages']:
            if t.get(k): errors.append('unexpected target '+k+' '+name)
        for k in ['path','sources','publicHeadersPath','pkgConfig','url','checksum']:
            if t.get(k) is not None: errors.append('custom target '+k+' '+name)
    if {p['name']:p['targets'] for p in package.get('products',[])} != POLICY['products']: errors.append('products differ')
    for product in package.get('products',[]):
        if set(product['type']) != {POLICY['product_types'].get(product['name'])}: errors.append('product type differs '+product['name'])
    return errors

def audit_source(path, text):
    parts=pathlib.PurePosixPath(path).parts; module=parts[1]
    allowed=set(POLICY['graph'].get(module,[])) | set(POLICY['extra_imports'].get(module,[])) | set(POLICY.get('file_imports', {}).get(path, []))
    errors=[]
    # Import declarations, not prose such as 'No SDK headers or import libraries'.
    for match in re.finditer(r'(?:^|;)\s*(?:@\w+(?:\([^\n]*?\))?\s+)*import\s+(?:(?:struct|class|enum|protocol|func|var|let|typealias)\s+)?([A-Za-z_]\w*)',text, re.M):
        if match[1] not in allowed: errors.append(f'{path}: forbidden import {match[1]}')
    if path == WEB_REACTOR and text.startswith('#if os(WASI) && compiler(>=6.3)\n') and text.rstrip().endswith('#endif'):
        text=re.sub(r'^@_extern\(wasm, module: "saga_host", name: "(schedule|request|copyReply|finish)"\)$','',text,flags=re.M)
        text=re.sub(r'^@_expose\(wasm, "(saga_run_job|saga_reply|saga_start)"\)$','',text,flags=re.M)
    exception = POLICY.get('native_abi_exceptions', {}).get(path, {})
    exact_exception = (exception.get('sha256') == hashlib.sha256(text.encode()).hexdigest()
                       and bool(exception.get('reason')))
    for pattern in [r'@unchecked\s+Sendable',r'@_?(?:silgen_name|cdecl|extern|expose)\b',r'unsafeBitCast\b',r'\bdlopen\b',r'\bnonisolated\s*\(unsafe\)',r'@preconcurrency\b']:
        if re.search(pattern,text) and not (exact_exception and pattern in exception['patterns']):
            errors.append(f'{path}: unsafe escape {pattern}')
    return errors

def audit_manifest(manifest):
    # Exact platform-conditioned system links only; audit_package checks the evaluated settings too.
    for declaration in POLICY['manifest_linker_declarations']:
        if manifest.count(declaration) != 1:
            return ['missing or repeated reviewed system link '+declaration]
        manifest = manifest.replace(declaration, '', 1)
    if re.search(r'unsafeFlags|linkedLibrary|linkedFramework|binaryTarget|systemLibrary|plugin\s*\(|@_silgen_name',manifest):
        return ['unsafe manifest escape']
    return []

def audit(root=ROOT):
    package=json.loads(subprocess.check_output(['swift','package','--package-path',str(root),'dump-package'],text=True))
    errors=audit_package(package)
    paths=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z'],text=True).rstrip('\0').split('\0')
    errors += ['unexpected project file '+p for p in unexpected_project_files(paths)]
    for relative in WEB_FILES:
        file=root/relative
        if file.is_symlink() or not file.is_file():
            errors.append('missing or symlink browser source '+relative)
    # Walk the actual source tree too, so ignored foreign code cannot evade the git scan.
    for folder in ['Sources','Tests']:
        for p in (root/folder).rglob('*'):
            relative=p.relative_to(root).as_posix()
            if p.is_symlink():
                errors.append('symlink source '+relative)
                continue
            if p.is_dir(): continue
            # Finder metadata is never compiled by SwiftPM; do not treat it as a foreign implementation.
            if p.name == '.DS_Store': continue
            if is_docc_markdown(pathlib.PurePosixPath(relative)): continue
            if p.suffix!='.swift': errors.append('foreign or symlink source '+relative)
            else: errors += audit_source(relative,p.read_text())
    for relative, digest in POLICY['standalone_benchmark_hashes'].items():
        file = root/relative
        if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            errors.append('standalone benchmark hash drift '+relative)
    for relative, artifact in POLICY['archived_shader_artifacts'].items():
        file = root/relative
        if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != artifact['sha256']:
            errors.append('archived shader artifact hash drift '+relative)
    errors += audit_manifest((root/'Package.swift').read_text())
    if errors: raise AssertionError('\n'.join(errors))
    return {'status':'pass','targets':len(package['targets']),'external_dependencies':0,'scope':'foundation dependency/module policy, not GPU/compiler conformance'}
if __name__=='__main__': print(json.dumps(audit(),indent=2))
