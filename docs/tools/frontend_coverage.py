#!/usr/bin/env python3
"""Fail-closed SAGA-5 inventory/evidence audit. Never a compiler parity oracle.

Reads pinned Git objects AND checks their worktree bytes. Generation writes only the
explicitly named coverage/provenance files; check is read-only. No Rust is executed.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

PIN = 'a55352911bb26a1b33bc8d5fc03bdebb0a967f5a'
ROOT = pathlib.Path(__file__).resolve().parents[2]
INVENTORY_SHA256 = '8bc5d084f42e6bc641a96bd6a67c808e3c9ca727a373653b5d1d249821f68084'
SCOPE_SHA256 = '0c6c5f3fb05333b5fed1521e14930a6b70e285159dc1a66def0286260aece5e8'
STATUSES = {'not-implemented', 'not-run', 'unsupported', 'tested'}
EVIDENCE = 'docs/evidence/saga-5/coverage/'
# Named semantic obligations are additional requirements, NOT a claim that a small
# hand-curated list exhausts WGSL/Naga rules. All reference source stays denominator.
REQUIREMENTS = {
    'stage': ('naga/src/valid/interface.rs', 'entrypoint stage, builtins, interface bindings, workgroup size and stage-restricted operations'),
    'type': ('naga/src/valid/type.rs', 'scalar widths, vectors/matrices, type equivalence, constructibility and recursive type rejection'),
    'uniformity': ('naga/src/valid/analyzer.rs', 'control-flow and value uniformity across calls, derivatives, barriers, subgroup operations'),
    'address-space': ('naga/src/valid/interface.rs', 'global/local address spaces, pointer access modes and host-shareability'),
    'capability': ('naga/src/valid/mod.rs', 'enabled/disabled capabilities, subgroup stage/operation sets and combinations'),
    'layout': ('naga/src/valid/type.rs', 'alignment, offsets, strides, runtime-sized arrays and uniform/storage layouts'),
    'expression': ('naga/src/valid/expression.rs', 'operand/result typing, literals, constructors, image and atomic operations'),
    'control-flow': ('naga/src/valid/function.rs', 'return paths, switch/loop/continuing, break/continue and reachable expression scopes'),
    'handles': ('naga/src/valid/handles.rs', 'invalid and forward arena handles and malformed raw IR without frontend filtering'),
    'constants-overrides': ('naga/src/valid/mod.rs', 'constant expressions, override IDs, initializer types and resolved/unresolved modes'),
    'bindings': ('naga/src/valid/interface.rs', 'duplicate resource bindings/locations, interpolation and builtin stage/type compatibility'),
    'source-diagnostics': ('naga/src/span.rs', 'typed diagnostics and exact UTF8 source ranges'),
    'binary-diagnostics': ('naga/src/front/spv/error.rs', 'typed SPIR-V diagnostics and exact binary word positions'),
    'wgsl-grammar': ('naga/src/front/wgsl/parse/mod.rs', 'pinned WGSL grammar, Unicode identifiers, number syntax, nested comments and attributes'),
    'glsl-preprocessing': ('naga/src/front/glsl/lex.rs', 'versions/profiles, object/function macros, conditional directives and expansion budgets'),
    'spirv-structure': ('naga/src/front/spv/mod.rs', 'header/version/endianness, word lengths, IDs, decorations, CFG dominance and phi lowering'),
    'malformed-budgets': ('naga/fuzz/fuzz_targets/wgsl_parser.rs', 'bounded source/word/token/node/depth/work and cancellation for all three frontends'),
    'capability-extension-matrix': ('naga/src/valid/mod.rs', 'reference-supported combinations, not arbitrary all-to-all; positive and negative configuration cases'),
}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(reference, *args, **kwargs):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(reference), *args], **kwargs)


def classify(path):
    suffix = pathlib.PurePosixPath(path).suffix
    fmt = {'.wgsl': 'wgsl', '.frag': 'glsl', '.vert': 'glsl', '.comp': 'glsl',
           '.spvasm': 'spirv-assembly', '.spv': 'spirv-binary', '.slang': 'slang',
           '.toml': 'configuration', '.rs': 'rust-test-source'}.get(suffix, 'auxiliary')
    if '/out/' in path:
        return 'output', fmt
    if suffix == '.toml':
        return 'config', fmt
    if suffix == '.rs':
        return 'inline-test-container', fmt
    if suffix == '.slang':
        return 'upstream-auxiliary', fmt  # NOT an invented Slang frontend
    if '/tests/naga/' in path and suffix == '.wgsl':
        return 'negative-input', fmt  # two deep-* fixtures, referenced by wgsl_errors
    return ('input' if '/in/' in path else 'auxiliary'), fmt


def struct_fields(text, name):
    match = re.search(r'pub struct ' + re.escape(name) + r'\b[^\{]*\{', text)
    if not match:
        return []
    # Source field inventory, not Rust parsing or semantic feature extraction.
    tail = text[match.end():]
    end = tail.find('\n}')
    return re.findall(r'^\s*pub\s+(\w+)\s*:', tail[:end], re.M)


def authoritative(reference, root=ROOT):
    """Derive membership from pinned tree + immutable inventory, NOT audited JSON."""
    reference, root = pathlib.Path(reference).resolve(), pathlib.Path(root)
    require(git(reference, 'rev-parse', 'HEAD').decode().strip() == PIN, 'reference HEAD drift')
    require(not git(reference, 'status', '--porcelain'), 'reference worktree is dirty')
    inventory_bytes = (root / 'docs/compiler-feature-inventory.json').read_bytes()
    scope_bytes = (root / 'docs/naga-scope.json').read_bytes()
    require(sha(inventory_bytes) == INVENTORY_SHA256, 'immutable inventory hash drift')
    require(sha(scope_bytes) == SCOPE_SHA256, 'immutable scope hash drift')
    inventory = json.loads(inventory_bytes)
    scope = json.loads(scope_bytes)
    require(inventory['reference_revision'] == scope['reference_revision'] == PIN, 'pin drift')
    require({x['id'] for x in scope['items'] if x['owner'] == 'SAGA-5'} ==
            {'front-wgsl', 'front-glsl', 'front-spv', 'validation', 'diagnostics'}, 'scope drift')
    tree = {}
    for entry in git(reference, 'ls-tree', '-rz', PIN).split(b'\0'):
        if not entry:
            continue
        info, path = entry.split(b'\t', 1)
        mode, kind, oid = info.decode().split()
        path = path.decode()
        if (path in inventory['source_hashes'] or path.startswith(('naga/tests/', 'naga/fuzz/'))
                or path in {'LICENSE.MIT', 'LICENSE.APACHE', 'naga/README.md', 'naga/Cargo.toml'}):
            require(kind == 'blob' and mode in {'100644', '100755'}, 'nonregular reference ' + path)
            tree[path] = oid
    # One bounded process for pinned blobs; Git object ids are not content SHA-256s.
    paths = sorted(tree)
    payload = ''.join(tree[p] + '\n' for p in paths).encode()
    raw = git(reference, 'cat-file', '--batch', input=payload)
    offset, blobs = 0, {}
    for path in paths:
        end = raw.index(b'\n', offset)
        header = raw[offset:end].split()
        require(len(header) == 3 and header[1] == b'blob', 'invalid Git object')
        size = int(header[2])
        data = raw[end + 1:end + 1 + size]
        offset = end + 2 + size
        require((reference / path).is_file() and not (reference / path).is_symlink(), 'reference path drift ' + path)
        require((reference / path).read_bytes() == data, 'reference hash drift ' + path)
        blobs[path] = data
    hashes = {p: sha(b) for p, b in blobs.items()}
    for path, digest in inventory['source_hashes'].items():
        require(hashes.get(path) == digest, 'reference inventory source hash drift ' + path)
    for fixture in inventory['fixtures']:
        require(hashes.get(fixture['path']) == fixture['sha256'], 'fixture hash drift ' + fixture['path'])
    records = []
    def add(identifier, kind, path, role, fmt='shared', **extra):
        require(path in hashes, 'missing reference ' + path)
        records.append(dict(id=identifier, kind=kind, source=path, sha256=hashes[path], role=role, format=fmt, **extra))
    # Pin all compiler sources too: frontend semantics depend on IR/proc/common/arena.
    for path in sorted(inventory['source_hashes']):
        add('source/' + path, 'source', path, 'source')
    for item in inventory['items']:
        if item['owner'] == 'SAGA-5':
            add('inventory/' + item['id'], 'lexical-feature', item['source'], 'lexical',
                {'front-wgsl': 'wgsl', 'front-glsl': 'glsl', 'front-spv': 'spirv-binary'}.get(item['component'], 'shared'),
                component=item['component'], line=item['line'], inventory_kind=item['kind'],
                symbol=item.get('symbol'))
    for path in paths:
        if path.startswith('naga/tests/'):
            role, fmt = classify(path)
            add('fixture/' + path, 'fixture', path, role, fmt)
        elif path.startswith('naga/fuzz/'):
            add('fuzz/' + path, 'fuzz-reference', path, 'source')
    # Named test anchors keep inline negative/validator cases visible. Functions,
    # macros, parameterized arrays and asserted subcases are NOT expanded semantic tests.
    for path, data in blobs.items():
        if path.endswith('.rs') and path.startswith(('naga/tests/', 'naga/src/front/', 'naga/src/valid/')):
            text = data.decode()
            for match in re.finditer(r'#\[test\](?:(?!\bfn\b).){0,1000}?\bfn\s+(\w+)\s*\(', text, re.S):
                name = match[1]
                line = text.count('\n', 0, match.start()) + 1
                add('test-anchor/' + path + '/' + name + ':' + str(line), 'inline-test-anchor', path,
                    'inline-case', line=line, symbol=name, expansion='not-expanded')
    for name, (path, description) in REQUIREMENTS.items():
        add('requirement/' + name, 'requirement', path, 'rule', requirement=name,
            description=description, polarities=['positive', 'negative'])
    valid_text = blobs['naga/src/valid/mod.rs'].decode()
    caps = []
    cap_block = valid_text.split('pub struct Capabilities:', 1)[1].split('\n}', 1)[0]
    for match in re.finditer(r'^\s*const (\w+)\s*=\s*([^;]+);', cap_block, re.M):
        caps.append({'name': match[1], 'expression': match[2].strip()})
        add('capability/' + match[1], 'requirement', 'naga/src/valid/mod.rs', 'rule',
            requirement='capability-' + match[1], polarities=['positive', 'negative'])
    # Error variant labels are rejection anchors, not expanded predicate proofs.
    for path, data in blobs.items():
        if path.startswith('naga/src/valid/') and path.endswith('.rs'):
            text = data.decode()
            for enum in re.finditer(r'^pub enum (\w*Error)\s*\{(.*?)^\}', text, re.M | re.S):
                for variant in re.finditer(r'^    ([A-Z]\w*)\b(?:\s*[(,{=]|\s*$)', enum[2], re.M):
                    symbol = enum[1] + '::' + variant[1]
                    line = text.count('\n', 0, enum.start(2) + variant.start()) + 1
                    add('validation-anchor/' + path + '/' + symbol, 'validation-error-anchor',
                        path, 'lexical', line=line, symbol=symbol,
                        expansion='trigger predicates and subcases not expanded')
    options = {}
    for fmt, path in [('wgsl', 'naga/src/front/wgsl/parse/mod.rs'),
                      ('glsl', 'naga/src/front/glsl/mod.rs'), ('spirv', 'naga/src/front/spv/mod.rs')]:
        options[fmt] = {'source': path, 'sha256': hashes[path],
                        'fields': struct_fields(blobs[path].decode(), 'Options')}
    metadata = {
        'options': options, 'capabilities': caps,
        'subgroup_operations': re.findall(r'^\s*const (\w+)\s*=', valid_text.split('pub struct SubgroupOperationSet:', 1)[1].split('\n}', 1)[0], re.M),
        'stage_flags': re.findall(r'^\s*const (\w+)\s*=', valid_text.split('pub struct ShaderStages:', 1)[1].split('\n}', 1)[0], re.M),
        'validation_flags': ['EXPRESSIONS', 'BLOCKS', 'CONTROL_FLOW_UNIFORMITY', 'STRUCT_LAYOUTS', 'CONSTANTS', 'BINDINGS'],
        'configuration_policy': 'Each concrete execution must name stage/options/capabilities. All combinations are NOT inferred from this inventory.',
        'glsl_versions': {'440': 'partial reference support', '450': 'reference', '460': 'reference'},
        'spirv_assembly': 'Input .spvasm requires independently verified assembly tool and resulting binary hash; assembly text is not binary frontend execution.',
        'inline_cases': 'Lexical #[test] anchors only; macro/parameter expansion and positive/negative subassertions need manual mapping before parity.',
        'semantic_denominator': 'Named obligations PLUS all pinned source/feature/fixture entries; not an exhaustive expanded normative assertion inventory.',
        'licenses': {p: hashes[p] for p in ['LICENSE.MIT', 'LICENSE.APACHE']},
        'wgsl_spec': {'revision': '4e829b310ba7381034b93b974eb1a6aabfb464f7',
                      'source': 'docs/sources/wgsl-spec.bs',
                      'sha256': sha((root / 'docs/sources/wgsl-spec.bs').read_bytes())},
        'fixture_configurations': [p for p in paths if p.startswith('naga/tests/') and p.endswith('.toml')],
    }
    fingerprints = {'inventory': sha(inventory_bytes), 'scope': sha(scope_bytes),
                    'provenance': sha((root / 'docs/provenance.md').read_bytes()),
                    'full_naga_adr': sha((root / 'docs/adr/port-full-naga-on-swift.md').read_bytes())}
    return dict(schema=1, reference_revision=PIN, input_fingerprints=fingerprints,
                metadata=metadata, records=sorted(records, key=lambda x: x['id']))


def initial_documents(expected):
    manifest = copy.deepcopy(expected)
    manifest['compiler_parity'] = False
    for record in manifest['records']:
        record.update(status='not-implemented' if record['kind'] in {'requirement', 'lexical-feature'} else 'not-run',
                      result_files=[], reason='No compatible native frontend execution imported by this harness.')
    provenance = dict(schema=1, reference_revision=PIN, copied_shader_count=0, fixtures=[])
    for record in expected['records']:
        if record['kind'] == 'fixture':
            provenance['fixtures'].append(dict(
                id=record['id'], source=record['source'], sha256=record['sha256'],
                revision=PIN, role=record['role'], copied_to=[], modifications='none; metadata only',
                license_status='not-audited-for-copy',
                attribution='Upstream path/revision/hash only. Per-file copyright, license exceptions and notices must be reviewed before copying; root notices alone are not clearance.'))
    return manifest, provenance


def local_file(root, relative, prefix):
    require(isinstance(relative, str) and relative.startswith(prefix), 'invalid evidence path')
    parts = pathlib.PurePosixPath(relative)
    require(not parts.is_absolute() and '..' not in parts.parts, 'invalid evidence path')
    root = pathlib.Path(root).resolve()
    path = root / relative
    require(path.resolve().is_relative_to(root), 'escaping evidence path')
    current = path
    while current != root:
        require(not current.is_symlink(), 'symlink evidence path')
        current = current.parent
    require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024, 'missing/oversize evidence path')
    return path


def check_result(root, relative, record, metadata):
    path = local_file(root, relative, EVIDENCE)
    try:
        result = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise AuditError('invalid result JSON: ' + str(error)) from error
    require(result.get('schema') == 1 and result.get('reference_revision') == PIN, 'result pin/schema')
    require(result.get('record_id') == record['id'] and result.get('reference_sha256') == record['sha256'], 'result identity')
    require(result.get('format') == record['format'], 'result format mismatch')
    require(result.get('stage') in {'vertex', 'fragment', 'compute', 'mesh', 'task', 'ray-generation', 'any-hit', 'closest-hit', 'miss', 'shared'}, 'result stage missing/invalid')
    require(record['format'] != 'spirv-assembly', 'assembly is not binary execution; audited assembler/binary mapping still required')
    require(result.get('kind') == 'native-frontend-semantic', 'result kind must not be parser/demo/inventory')
    require(type(result.get('exit_code')) is int and result['exit_code'] == 0, 'result exit code')
    require(type(result.get('tests_executed')) is int and result['tests_executed'] > 0, 'result tests must be nonzero')
    command = result.get('command')
    require(isinstance(command, list) and len(command) >= 2 and
            pathlib.PurePosixPath(command[0]).name == 'swift' and command[1] == 'test', 'result command')
    require(result.get('expected') in {'accept', 'reject'} and result.get('actual') == result['expected'], 'result actual/expected mismatch')
    polarity = result.get('polarity')
    require(polarity in {'positive', 'negative'}, 'result polarity')
    require(result['expected'] == ('accept' if polarity == 'positive' else 'reject'), 'result polarity mismatch')
    if polarity == 'negative':
        diagnostic = result.get('diagnostic')
        require(isinstance(diagnostic, dict) and isinstance(diagnostic.get('code'), str) and diagnostic['code'], 'negative result needs typed diagnostic')
        require(diagnostic.get('location_unit') in {'utf8', 'word', 'ir-handle'}, 'negative diagnostic location unit')
        require(type(diagnostic.get('start')) is int and type(diagnostic.get('end')) is int and
                0 <= diagnostic['start'] <= diagnostic['end'], 'negative diagnostic range')
    if record['role'] == 'negative-input':
        require(polarity == 'negative', 'negative-input result polarity')
    require(isinstance(result.get('options'), dict) and isinstance(result.get('capabilities'), list), 'result options/capabilities')
    known_caps = {x['name'] if isinstance(x, dict) else x for x in metadata['capabilities']}
    require(all(isinstance(c, str) and c in known_caps for c in result['capabilities']), 'result unknown capabilities')
    fmt = 'spirv' if record['format'] == 'spirv-binary' else record['format']
    if fmt in metadata['options']:
        option_meta = metadata['options'][fmt]
        fields = option_meta['fields'] if isinstance(option_meta, dict) else option_meta
        require(set(result['options']) == set(fields), 'result option fields missing/unknown; record defaults explicitly')
    require(isinstance(result.get('assertions'), list) and all(isinstance(x, str) and x.strip() for x in result['assertions'])
            and result['assertions'], 'result assertions')
    log = local_file(root, result.get('log'), EVIDENCE).read_bytes()
    require(sha(log) == result.get('log_sha256'), 'result log hash drift')
    require(isinstance(result.get('case'), str) and result['case'] and result['case'] in log.decode(errors='replace'), 'result log missing case')
    log_text = log.decode(errors='replace')
    require(re.search(re.escape(result['case']) + r'[^\n]*\bpassed\b', log_text), 'result log no passing named case')
    observed_counts = [int(n) for n in re.findall(r'\b(\d+) tests? passed\b', log_text)]
    require(result['tests_executed'] in observed_counts and result['tests_executed'] > 0, 'result tests count not observed in log')
    sources = result.get('implementation_hashes')
    require(isinstance(sources, dict) and sources, 'result implementation hashes missing')
    for source, digest in sources.items():
        require(source.startswith(('Sources/SAGAFrontends/', 'Sources/SAGAValidation/', 'Sources/SAGAShaderIR/'))
                and source.endswith('.swift'), 'result implementation path')
        require(sha(local_file(root, source, 'Sources/').read_bytes()) == digest, 'result implementation hash drift')
    return polarity


def audit_documents(manifest, provenance, expected, root=ROOT):
    require(isinstance(manifest, dict), 'manifest object required')
    require(set(manifest) == set(expected) | {'compiler_parity'}, 'unknown/missing manifest fields')
    require(manifest['compiler_parity'] is False, 'inventory/demo cannot claim compiler parity')
    for key in expected.keys() - {'records'}:
        require(manifest.get(key) == expected[key], 'manifest header drift: ' + key)
    rows = manifest.get('records')
    require(isinstance(rows, list), 'denominator records must be list')
    require(all(isinstance(x, dict) and isinstance(x.get('id'), str) for x in rows), 'denominator invalid record')
    require(len(rows) == len({r['id'] for r in rows}), 'duplicate denominator id')
    by_id = {x['id']: x for x in rows}
    require(set(by_id) == {x['id'] for x in expected['records']}, 'dropped/extra denominator')
    expected_prov = initial_documents(expected)[1]
    require(provenance == expected_prov, 'provenance drift/copy not audited: no copying authorized by metadata')
    counts = collections.Counter()
    for reference in expected['records']:
        record = by_id[reference['id']]
        require(set(record) == set(reference) | {'status', 'result_files', 'reason'}, 'record fields drift')
        require({k: record[k] for k in reference} == reference, 'record metadata drift ' + reference['id'])
        status = record['status']
        require(isinstance(status, str) and status in STATUSES, 'unknown status')
        require(isinstance(record['result_files'], list), 'invalid result list')
        require(isinstance(record['reason'], str) and (status == 'tested' or record['reason'].strip()), 'unsupported/unexecuted reason required')
        if status != 'tested':
            require(not record['result_files'], 'unexecuted record cannot have results')
        else:
            require(record['role'] in {'input', 'negative-input', 'rule', 'inline-case', 'lexical'}, 'non-executable config/output/source cannot count as shader tests')
            require(record['result_files'] and len(set(record['result_files'])) == len(record['result_files']), 'tested record needs actual unique result files')
            polarities = {check_result(root, p, record, expected['metadata']) for p in record['result_files']}
            if record['kind'] == 'requirement':
                require(polarities == {'positive', 'negative'}, 'rule needs positive and negative polarities')
        counts[status] += 1
    return dict(audit='pass', compiler_parity=False, records=len(rows), tested=counts['tested'],
                statuses=dict(sorted(counts.items())),
                roles=dict(sorted(collections.Counter(x['role'] for x in rows).items())),
                kinds=dict(sorted(collections.Counter(x['kind'] for x in rows).items())),
                reference_revision=PIN,
                limitation='Coverage audit success is not frontend/compiler execution or exhaustive semantic parity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=os.environ.get('SAGA_REFERENCE', str(ROOT.parent / 'wgpu')))
    parser.add_argument('--generate', action='store_true', help='Create initial unimplemented manifests; refuses overwrite')
    args = parser.parse_args()
    expected = authoritative(args.reference)
    paths = [ROOT / 'docs/frontend-coverage.json', ROOT / 'docs/frontend-fixture-provenance.json']
    if args.generate:
        require(not any(p.exists() for p in paths), 'refuse to overwrite manifests/results')
        for path, document in zip(paths, initial_documents(expected)):
            path.write_text(json.dumps(document, indent=2) + '\n')
    result = audit_documents(*(json.loads(p.read_text()) for p in paths), expected, ROOT)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (AuditError, OSError, subprocess.CalledProcessError) as error:
        print('coverage audit failed: ' + str(error), file=sys.stderr)
        sys.exit(1)
