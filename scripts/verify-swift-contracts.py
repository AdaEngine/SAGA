#!/usr/bin/env python3
"""Compile-time contract tests against built library modules (no link or GPU run)."""
import argparse
import pathlib
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--swiftc', default='swiftc')
parser.add_argument('--modules', required=True)
args = parser.parse_args()
modules = pathlib.Path(args.modules).resolve()
assert modules.is_dir(), f'Build library modules first: {modules}'
cases = [
    ('valid-value-ir', '''import SAGAShaderIR
import SAGAValidation
struct IR: ShaderIRModule { let schemaVersion: UInt32 = 1 }
struct Validator: IRValidator { func validate(_ module: IR) throws {} }
let valid = try ValidatedModule.checking(IR(), with: Validator())
''', None),
    ('cannot-forge-validated-witness', '''import SAGAShaderIR
import SAGAValidation
struct IR: ShaderIRModule { let schemaVersion: UInt32 = 1 }
let forged = ValidatedModule(IR())
''', 'inaccessible'),
    ('raw-ir-cannot-enter-codegen', '''import SAGAShaderIR
import SAGAValidation
import SAGABackends
struct IR: ShaderIRModule { let schemaVersion: UInt32 = 1 }
func bypass<G: ShaderCodeGenerator>(_ generator: G, _ raw: G.Module) throws {
    _ = try generator.generate(raw)
}
''', 'cannot convert'),
    ('non-sendable-ir-rejected', '''import SAGAShaderIR
struct BadIR: ShaderIRModule {
    let callback: () -> Void
    let schemaVersion: UInt32 = 1
}
''', 'non-sendable'),
    ('mutable-backend-rejected', '''import SAGATypes
import SAGABackend
final class BadBackend: BackendDevice {
    var state = 0
    let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {}
    func destroyBuffer(id: ResourceID) async {}
    func submit(_ submission: Submission) async throws {}
}
''', 'mutable'),
    ('device-actor-isolation-enforced', '''import SAGACore
func bypass(_ device: Device) { _ = device.takeUncapturedErrors() }
''', 'actor-isolated'),
]
with tempfile.TemporaryDirectory(prefix='saga-contracts-') as directory:
    for name, source, diagnostic in cases:
        file = pathlib.Path(directory) / f'{name}.swift'
        file.write_text(source)
        result = subprocess.run([args.swiftc, '-typecheck', '-swift-version', '6',
                                 '-strict-concurrency=complete', '-warnings-as-errors',
                                 '-I', str(modules), str(file)], capture_output=True, text=True)
        if diagnostic is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0 and diagnostic in result.stderr.lower(), (name, result.stderr)
        print(f'PASS {name}')
print(f'Compile-time contracts: {len(cases)} checks passed (1 positive, 5 expected rejections).')
