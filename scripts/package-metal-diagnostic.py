"""Package the already-built native diagnostic for macOS Play; no external dependencies."""
import pathlib, plistlib, shutil

root = pathlib.Path(__file__).resolve().parents[1]
binary = root / '.build-apple/debug/SAGAMetalDiagnostic'
app = root / '.build-play/SAGAMetalDiagnostic.app'
(app / 'Contents/MacOS').mkdir(parents=True, exist_ok=True)
shutil.copy2(binary, app / 'Contents/MacOS/SAGAMetalDiagnostic')
with (app / 'Contents/Info.plist').open('wb') as stream:
    plistlib.dump({
        'CFBundleExecutable': 'SAGAMetalDiagnostic',
        'CFBundleIdentifier': 'team.saga.MetalDiagnostic',
        'CFBundleName': 'SAGA Metal Diagnostic',
        'CFBundleVersion': '1',
        'CFBundleShortVersionString': '0.1',
        'CFBundlePackageType': 'APPL',
        'LSMinimumSystemVersion': '14.0',
        'LSBackgroundOnly': True,
    }, stream)
print(app)
print('Headless GPU diagnostic; report: .build-play/saga-metal-diagnostic.log')
