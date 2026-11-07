#!/usr/bin/env python3
"""Install only the pinned SDK, with the official swift.org release checksum."""
import json, pathlib, subprocess
pin=json.loads((pathlib.Path(__file__).parent/'swift-web-sdk.json').read_text())
version=subprocess.check_output(['swift','--version'],text=True)
if not __import__('re').search(r'\bversion 6\.3\.2\b',version): raise SystemExit('SDK requires the exact Swift 6.3.2 toolchain, not merely language mode 6')
sdk=pin['wasm_sdk']
subprocess.run(['swift','sdk','install',sdk['url'],'--checksum',sdk['checksum']],check=True)
