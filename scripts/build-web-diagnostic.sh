#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
swift=${SAGA_SWIFT:-swift}
scratch="$repo/.build-web-diagnostic"
sdk=swift-6.3.2-RELEASE_wasm

# Keep static data above the stack used by the debug compiler's bounded tree walks.
stack_bytes=4194304
"$swift" build --package-path "$repo" --product SAGAWebDiagnostic --scratch-path "$scratch" --swift-sdk "$sdk" \
    -Xswiftc -enable-experimental-feature -Xswiftc Extern \
    -Xswiftc -Xclang-linker -Xswiftc -mexec-model=reactor \
    -Xlinker -z -Xlinker "stack-size=$stack_bytes" -Xlinker --stack-first -Xlinker "--global-base=$stack_bytes" \
    -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
bin=$("$swift" build --package-path "$repo" --show-bin-path --scratch-path "$scratch" --swift-sdk "$sdk")
site="$scratch/site"
mkdir -p "$site"
cp "$bin/SAGAWebDiagnostic.wasm" "$site/"
cp "$repo/web/index.html" "$repo/web/reactor.mjs" "$repo/web/wasi.mjs" "$repo/web/webgpu.mjs" "$site/"
printf '%s\n' "$site"
