#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
swift=${SAGA_SWIFT:-swift}
output="$repo/.build-metal-window"
product=SAGAMetalWindowDiagnostic
scratch="$output/build"
smoke=false
case "${1:-}" in
    --smoke) smoke=true ;;
    "") ;;
    *) printf 'Usage: %s [--smoke]\n' "$0" >&2; exit 2 ;;
esac

"$swift" build --package-path "$repo" --product "$product" --scratch-path "$scratch" \
    -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
bin=$("$swift" build --package-path "$repo" --show-bin-path --scratch-path "$scratch")
if [ "$smoke" = true ]; then
    output=$(mktemp -d "$output/smoke.XXXXXX")
fi
app="$output/$product.app"
mkdir -p "$app/Contents/MacOS"
cp "$bin/$product" "$app/Contents/MacOS/"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleExecutable</key><string>SAGAMetalWindowDiagnostic</string>
    <key>CFBundleIdentifier</key><string>team.saga.MetalWindowDiagnostic</string>
    <key>CFBundleName</key><string>SAGA Metal Window</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict></plist>
PLIST
if [ "$smoke" = true ]; then
    /usr/bin/open -n -W "$app" --args --smoke
    report="$output/saga-metal-window.log"
    cat "$report"
    grep -Fxq 'PRESENTED 96x96 rgba8Unorm' "$report"
    grep -Fxq 'PRESENTED 128x96 bgra8Unorm' "$report"
    grep -Fxq 'PASS: 2 window frames; pixel checks passed; resize passed' "$report"
else
    /usr/bin/open "$app"
fi
