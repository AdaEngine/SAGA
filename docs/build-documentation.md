# Building API documentation

Use Xcode with Swift 6.2 or newer. Open the package in Xcode, select the
`SAGA` scheme, and choose **Product > Build Documentation**.

Run these commands from the repository root to build and open the same catalog.

```sh
xcodebuild docbuild -scheme SAGA -destination 'generic/platform=macOS' \
  -derivedDataPath .build-docc/xcode
open .build-docc/xcode/Build/Products/Debug/SAGA.doccarchive
```

Build the `SAGACompiler`, `SAGAMetal`, and `SAGAWebGPU` schemes with the same
command to generate reference archives for those modules.

Swift Package Index reads the root `.spi.yml` to build documentation on macOS
once the package is indexed. `SAGA` is the landing module. The target list also
includes the core, descriptor, backend contract, and shader IR modules because
their types appear in public signatures. [Swift Package Index manifest guide](https://swiftpackageindex.com/swiftpackageindex/spimanifest/documentation/spimanifest/commonusecases)
