# Working on SAGA

SAGA is an independent Swift WebGPU-style runtime and shader compiler. The runtime paths are macOS/Metal and Swift/Wasm through an explicit browser WebGPU host. The compiler aims to cover the pinned Naga feature set in Swift.

- Use Swift 6.2 or newer in Swift 6 language mode. Match the strict concurrency and warnings-as-errors flags in [CI](.github/workflows/foundation.yml).
- Keep the SwiftPM graph dependency-free. Treat the sibling `../wgpu` checkout as read-only reference material.
- Verify compiler, native GPU, and browser behavior at their respective boundaries. Use [delivery slices](docs/saga-delivery-slices.md) for scope and keep implementation claims tied to evidence.
- For module boundaries, invariants, build and test recipes, and detailed feature notes, use the [SAGA development skill](.agents/skills/saga-development/SKILL.md).
- For API documentation, edit [the SAGA DocC catalog](Sources/SAGA/SAGA.docc/SAGA.md). [Documentation build instructions](docs/build-documentation.md)

<!-- CODEGRAPH_START -->
## CodeGraph

If `.codegraph/` exists at the repository root, use `codegraph_explore` or `codegraph explore "<symbol names or question>"` before text search or file reads for code exploration. Skip CodeGraph when the index is absent.
<!-- CODEGRAPH_END -->
