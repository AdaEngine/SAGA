# SAGA-15 in-progress durable handoff

Scope only SPIRV sources, SPIRVFrontendTests.swift, this evidence directory and spirv-frontend-coverage.json.
Current implementation: actual binary header/word decoding, sparse definitions, typed scalar/vector/matrix/array/struct/pointer declarations, constants, globals, function parameters/calls, load/store and arithmetic; structured selection and phi spilling.
Not full supported Naga frontend. IO wrapper/coordinate adjustment, images, atomics, extended instructions, generalized CFG/loops/switch, specialization and several decorations remain explicit unsupported pending continued implementation. Semantic validation remains parent-owned.
API `SPIRVFrontend:BoundedShaderFrontend`, Module=IRModule, init(options:SPIRVFrontendOptions=.init(),limits:ShaderLimits=.default). No shared edits.
Reference pin a55352911bb26a1b33bc8d5fc03bdebb0a967f5a verified clean. Numeric operand metadata from SPIRV-Headers vulkan-sdk-1.4.341.0 matching pinned Cargo.lock spirv0.4.0+sdk-1.4.341.0, archived metadata hash508e6677e27b0b58eac8395aa49a90ac1766a8d392c247a0aba7b03f4fdd85b9. No Rust compilation/dependency/runtime/FFI.
Independent binary fixtures authored in Swift; no reference shader code/expected output copied. Grammar copyright retained in grammar-reference.json; fetch matching upstream license before redistribution. Grammar metadata is read-only verification input, not runtime dependency or generated implementation.
First RED captured missing SPIRVFrontend API (also concurrent GLSL/WGSL in-progress compile failures). First strict build had no SPIRV file diagnostics, but was blocked on concurrent WGSL missing helper/rethrows errors. Do not modify sibling sources to force pass.
Verification pending. This report will be replaced by final results when ready; no completion claim.
