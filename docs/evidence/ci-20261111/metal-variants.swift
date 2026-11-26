import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
print("Adapter: \(device.name)")
let directory = "docs/evidence/ci-20261111/"
let inputs: [(String, String, [UInt32], [UInt32])] = [
    ("access", "saga-path-repro.metal", [0, 10, 20, 30, 40], [0, 10, 20, 99, 40]),
    ("loop", "saga-loop-original.metal", [12345], [22]),
]

for (name, path, initial, expected) in inputs {
    let original = try String(contentsOfFile: directory + path, encoding: .utf8)
    let invert = original
        .replacingOccurrences(of: "if (saga_load_5 < 6u) {\n        } else {", with: "if (!(saga_load_5 < 6u)) {")
        .replacingOccurrences(of: "if (saga_load_4 < 2u) {\n            continue;\n        } else {\n            break;\n        }", with: "if (!(saga_load_4 < 2u)) { break; }")
    let variants = [
        ("base", original), ("size", original), ("invert", invert), ("msl32", original), ("msl40", original),
        ("unroll-disable", original.replacingOccurrences(of: "    while (true)", with: "    #pragma clang loop unroll(disable)\n    while (true)")),
        ("volatile-device", original.replacingOccurrences(of: "device uint*", with: "volatile device uint*")),
        ("coherent-device", original.replacingOccurrences(of: "device uint*", with: "coherent device uint*")),
        ("volatile-local", original.replacingOccurrences(of: "(uint|bool) saga_(local|first|loop_value)", with: "volatile $1 saga_$2", options: .regularExpression)),
        ("volatile-state", original.replacingOccurrences(of: "(uint|bool) saga_(first|loop_value)", with: "volatile $1 saga_$2", options: .regularExpression)),
        ("volatile-first", original.replacingOccurrences(of: "bool saga_first", with: "volatile bool saga_first")),
        ("optnone", original.replacingOccurrences(of: "kernel void", with: "[[clang::optnone]] kernel void")),
        ("bottom", bottomContinuing(original, isAccess: name == "access")),
        ("manual", manualEquivalent(isAccess: name == "access")),
    ]
    for (variant, source) in variants {
        do {
            let options = MTLCompileOptions()
            if variant == "size" { options.optimizationLevel = .size }
            if variant == "msl32" || variant == "coherent-device" { options.languageVersion = .version3_2 }
            if variant == "msl40" { options.languageVersion = .version4_0 }
            let library = try device.makeLibrary(source: source, options: options)
            let pso = try device.makeComputePipelineState(function: library.makeFunction(name: "saga_main")!)
            let buffer = device.makeBuffer(length: initial.count * 4, options: .storageModeShared)!
            let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: initial.count)
            for (index, word) in initial.enumerated() { ptr[index] = word }
            let commands = queue.makeCommandBuffer()!
            let encoder = commands.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            let one = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
            encoder.endEncoding()
            commands.commit()
            commands.waitUntilCompleted()
            let words = Array(UnsafeBufferPointer(start: ptr, count: initial.count))
            print("\(name) \(variant) \(words == expected ? "PASS" : "FAIL") \(words) \(String(describing: commands.error))")
        } catch {
            print("\(name) \(variant) ERROR \(error)")
        }
    }
}

func bottomContinuing(_ source: String, isAccess: Bool) -> String {
    guard let start = source.range(of: "    bool saga_first_"),
          let condition = source.range(of: "        if (!saga_first_"),
          let closing = source.range(of: "        }\n        saga_first_"),
          let after = source[closing.upperBound...].firstIndex(of: "\n"),
          let end = source[after...].range(of: "\n    }\n") else { return source }
    let continuing = source[source[condition.upperBound...].firstIndex(of: "\n")!..<closing.lowerBound]
    var body = String(source[source.index(after: after)..<end.lowerBound])
    if isAccess {
        body = body.replacingOccurrences(of: "if (saga_load_4 < 2u) {\n            continue;\n        } else {\n            break;\n        }",
                                         with: "if (!(saga_load_4 < 2u)) { break; }")
    } else {
        body = body.replacingOccurrences(of: "                continue;", with: String(continuing) + "\n                continue;")
        body = body.replacingOccurrences(of: "\n        continue;", with: "")
    }
    return source[..<start.lowerBound] + "    while (true) {\n" + body + continuing + source[end.lowerBound...]
}

func manualEquivalent(isAccess: Bool) -> String {
    let code = isAccess ? """
        uint index = 0u;
        for (uint i = 0; i <= 2; ++i) {
            index = buf[0];
            if (i < 2u) { buf[0] = index + 1u; }
        }
        buf[0] = 0u;
        buf[1 + (index >> 1u) * 2u + (index & 1u)] = 99u;
        """ : """
        uint value = 0u;
        for (uint i = 0; i < 6; ++i) {
            if (i == 1u) { continue; }
            if (i == 3u) { break; }
            value += i + 10u;
        }
        buf[0] = value;
        """
    return """
    #include <metal_stdlib>
    using namespace metal;
    kernel void saga_main(device uint* buf [[buffer(0)]]) {
    \(code)
    }
    """
}