import Foundation
import SAGATypes
import SAGACore
import SAGACompiler
import SAGAVulkan
#if os(Windows)
import WinSDK
#endif

// Mirror of Benchmarks/wgpu-bench. Keep sizes, shaders and verification identical.
let aluElements = 1 << 22
let aluRounds = 256
let streamElements = 1 << 24
let roundtripWords = 4 << 20
let aluDispatches = 8
let streamDispatches = 8
let batchDispatches = 1000
let submitCount = 100
let compileVariants = 20
let warmup = 3

struct Result { var name: String; var samples: [Double]; var detail: String }

func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

func measure(_ samples: Int, _ body: () async throws -> Void) async throws -> [Double] {
    let clock = ContinuousClock()
    var out: [Double] = []
    for index in 0..<(warmup + samples) {
        let start = clock.now
        try await body()
        let elapsed = start.duration(to: clock.now)
        if index >= warmup {
            out.append(Double(elapsed.components.seconds) * 1e3 + Double(elapsed.components.attoseconds) / 1e15)
        }
    }
    return out
}

func fail(_ message: String) -> Never { fputs("FAILED: \(message)\n", stderr); exit(1) }

func rowGroups(_ elements: Int) -> DispatchWorkgroups {
    // Shaders index rows of 1024 workgroups x 256 invocations.
    DispatchWorkgroups(x: 1024, y: UInt32((elements + 262_143) / 262_144))
}

func format(_ value: Double) -> String {
    let rounded = (value * 1000).rounded() / 1000
    return "\(rounded)"
}

func pad(_ text: String, _ width: Int) -> String {
    text.count >= width ? text : String(repeating: " ", count: width - text.count) + text
}

@main struct Benchmark {
    static func main() async {
        do { try await run() } catch { fail("\(error)") }
    }

    static func run() async throws {
        var only: Set<String> = []
        var samples = 15
        var shaderDirectory = "Benchmarks/shaders"
        var args = Array(CommandLine.arguments.dropFirst())[...]
        while let arg = args.popFirst() {
            switch arg {
            case "--only": if let v = args.popFirst() { only.insert(v) }
            case "--samples": if let v = args.popFirst(), let n = Int(v) { samples = n }
            case "--timer-ms":
                // Windows sleeps in ~15.6 ms ticks by default, which dominates SAGA's fence polling.
                #if os(Windows)
                if let v = args.popFirst(), let n = UInt32(v) { timeBeginPeriod(n) }
                #endif
            case "--shaders": if let v = args.popFirst() { shaderDirectory = v }
            default: fail("unknown argument \(arg)")
            }
        }
        func shader(_ name: String) throws -> String {
            try String(contentsOfFile: "\(shaderDirectory)/\(name).wgsl", encoding: .utf8)
        }
        func wants(_ name: String) -> Bool { only.isEmpty || only.contains(name) }

        guard let adapter = VulkanBackend.requestAdapter() else { fail("no Vulkan adapter") }
        print("SAGA (Vulkan) on \(adapter.name)")
        let backend = try VulkanBackend.makeDevice(adapter: adapter)
        let device = try await Device.make(backend: backend)
        let compiler = IndependentShaderCompiler()
        var results: [Result] = []

        func pipeline(_ source: String) async throws -> ComputePipeline {
            try await device.createComputePipeline(try compiler.compileCompute(.wgsl(source), to: .spirv))
        }
        func buffer(_ words: Int, _ usage: BufferUsage) async throws -> ResourceID {
            try await device.createBuffer(.init(size: UInt64(words * 4), usage: usage))
        }
        func slot(_ i: UInt32) -> ComputeBindingSlot { ComputeBindingSlot(binding: i) }

        if wants("compile") {
            let template = try shader("alu")
            var variant = 0
            let s = try await measure(samples) {
                for _ in 0..<compileVariants {
                    variant += 1
                    _ = try await pipeline(template.replacingOccurrences(of: "__ITERS__", with: "\(1000 + variant)u"))
                }
            }
            results.append(.init(name: "compile", samples: s.map { $0 / Double(compileVariants) }, detail: "ms per pipeline"))
        }

        if wants("dispatch-batch") || wants("dispatch-submit") || wants("dispatch-list") || wants("dispatch-record") {
            let empty = try await pipeline(try shader("empty"))
            if wants("dispatch-batch") {
                let commands = [BufferCommand](repeating: .dispatchCompute(pipeline: empty, bindings: [:], workgroups: .init(x: 1)), count: batchDispatches)
                let s = try await measure(samples) { _ = try await device.submitCommands(commands).wait() }
                results.append(.init(name: "dispatch-batch", samples: s, detail: "\(batchDispatches) dispatches/submit"))
            }
            if wants("dispatch-list") {
                // The same work through the compact command list, recorded once and submitted repeatedly.
                var list = CommandList()
                for _ in 0..<batchDispatches { list.dispatch(empty, workgroups: .init(x: 1)) }
                let s = try await measure(samples) { _ = try await device.submit(list).wait() }
                results.append(.init(name: "dispatch-list", samples: s, detail: "\(batchDispatches) dispatches, prerecorded list"))
            }
            if wants("dispatch-record") {
                // Recording is part of the sample, like encoding in wgpu.
                let s = try await measure(samples) {
                    var list = CommandList()
                    for _ in 0..<batchDispatches { list.dispatch(empty, workgroups: .init(x: 1)) }
                    _ = try await device.submit(list).wait()
                }
                results.append(.init(name: "dispatch-record", samples: s, detail: "\(batchDispatches) dispatches, record + submit"))
            }
            if wants("dispatch-submit") {
                let one = [BufferCommand.dispatchCompute(pipeline: empty, bindings: [:], workgroups: .init(x: 1))]
                let s = try await measure(samples) {
                    for _ in 0..<submitCount { _ = try await device.submitCommands(one).wait() }
                }
                results.append(.init(name: "dispatch-submit", samples: s.map { $0 / Double(submitCount) }, detail: "ms per submit+wait"))
            }
        }

        if wants("alu") {
            let source = try shader("alu").replacingOccurrences(of: "__ITERS__", with: "\(aluRounds)u")
            let p = try await pipeline(source)
            let out = try await buffer(aluElements, [.storage, .copySource])
            let readback = try await buffer(1024, [.copyDestination, .mapRead])
            let dispatch = BufferCommand.dispatchCompute(pipeline: p, bindings: [slot(0): out], workgroups: rowGroups(aluElements))
            let s = try await measure(samples) {
                _ = try await device.submitCommands([BufferCommand](repeating: dispatch, count: aluDispatches)).wait()
            }
            let r = try await device.submitCommands([dispatch, .copy(source: out, destination: readback, byteCount: 4096), .read(buffer: readback, byteCount: 4096)]).wait()
            for i in 0..<1024 {
                var x = UInt32(i)
                for _ in 0..<aluRounds { x = x &* 1664525 &+ 1013904223; x ^= x >> 13 }
                if r[0][i] != x { fail("alu mismatch at \(i)") }
            }
            let grounds = Double(aluElements * aluRounds * aluDispatches) / 1e9
            results.append(.init(name: "alu", samples: s, detail: "\(format(grounds / (median(s) / 1e3))) G rounds/s"))
        }

        if wants("stream") {
            let p = try await pipeline(try shader("stream"))
            let a = try await buffer(streamElements, [.storage, .copyDestination])
            let b = try await buffer(streamElements, [.storage, .copyDestination])
            let out = try await buffer(streamElements, [.storage, .copySource])
            let readback = try await buffer(1024, [.copyDestination, .mapRead])
            let chunk = 1 << 20
            for start in stride(from: 0, to: streamElements, by: chunk) {
                let n = min(chunk, streamElements - start)
                let upload = try await device.submitCommands([
                    .write(buffer: a, offset: UInt64(start * 4), words: (0..<n).map { UInt32(truncatingIfNeeded: start + $0) }),
                    .write(buffer: b, offset: UInt64(start * 4), words: (0..<n).map { UInt32(truncatingIfNeeded: (start + $0) &* 7) }),
                ])
                _ = try await upload.wait()
            }
            let dispatch = BufferCommand.dispatchCompute(pipeline: p, bindings: [slot(0): a, slot(1): b, slot(2): out], workgroups: rowGroups(streamElements))
            let s = try await measure(samples) {
                _ = try await device.submitCommands([BufferCommand](repeating: dispatch, count: streamDispatches)).wait()
            }
            let r = try await device.submitCommands([dispatch, .copy(source: out, destination: readback, byteCount: 4096), .read(buffer: readback, byteCount: 4096)]).wait()
            for i in 0..<1024 where r[0][i] != UInt32(i) &* 10 { fail("stream mismatch at \(i)") }
            let gb = Double(streamElements * 12 * streamDispatches) / 1e9
            results.append(.init(name: "stream", samples: s, detail: "\(format(gb / (median(s) / 1e3))) GB/s"))
        }

        if wants("roundtrip") {
            let upload = try await buffer(roundtripWords, [.copyDestination, .copySource])
            let readback = try await buffer(roundtripWords, [.copyDestination, .mapRead])
            let data = (0..<roundtripWords).map { UInt32(truncatingIfNeeded: $0) &* 2654435761 }
            let bytes = UInt64(roundtripWords * 4)
            var last: [[UInt32]] = []
            let s = try await measure(samples) {
                last = try await device.submitCommands([
                    .write(buffer: upload, offset: 0, words: data),
                    .copy(source: upload, destination: readback, byteCount: bytes),
                    .read(buffer: readback, byteCount: bytes),
                ]).wait()
            }
            if last.first != data { fail("roundtrip mismatch") }
            results.append(.init(name: "roundtrip", samples: s, detail: "\(format(Double(bytes * 2) / 1e9 / (median(s) / 1e3))) GB/s up+down"))
        }

        if only.contains("compile-split") {
            let template = try shader("alu")
            var variant = 5000
            let clock = ContinuousClock()
            func ms(_ d: Duration) -> Double { Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15 }
            var compileTimes: [Double] = [], createTimes: [Double] = []
            for _ in 0..<(warmup + samples * 4) {
                variant += 1
                let source = template.replacingOccurrences(of: "__ITERS__", with: "\(variant % 40 + 1000)u")
                let start = clock.now
                let artifact = try compiler.compileCompute(.wgsl(source), to: .spirv)
                let mid = clock.now
                _ = try await device.createComputePipeline(artifact)
                let end = clock.now
                compileTimes.append(ms(start.duration(to: mid))); createTimes.append(ms(mid.duration(to: end)))
            }
            results.append(.init(name: "compile-frontend", samples: Array(compileTimes.dropFirst(warmup)), detail: "WGSL -> SPIR-V"))
            results.append(.init(name: "compile-create", samples: Array(createTimes.dropFirst(warmup)), detail: "createComputePipeline"))
        }

        if only.contains("spike") {
            let empty = try await pipeline(try shader("empty"))
            let commands = [BufferCommand](repeating: .dispatchCompute(pipeline: empty, bindings: [:], workgroups: .init(x: 1)), count: batchDispatches)
            try runSpike(commands: commands, deviceIdentity: device.identity)
        }

        if only.contains("batch-phases") {
            let empty = try await pipeline(try shader("empty"))
            let commands = [BufferCommand](repeating: .dispatchCompute(pipeline: empty, bindings: [:], workgroups: .init(x: 1)), count: batchDispatches)
            let clock = ContinuousClock()
            var enqueue: [Double] = [], total: [Double] = []
            for _ in 0..<(warmup + samples) {
                let start = clock.now
                let ticket = try await device.submitCommands(commands)
                let mid = clock.now
                _ = try await ticket.wait()
                let end = clock.now
                func ms(_ d: Duration) -> Double { Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15 }
                enqueue.append(ms(start.duration(to: mid))); total.append(ms(start.duration(to: end)))
            }
            results.append(.init(name: "batch-enqueue", samples: Array(enqueue.dropFirst(warmup)), detail: "Core validation + enqueue"))
            results.append(.init(name: "batch-total", samples: Array(total.dropFirst(warmup)), detail: "through completion"))
        }

        if only.contains("phases") {
            let upload = try await buffer(roundtripWords, [.copyDestination, .copySource])
            let readback = try await buffer(roundtripWords, [.copyDestination, .mapRead])
            let data = (0..<roundtripWords).map { UInt32(truncatingIfNeeded: $0) &* 2654435761 }
            let bytes = UInt64(roundtripWords * 4)
            let w = try await measure(samples) { _ = try await device.submitCommands([.write(buffer: upload, offset: 0, words: data)]).wait() }
            let c = try await measure(samples) { _ = try await device.submitCommands([.copy(source: upload, destination: readback, byteCount: bytes)]).wait() }
            let r = try await measure(samples) { _ = try await device.submitCommands([.read(buffer: readback, byteCount: bytes)]).wait() }
            results.append(.init(name: "phase-write", samples: w, detail: ""))
            results.append(.init(name: "phase-copy", samples: c, detail: ""))
            results.append(.init(name: "phase-read", samples: r, detail: ""))
        }

        print("\(pad("case", 16)) \(pad("median ms", 12)) \(pad("min ms", 12))  detail")
        for r in results {
            print("\(pad(r.name, 16)) \(pad(format(median(r.samples)), 12)) \(pad(format(r.samples.min() ?? 0), 12))  \(r.detail)")
        }
        await device.close()
    }
}
