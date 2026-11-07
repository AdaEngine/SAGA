import Foundation
import SAGATypes
import SAGACore
import SAGAMetal

@main struct MetalDiagnostic {
    static func report(_ message: String) {
        print(message)
        if Bundle.main.bundleURL.pathExtension == "app" {
            let url = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("saga-metal-diagnostic.log")
            if let data = (message + "\n").data(using: .utf8) {
                if FileManager.default.fileExists(atPath: url.path), let handle = try? FileHandle(forWritingTo: url) {
                    defer { try? handle.close() }
                    _ = try? handle.seekToEnd(); try? handle.write(contentsOf: data)
                } else { try? data.write(to: url) }
            }
        }
    }

    static func main() async {
        do { try await run() }
        catch {
            report("SAGA Metal backend diagnostic FAILED: \(error). No CPU fallback.")
            exit(1)
        }
    }

    static func run() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        var count = 4099
        var timeoutMS = 10000
        var noDevice = false
        var index = 0
        while index < args.count {
            switch args[index] {
            case "--no-device": noDevice = true
            case "--count", "--timeout-ms":
                let option = args[index]
                index += 1
                guard index < args.count, let value = Int(args[index]), value > 0 else {
                    throw GPUError.validation("\(option) needs a positive integer")
                }
                if option == "--count" { count = value } else { timeoutMS = value }
            default: throw GPUError.validation("Unknown argument \(args[index])")
            }
            index += 1
        }
        guard count <= 1_048_576, timeoutMS <= 60_000 else {
            throw GPUError.validation("Diagnostic limits: count <= 1048576, timeout-ms <= 60000")
        }
        let backend = try MetalBackend.makeDevice(adapter: noDevice ? nil : MetalBackend.requestAdapter(),
                                                   timeout: .milliseconds(timeoutMS))
        let device = try await Device.make(backend: backend)
        do {
            let pipeline = try await backend.makeDiagnosticPipeline(owner: device.identity)
            let size = UInt64(count * 4)
            let input = (0..<count).map { UInt32(truncatingIfNeeded: $0 &* 104729) ^ UInt32.max }
            let a = try await device.createBuffer(.init(label: "input", size: size, usage: [.storage, .copyDestination]))
            let b = try await device.createBuffer(.init(label: "output", size: size, usage: [.storage, .copySource]))
            let r = try await device.createBuffer(.init(label: "readback", size: size, usage: [.copyDestination, .mapRead]))
            let ticket = try await device.submitCommands([
                .write(buffer: a, offset: 0, words: input),
                .dispatch(pipeline: pipeline, input: a, output: b, count: UInt32(count)),
                .copy(source: b, destination: r, byteCount: size),
                .read(buffer: r, byteCount: size),
            ])
            let result = try await ticket.wait()
            guard result.count == 1, result[0].count == count else {
                throw GPUError.backendFailure("Readback length mismatch")
            }
            for i in 0..<count {
                let expected = input[i] &* 3 &+ 7
                guard result[0][i] == expected else {
                    throw GPUError.backendFailure("Element \(i): got \(result[0][i]), expected \(expected)")
                }
            }
            await device.close()
            report("PASS: adapter=\(backend.adapterName); elements=\(count); mismatches=0; GPU=Metal; CPU-fallback=false")
            report("Direct MSL backend diagnostic only. NOT WGSL/WebGPU/compiler conformance.")
        } catch { await device.close(); throw error }
    }
}
