import Testing

/// Native GPU tests share one physical device. Running Vulkan and Direct3D 12 work from many tests at once makes
/// drivers reset or return stale results, so these tests take turns.
private actor GPUGate {
    static let shared = GPUGate()
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !busy { busy = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }
}

struct GPUExclusiveTrait: TestTrait, TestScoping {
    func provideScope(for test: Test, testCase: Test.Case?, performing function: @Sendable () async throws -> Void) async throws {
        await GPUGate.shared.acquire()
        do { try await function() } catch { await GPUGate.shared.release(); throw error }
        await GPUGate.shared.release()
    }
}

extension Trait where Self == GPUExclusiveTrait {
    static var gpuExclusive: Self { Self() }
}
