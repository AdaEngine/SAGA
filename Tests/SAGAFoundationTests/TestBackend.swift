import SAGATypes
import SAGABackend
import SAGAWebGPU

actor Gate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() {
        open = true
        let pending = waiters; waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

actor TestBackend: BrowserGPUHost {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    let allocationStarted = Gate()
    let submissionStarted = Gate()
    private let allocationGate: Gate?
    private let submissionGate: Gate?
    private let allocationError: GPUError?
    private let failFirstSubmission: Bool
    private let submissionError: GPUError?
    private var events: [String] = []
    private var allocated: Set<ResourceID> = []
    init(allocationGate: Gate? = nil, submissionGate: Gate? = nil,
         allocationError: GPUError? = nil, failFirstSubmission: Bool = false, submissionError: GPUError? = nil) {
        self.allocationGate = allocationGate; self.submissionGate = submissionGate
        self.allocationError = allocationError; self.failFirstSubmission = failFirstSubmission
        self.submissionError = submissionError
    }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        events.append("allocate:\(id.serial)")
        await allocationStarted.release()
        if let allocationGate { await allocationGate.wait() }
        if let allocationError { throw allocationError }
        allocated.insert(id)
    }
    func destroyBuffer(id: ResourceID) {
        events.append("destroy:\(id.serial)")
        allocated.remove(id)
    }
    func submit(_ submission: Submission) async throws {
        events.append("begin:\(submission.sequence)")
        await submissionStarted.release()
        if submission.sequence == 1, let submissionGate { await submissionGate.wait() }
        if let submissionError { throw submissionError }
        if submission.sequence == 1 && failFirstSubmission { throw GPUError.backendFailure("test failure") }
        events.append("end:\(submission.sequence)")
    }
    func snapshot() -> [String] { events }
    func allocationCount() -> Int { allocated.count }
}

actor TwoDestroyBackend: BrowserGPUHost {
    nonisolated let capabilities = try! DeviceCapabilities(maxBufferSize: 1024)
    let firstStarted = Gate()
    let secondStarted = Gate()
    let releaseFirst = Gate()
    let releaseSecond = Gate()
    private var calls = 0
    private var completed = 0
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func submit(_ submission: Submission) {}
    func destroyBuffer(id: ResourceID) async {
        calls += 1
        let call = calls
        if call == 1 { await firstStarted.release(); await releaseFirst.wait() }
        if call == 2 { await secondStarted.release(); await releaseSecond.wait() }
        completed += 1
    }
    func completedCount() -> Int { completed }
}
