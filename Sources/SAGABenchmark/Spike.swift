import Foundation
import SAGATypes
import SAGACore

// Spike: how much of the per-command CPU cost belongs to the BufferCommand representation itself?
// Every probe walks the same 1000 dispatchCompute commands and reports nanoseconds per command.

nonisolated(unsafe) var spikeSink = 0

/// The compact record the redesign would use: trivially copyable, no reference counting.
struct CompactRecord {
    var kind: UInt8
    var pipeline: UInt64
    var bindingStart: UInt32
    var bindingCount: UInt32
    var groupsX: UInt32, groupsY: UInt32, groupsZ: UInt32
}

@inline(never) func borrowedTouch(_ command: borrowing BufferCommand) -> Int {
    switch command {
    case .dispatchCompute(let pipeline, let bindings, let groups, _, _, let immediates, _):
        return Int(pipeline.id.serial) &+ bindings.count &+ Int(groups.x) &+ immediates.count
    default: return 0
    }
}

@inline(never) func copiedTouch(_ command: BufferCommand) -> Int {
    switch command {
    case .dispatchCompute(let pipeline, let bindings, let groups, _, _, let immediates, _):
        return Int(pipeline.id.serial) &+ bindings.count &+ Int(groups.x) &+ immediates.count
    default: return 0
    }
}

func compact(_ commands: [BufferCommand]) -> [CompactRecord] {
    var out: [CompactRecord] = []
    out.reserveCapacity(commands.count)
    for command in commands {
        if case .dispatchCompute(let pipeline, let bindings, let groups, _, _, _, _) = command {
            out.append(CompactRecord(kind: 1, pipeline: pipeline.id.serial, bindingStart: 0, bindingCount: UInt32(bindings.count),
                                     groupsX: groups.x, groupsY: groups.y, groupsZ: groups.z))
        }
    }
    return out
}

func runSpike(commands: [BufferCommand], deviceIdentity: UInt64) throws {
    let reps = 300
    let clock = ContinuousClock()
    func probe(_ name: String, _ body: () throws -> Int) rethrows {
        for _ in 0..<20 { spikeSink &+= try body() }
        let start = clock.now
        for _ in 0..<reps { spikeSink &+= try body() }
        let d = start.duration(to: clock.now)
        let ns = (Double(d.components.seconds) * 1e9 + Double(d.components.attoseconds) / 1e9) / Double(reps * commands.count)
        print("  \(name.padding(toLength: 46, withPad: " ", startingAt: 0)) \(String(format: "%7.1f", ns)) ns/command")
    }
    print("spike: \(commands.count) commands, \(reps) reps")
    probe("iterate only (count)") { var n = 0; for _ in commands { n &+= 1 }; return n }
    probe("pattern-match payload, copied param") { var n = 0; for c in commands { n &+= copiedTouch(c) }; return n }
    probe("pattern-match payload, borrowing param") { var n = 0; for c in commands { n &+= borrowedTouch(c) }; return n }
    probe("resources") { var n = 0; for c in commands { n &+= c.resources.count }; return n }
    probe("recordedPass") { var n = 0; for c in commands { n &+= c.recordedPass.command.resources.isEmpty ? 1 : 0 }; return n }
    probe("computeDispatch") { var n = 0; for c in commands { n &+= c.computeDispatch?.bindings.count ?? 0 }; return n }
    try probe("validate (one pass)") {
        for c in commands {
            try c.validate(device: deviceIdentity, descriptors: [:], textures: [:], samplerDescriptors: [:], textureViews: [:])
        }
        return 0
    }
    probe("convert to compact records (one pass)") { compact(commands).count }
    let records = compact(commands)
    probe("iterate compact records") {
        var n = 0
        for r in records { n &+= Int(r.pipeline) &+ Int(r.groupsX) &+ Int(r.bindingCount) }
        return n
    }
    print("sink \(spikeSink)")
}
