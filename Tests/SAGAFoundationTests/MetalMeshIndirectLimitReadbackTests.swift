#if canImport(Metal)
import Metal
import Testing
@testable import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
func metalIndirectDispatchSanitizesStageLimitsWithoutOverflow() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let validator = try MetalDispatchValidator(device: device)
    // Small representative limits make boundary cases inexpensive and exercise the
    // same GPU sanitizer used for both task and mesh stage limits.
    let cases: [(SIMD3<UInt32>, UInt32, UInt32?, [UInt32])] = [
        (.init(4, 2, 1), 4, 8, [4, 2, 1]),
        (.init(5, 1, 1), 4, 8, [0, 0, 0]),
        (.init(1, 5, 1), 4, 8, [0, 0, 0]),
        (.init(1, 1, 5), 4, 8, [0, 0, 0]),
        (.init(4, 3, 1), 4, 8, [0, 0, 0]),
        (.init(.max, .max, .max), .max, .max, [0, 0, 0]),
        (.init(1, 1, 1), 4, 0, [0, 0, 0]),
        (.init(0, 1, 1), 4, 8, [0, 0, 0]),
        (.init(4, 3, 1), 4, nil, [4, 3, 1]),
    ]
    for (groups, axisLimit, totalLimit, expected) in cases {
        var words = [groups.x, groups.y, groups.z]
        let source = device.makeBuffer(bytes: &words, length: 12, options: .storageModeShared)!
        let output = device.makeBuffer(length: 12, options: .storageModeShared)!
        let commands = queue.makeCommandBuffer()!
        let sanitized = try validator.encode(on: commands, device: device, source: source, offset: 0,
            maxWorkgroupsPerDimension: axisLimit, maximumLinearWorkgroupCount: totalLimit)
        let blit = commands.makeBlitCommandEncoder()!
        blit.copy(from: sanitized, sourceOffset: 0, to: output, destinationOffset: 0, size: 12)
        blit.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        #expect(commands.status == .completed)
        let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: 3))
        #expect(actual == expected)
    }
}
#endif
