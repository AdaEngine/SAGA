/// Carries opcode packets to the JavaScript WebGPU host and returns its reply bytes.
public protocol BrowserTransport: Sendable {
    func send(_ bytes: [UInt8]) async throws -> [UInt8]
}

/// Largest request or reply, in bytes, that a transport must carry.
package let browserPacketLimit = 16 * 1024 * 1024
