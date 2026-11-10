#if os(Windows)
import WinSDK
import SAGATypes

// Optional Direct3D 12 debug layer. Set `SAGA_D3D12_DEBUG=1` to enable it and print its messages.

enum D3D12DebugLayer {
    nonisolated(unsafe) static var device: COMObject?
    static var enabled: Bool { environmentValue("SAGA_D3D12_DEBUG") != nil }

    /// Turns the debug layer on. Must run before the device is created.
    static func enableIfRequested() {
        guard enabled, let module = "d3d12.dll".withCString(encodedAs: UTF16.self, { LoadLibraryW($0) }),
              let address = GetProcAddress(module, "D3D12GetDebugInterface") else { return }
        typealias GetDebug = @convention(c) (UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var iid = D3DGUID("344488b7-6846-474b-b989-f027448245e0")
        var object: UnsafeMutableRawPointer?
        guard unsafeBitCast(address, to: GetDebug.self)(&iid, &object) >= 0, let object else { return }
        let debug = COMObject(object)
        typealias Enable = @convention(c) (UnsafeMutableRawPointer) -> Void
        debug.slot(3, Enable.self)(debug.pointer)
        print("D3D12 debug layer enabled")
    }

    /// Prints and clears the messages stored by the debug layer.
    static func drain(device: COMObject) {
        guard enabled else { return }
        typealias Query = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var iid = D3DGUID("0742a90b-c387-483f-b946-30a7e4e61458")
        var object: UnsafeMutableRawPointer?
        guard device.slot(0, Query.self)(device.pointer, &iid, &object) >= 0, let object else { return }
        let queue = COMObject(object)
        typealias Count = @convention(c) (UnsafeMutableRawPointer) -> UInt64
        typealias GetMessage = @convention(c) (UnsafeMutableRawPointer, UInt64, UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>) -> HRESULT
        typealias Clear = @convention(c) (UnsafeMutableRawPointer) -> Void
        let total = queue.slot(8, Count.self)(queue.pointer)
        for index in 0..<total {
            var size = 0
            _ = queue.slot(5, GetMessage.self)(queue.pointer, index, nil, &size)
            let memory = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 8)
            defer { memory.deallocate() }
            guard queue.slot(5, GetMessage.self)(queue.pointer, index, memory, &size) >= 0 else { continue }
            let severity = memory.load(fromByteOffset: 4, as: UInt32.self)
            if let text = memory.load(fromByteOffset: 16, as: UnsafePointer<CChar>?.self) {
                print("D3D12 debug (severity \(severity)): \(String(cString: text))")
            }
        }
        queue.slot(4, Clear.self)(queue.pointer)
    }
}

func environmentValue(_ name: String) -> String? {
    var buffer = [UInt16](repeating: 0, count: 4096)
    let length = name.withCString(encodedAs: UTF16.self) { GetEnvironmentVariableW($0, &buffer, UInt32(buffer.count)) }
    guard length > 0, Int(length) < buffer.count else { return nil }
    return String(decoding: buffer[0..<Int(length)], as: UTF16.self)
}
#endif
