#if os(Windows)
import WinSDK
import SAGATypes

// Direct3D 12 and DXC are used through their COM vtables, resolved at runtime from d3d12.dll and
// dxcompiler.dll. No SDK headers or import libraries are needed.

typealias HRESULT = Int32

struct D3DGUID {
    var data1: UInt32
    var data2: UInt16
    var data3: UInt16
    var data4: UInt64

    /// Parses the textual form `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
    init(_ text: String) {
        let hex = text.replacingOccurrences(of: "-", with: "")
        let bytes = stride(from: 0, to: hex.count, by: 2).map { offset -> UInt8 in
            let start = hex.index(hex.startIndex, offsetBy: offset)
            return UInt8(hex[start..<hex.index(start, offsetBy: 2)], radix: 16) ?? 0
        }
        data1 = UInt32(bytes[0]) << 24 | UInt32(bytes[1]) << 16 | UInt32(bytes[2]) << 8 | UInt32(bytes[3])
        data2 = UInt16(bytes[4]) << 8 | UInt16(bytes[5])
        data3 = UInt16(bytes[6]) << 8 | UInt16(bytes[7])
        var tail: UInt64 = 0
        for index in 0..<8 { tail |= UInt64(bytes[8 + index]) << UInt64(index * 8) }
        data4 = tail
    }
}

private extension String {
    func replacingOccurrences(of target: String, with replacement: String) -> String {
        split(separator: Character(target), omittingEmptySubsequences: false).joined(separator: replacement)
    }
}

enum IID {
    static let device = D3DGUID("189819f1-1db6-4b57-be54-1821339b85f7")
    static let commandQueue = D3DGUID("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
    static let commandAllocator = D3DGUID("6102dee4-af59-4b09-b999-b44d73f09b24")
    static let commandList = D3DGUID("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455")
    static let resource = D3DGUID("696442be-a72e-4059-bc79-5b5c98040fad")
    static let fence = D3DGUID("0a753dcf-c4d8-4b91-adf6-be5a60d95a76")
    static let commandSignature = D3DGUID("c36a797c-ec80-4f0a-8985-a7b2475082d1")
    static let queryHeap = D3DGUID("0d9658ae-ed45-469e-a61d-970ec583cab4")
    static let descriptorHeap = D3DGUID("8efb471d-616c-4f49-90f7-127bb763fa51")
    static let rootSignature = D3DGUID("c54a6b66-72df-4ee8-8be5-a946a1429214")
    static let pipelineState = D3DGUID("765a30f3-f624-4c6f-a828-ace948622445")
    static let dxcCompiler = D3DGUID("228b4687-5a6a-4730-900c-9702b2203f54")
    static let dxcOperationResult = D3DGUID("cedb484a-d4e9-445a-b991-ca21ca157dc2")
    static let dxcCompilerClass = D3DGUID("73e22d93-e6ce-47f3-b5bf-f0664f39c1b0")
}

/// An owned COM interface pointer. Direct3D 12 objects are free-threaded.
final class COMObject: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer

    init(_ pointer: UnsafeMutableRawPointer) { self.pointer = pointer }

    deinit {
        typealias Release = @convention(c) (UnsafeMutableRawPointer) -> UInt32
        _ = slot(2, Release.self)(pointer)
    }

    func slot<T>(_ index: Int, _ type: T.Type) -> T {
        let table = pointer.load(as: UnsafePointer<UnsafeRawPointer?>.self)
        return unsafeBitCast(table[index]!, to: T.self)
    }
}

func d3dCheck(_ result: HRESULT, _ operation: String) throws {
    guard result < 0 else { return }
    if let device = D3D12DebugLayer.device { D3D12DebugLayer.drain(device: device) }
    switch UInt32(bitPattern: result) {
    case 0x887A_0005, 0x887A_0006, 0x887A_0007: throw GPUError.deviceLost("Direct3D 12 device removed during \(operation)")
    case 0x8007_000E: throw GPUError.outOfMemory
    default: throw GPUError.backendFailure("Direct3D 12 \(operation) failed with HRESULT 0x\(String(UInt32(bitPattern: result), radix: 16))")
    }
}

enum D3D12 {
    static let featureLevel12_0: UInt32 = 0xC000
    static let heapDefault: UInt32 = 1
    static let heapUpload: UInt32 = 2
    static let heapReadback: UInt32 = 3
    static let resourceDimensionBuffer: UInt32 = 1
    static let layoutRowMajor: UInt32 = 1
    static let flagAllowUnorderedAccess: UInt32 = 0x4
    static let stateCommon: UInt32 = 0
    static let stateConstantBuffer: UInt32 = 0x1
    static let stateUnorderedAccess: UInt32 = 0x8
    static let stateNonPixelShaderResource: UInt32 = 0x40
    static let stateIndirectArgument: UInt32 = 0x200
    static let stateIndexBuffer: UInt32 = 0x2
    static let stateVertexAndConstantBuffer: UInt32 = 0x1
    static let statePixelShaderResource: UInt32 = 0x80
    static let stateCopyDestination: UInt32 = 0x400
    static let stateCopySource: UInt32 = 0x800
    static let stateGenericRead: UInt32 = 0xAC3
    static let barrierTransition: UInt32 = 0
    static let barrierUAV: UInt32 = 2
    static let rootParameterConstants: UInt32 = 1
    static let rootParameterCBV: UInt32 = 2
    static let rootParameterSRV: UInt32 = 3
    static let rootParameterUAV: UInt32 = 4
    static let commandListDirect: UInt32 = 0
}

struct D3D12HeapProperties {
    var type: UInt32 = 0
    var cpuPageProperty: UInt32 = 0
    var memoryPoolPreference: UInt32 = 0
    var creationNodeMask: UInt32 = 1
    var visibleNodeMask: UInt32 = 1
}

struct D3D12ResourceDesc {
    var dimension: UInt32 = D3D12.resourceDimensionBuffer
    var alignment: UInt64 = 0
    var width: UInt64 = 0
    var height: UInt32 = 1
    var depthOrArraySize: UInt16 = 1
    var mipLevels: UInt16 = 1
    var format: UInt32 = 0
    var sampleCount: UInt32 = 1
    var sampleQuality: UInt32 = 0
    var layout: UInt32 = D3D12.layoutRowMajor
    var flags: UInt32 = 0
}

struct D3D12CommandQueueDesc {
    var type: UInt32 = 0
    var priority: Int32 = 0
    var flags: UInt32 = 0
    var nodeMask: UInt32 = 0
}

struct D3D12ResourceBarrier {
    var type: UInt32 = 0
    var flags: UInt32 = 0
    var resource: UnsafeMutableRawPointer?
    var subresource: UInt32 = 0
    var stateBefore: UInt32 = 0
    var stateAfter: UInt32 = 0
}

struct D3D12RootParameter {
    var parameterType: UInt32 = 0
    var unionAlignment: UInt32 = 0
    var shaderRegister: UInt32 = 0
    var registerSpace: UInt32 = 0
    var unionTail: UInt64 = 0
    var shaderVisibility: UInt32 = 0
}

struct D3D12RootSignatureDesc {
    var parameterCount: UInt32 = 0
    var parameters: UnsafeRawPointer?
    var staticSamplerCount: UInt32 = 0
    var staticSamplers: UnsafeRawPointer?
    var flags: UInt32 = 0
}

struct D3D12ComputePipelineStateDesc {
    var rootSignature: UnsafeMutableRawPointer?
    var shaderBytecode: UnsafeRawPointer?
    var shaderBytecodeSize: Int = 0
    var nodeMask: UInt32 = 0
    var cachedBlob: UnsafeRawPointer?
    var cachedBlobSize: Int = 0
    var flags: UInt32 = 0
}

struct D3D12Range {
    var begin: Int = 0
    var end: Int = 0
}

struct DxcBuffer {
    var pointer: UnsafeRawPointer?
    var size: Int = 0
    var encoding: UInt32 = 0
}

/// Entry points resolved from d3d12.dll.
struct D3D12Library: Sendable {
    let createDevice: @convention(c) (UnsafeMutableRawPointer?, UInt32, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
    let serializeRootSignature:
        @convention(c) (UnsafeRawPointer, UInt32, UnsafeMutablePointer<UnsafeMutableRawPointer?>, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> HRESULT

    static func load() -> D3D12Library? {
        guard let module = "d3d12.dll".withCString(encodedAs: UTF16.self, { LoadLibraryW($0) }),
              let create = GetProcAddress(module, "D3D12CreateDevice"),
              let serialize = GetProcAddress(module, "D3D12SerializeRootSignature") else { return nil }
        typealias Create = @convention(c) (UnsafeMutableRawPointer?, UInt32, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias Serialize = @convention(c) (UnsafeRawPointer, UInt32, UnsafeMutablePointer<UnsafeMutableRawPointer?>, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> HRESULT
        return D3D12Library(createDevice: unsafeBitCast(create, to: Create.self),
                            serializeRootSignature: unsafeBitCast(serialize, to: Serialize.self))
    }
}
#endif
