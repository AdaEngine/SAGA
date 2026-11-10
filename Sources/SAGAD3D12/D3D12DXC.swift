#if os(Windows)
import WinSDK
import SAGATypes

/// DirectX Shader Compiler, loaded at runtime. HLSL artifacts target shader model 6 and need dxcompiler.dll and
/// dxil.dll (which signs the container). They are searched in `directories`, then in the normal DLL search path.
final class D3D12ShaderCompiler: @unchecked Sendable {
    private let compiler: COMObject

    init(directories: [String]) throws {
        func load(_ path: String) -> HMODULE? { path.withCString(encodedAs: UTF16.self) { LoadLibraryW($0) } }
        var module: HMODULE?
        for directory in directories {
            // dxil.dll must already be loaded when dxcompiler.dll looks it up by name.
            _ = load(directory + "\\dxil.dll")
            if module == nil { module = load(directory + "\\dxcompiler.dll") }
        }
        if module == nil { module = load("dxil.dll") ?? load("dxcompiler.dll"); _ = load("dxil.dll") }
        if module == nil || GetProcAddress(module, "DxcCreateInstance") == nil { module = load("dxcompiler.dll") }
        guard let module, let create = GetProcAddress(module, "DxcCreateInstance") else {
            throw GPUError.unsupported("dxcompiler.dll was not found; Direct3D 12 needs the DirectX Shader Compiler")
        }
        typealias Create = @convention(c) (UnsafeRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var clsid = IID.dxcCompilerClass, iid = IID.dxcCompiler
        var object: UnsafeMutableRawPointer?
        try d3dCheck(unsafeBitCast(create, to: Create.self)(&clsid, &iid, &object), "DXC creation")
        guard let object else { throw GPUError.backendFailure("DXC returned no compiler") }
        compiler = COMObject(object)
    }

    /// Compiles HLSL to a DXIL container.
    func compile(_ source: String, entryPoint: String, profile: String) throws -> [UInt8] {
        typealias Compile = @convention(c) (
            UnsafeMutableRawPointer, UnsafeRawPointer, UnsafePointer<UnsafePointer<UInt16>?>?, UInt32,
            UnsafeMutableRawPointer?, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        let arguments = ["-E", entryPoint, "-T", profile, "-O3"]
        let wide = arguments.map { argument -> UnsafeMutablePointer<UInt16> in
            let units = Array(argument.utf16) + [0]
            let pointer = UnsafeMutablePointer<UInt16>.allocate(capacity: units.count)
            pointer.initialize(from: units, count: units.count)
            return pointer
        }
        defer { wide.forEach { $0.deallocate() } }
        let argumentPointers: [UnsafePointer<UInt16>?] = wide.map { UnsafePointer($0) }
        var iid = IID.dxcOperationResult
        var result: UnsafeMutableRawPointer?
        var bytes = Array(source.utf8)
        let status: HRESULT = bytes.withUnsafeMutableBytes { sourceBytes in
            var buffer = DxcBuffer(pointer: UnsafeRawPointer(sourceBytes.baseAddress), size: sourceBytes.count, encoding: 65001)
            return argumentPointers.withUnsafeBufferPointer { pointers in
                compiler.slot(3, Compile.self)(compiler.pointer, &buffer, pointers.baseAddress, UInt32(pointers.count), nil, &iid, &result)
            }
        }
        try d3dCheck(status, "shader compilation")
        guard let result else { throw GPUError.backendFailure("DXC returned no result") }
        let operation = COMObject(result)

        typealias GetStatus = @convention(c) (UnsafeMutableRawPointer, UnsafeMutablePointer<HRESULT>) -> HRESULT
        typealias GetBlob = @convention(c) (UnsafeMutableRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias BufferPointer = @convention(c) (UnsafeMutableRawPointer) -> UnsafeRawPointer?
        typealias BufferSize = @convention(c) (UnsafeMutableRawPointer) -> Int
        var compileStatus: HRESULT = 0
        _ = operation.slot(3, GetStatus.self)(operation.pointer, &compileStatus)
        if compileStatus < 0 {
            var errors: UnsafeMutableRawPointer?
            _ = operation.slot(5, GetBlob.self)(operation.pointer, &errors)
            var message = "HLSL compilation failed"
            if let errors {
                let blob = COMObject(errors)
                if let text = blob.slot(3, BufferPointer.self)(blob.pointer) {
                    let size = blob.slot(4, BufferSize.self)(blob.pointer)
                    message += ": " + String(decoding: UnsafeRawBufferPointer(start: text, count: size), as: UTF8.self)
                }
            }
            throw GPUError.backendFailure(message)
        }
        var object: UnsafeMutableRawPointer?
        try d3dCheck(operation.slot(4, GetBlob.self)(operation.pointer, &object), "shader result")
        guard let object else { throw GPUError.backendFailure("DXC produced no object") }
        let blob = COMObject(object)
        guard let pointer = blob.slot(3, BufferPointer.self)(blob.pointer) else {
            throw GPUError.backendFailure("DXC produced an empty object")
        }
        return [UInt8](UnsafeRawBufferPointer(start: pointer, count: blob.slot(4, BufferSize.self)(blob.pointer)))
    }
}
#endif
