import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    static func atomicDescriptorModule(runtime: Bool) -> [UInt32] {
        var words = head + i(17, 1)
        if runtime {
            words += i(17, 5302) // RuntimeDescriptorArray
            words += i(10, 0x5f565053, 0x5f545845, 0x63736564, 0x74706972, 0x695f726f, 0x7865646e, 0x00676e69)
        }
        words += i(14, 0, 1) + i(15, 5, 40, 0x6e69616d, 0) + i(16, 40, 17, 1, 1, 1)
        words += i(71, 3, 2) + i(72, 3, 0, 35, 0) + i(71, 15, 34, 0) + i(71, 15, 33, 0)
        words += i(19, 1) + i(21, 2, 32, 0) + i(30, 3, 2) + i(43, 2, 8, 2)
        words += runtime ? i(29, 4, 3) : i(28, 4, 3, 8)
        for instruction in [
            i(32, 5, 12, 4), i(32, 6, 12, 2), i(33, 7, 1),
            i(43, 2, 9, 0), i(43, 2, 10, 1), i(43, 2, 11, 7),
            i(43, 2, 13, 24), i(43, 2, 14, 29), i(59, 5, 15, 12),
            i(54, 1, 40, 0, 7), i(248, 41), i(65, 6, 42, 15, 10, 9),
            i(227, 2, 43, 42, 10, 9), i(228, 42, 10, 9, 43), // Load and store
            i(234, 2, 44, 42, 10, 9, 11), // Add 7
            i(230, 2, 45, 42, 10, 9, 9, 14, 13), // Replace 24 with 29
            i(253), i(56),
        ] { words += instruction }
        return words
    }
}

@Test(arguments: [false, true])
func spirvAtomicBufferDescriptorImportPreservesElementTypes(_ runtime: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.atomicDescriptorModule(runtime: runtime))
    let module = try compiler.parseAndValidate(input, capabilities: .allKnown).module
    let variable = try #require(module.globalVariables.values.first)
    guard case .bindingArray(let element, let size) = try module.types.value(for: variable.type).kind,
          case .structure(let members, _) = try module.types.value(for: element).kind else {
        Issue.record("Expected a storage descriptor array of structures")
        return
    }
    #expect(size == (runtime ? .dynamic : .constant(2)))
    let member = try #require(members.first)
    #expect(try module.types.value(for: member.type).kind == .atomic(.u32))
    let counts: [IRResourceBinding: UInt32] = runtime ? [.init(group: 0, binding: 0): 2] : [:]
    let shader = try compiler.compileCompute(input, to: .wgsl, bindingArrayLengths: counts)
    #expect(shader.storageBindings.first?.bindingArray?.count == 2)
    guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("atomicLoad("))
    #expect(source.contains("atomicStore("))
    #expect(source.contains("atomicCompareExchangeWeak("))
    _ = try compiler.parseAndValidate(.wgsl(source), capabilities: .allKnown)
}
