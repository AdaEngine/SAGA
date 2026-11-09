import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import Testing

@Test func wgslDocumentationKeepsSourceCommentsOnResolvedDeclarations() throws {
    let input = ShaderInput.wgsl("""
        //! Module overview.
        /*! More about the module. */

        /// Entry point.
        @compute @workgroup_size(1)
        fn main() {
            /// A local comment.
            let value = helper();
            _ = value;
        }

        /// Helper.
        fn helper() -> u32 {
            let item = Item(1u, LIMIT, 0u);
            return item.middle + sharedValue;
        }

        /// Data layout.
        struct Item {
            first: u32,
            /** Unicode 📐 /* nested */ comment. */
            @align(8) middle: u32,
            /// Last member.
            last: u32,
        }

        /// Limit.
        /// A second line.
        const LIMIT = 8u;

        /// Shared value.
        var<private> sharedValue: u32;
        """)
    #expect(try WGSLFrontend().parse(input).documentation.isEmpty)
    let module = try WGSLFrontend(options: .init(parseDocComments: true)).parse(input)
    try ShaderSemanticValidator().validate(module)

    let item = try module.types.handle(at: #require(module.types.values.firstIndex { $0.name == "Item" }))
    let helper = try module.functions.handle(at: #require(module.functions.values.firstIndex { $0.name == "helper" }))
    let limit = try module.constants.handle(at: #require(module.constants.values.firstIndex { $0.name == "LIMIT" }))
    let global = try module.globalVariables.handle(at: #require(module.globalVariables.values.firstIndex { $0.name == "sharedValue" }))
    let expected: [IRDocumentation] = [
        .init(target: .module, lines: ["//! Module overview.", "/*! More about the module. */"]),
        .init(target: .entryPoint(0), lines: ["/// Entry point."]),
        .init(target: .function(helper), lines: ["/// Helper."]),
        .init(target: .type(item), lines: ["/// Data layout."]),
        .init(target: .member(type: item, index: 1), lines: ["/** Unicode 📐 /* nested */ comment. */"]),
        .init(target: .member(type: item, index: 2), lines: ["/// Last member."]),
        .init(target: .constant(limit), lines: ["/// Limit.", "/// A second line."]),
        .init(target: .globalVariable(global), lines: ["/// Shared value."]),
    ]
    #expect(module.documentation.count == expected.count)
    for record in expected { #expect(module.documentation.contains(record)) }
}

@Test func semanticValidationChecksDocumentationReferencesAndBudget() throws {
    let validator = ShaderSemanticValidator()
    let invalid: [IRDocumentationTarget] = [
        .entryPoint(0), .type(try .init(index: 0)), .member(type: try .init(index: 0), index: 0),
        .function(try .init(index: 0)), .constant(try .init(index: 0)), .globalVariable(try .init(index: 0)),
    ]
    for target in invalid {
        #expect(throws: ShaderFailure.self) {
            try validator.validate(.init(documentation: [.init(target: target, lines: ["/// Orphan."])]))
        }
    }
    let structure = try WGSLFrontend().parse(.wgsl("struct S { value: u32 }"))
    let handle = try structure.types.handle(at: #require(structure.types.values.firstIndex { $0.name == "S" }))
    #expect(throws: ShaderFailure.self) {
        try validator.validate(.init(types: structure.types,
            documentation: [.init(target: .member(type: handle, index: UInt32.max), lines: ["/// Out of range."])]))
    }
    let tiny = ShaderSemanticValidator(limits: try .init(sourceBytes: 4))
    #expect(throws: ShaderFailure.budgetExceeded(.sourceBytes)) {
        try tiny.validate(.init(documentation: [.init(target: .module, lines: ["//! 📐"])]))
    }
}
