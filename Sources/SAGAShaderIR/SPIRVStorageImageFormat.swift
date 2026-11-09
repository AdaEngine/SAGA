/// Explicit SPIR-V Image Format enumerants shared by the frontend and backend.
package enum SPIRVStorageImageFormat {
    package static let int64ImageCapability: UInt32 = 5016
    package static let int64ImageExtension = "SPV_EXT_shader_image_int64"

    private static let formats: [IRStorageFormat] = [
        .rgba32Float, // 1
        .rgba16Float, // 2
        .r32Float, // 3
        .rgba8Unorm, // 4
        .rgba8Snorm, // 5
        .rg32Float, // 6
        .rg16Float, // 7
        .rg11b10Ufloat, // 8
        .r16Float, // 9
        .rgba16Unorm, // 10
        .rgb10a2Unorm, // 11
        .rg16Unorm, // 12
        .rg8Unorm, // 13
        .r16Unorm, // 14
        .r8Unorm, // 15
        .rgba16Snorm, // 16
        .rg16Snorm, // 17
        .rg8Snorm, // 18
        .r16Snorm, // 19
        .r8Snorm, // 20
        .rgba32Sint, // 21
        .rgba16Sint, // 22
        .rgba8Sint, // 23
        .r32Sint, // 24
        .rg32Sint, // 25
        .rg16Sint, // 26
        .rg8Sint, // 27
        .r16Sint, // 28
        .r8Sint, // 29
        .rgba32Uint, // 30
        .rgba16Uint, // 31
        .rgba8Uint, // 32
        .r32Uint, // 33
        .rgb10a2Uint, // 34
        .rg32Uint, // 35
        .rg16Uint, // 36
        .rg8Uint, // 37
        .r16Uint, // 38
        .r8Uint, // 39
        .r64Uint, // 40
    ]
    private static let numbers = Dictionary(uniqueKeysWithValues: formats.enumerated().map {
        ($0.element, UInt32($0.offset + 1))
    })

    package static func decode(_ word: UInt32) -> IRStorageFormat? {
        guard let index = Int(exactly: word), index > 0, index <= formats.count else { return nil }
        return formats[index - 1]
    }

    package static func encode(_ format: IRStorageFormat) -> UInt32? {
        numbers[format]
    }

    package static func requiresExtendedFormats(_ word: UInt32) -> Bool {
        switch word {
        case 6...20, 25...29, 34...39: true
        default: false
        }
    }
}
