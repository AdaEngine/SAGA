import Testing
import SAGATypes
@testable import SAGAVulkan

// No development device reports VK_EXT_texture_compression_astc_hdr, so the ASTC HDR path cannot store real textures
// here. These tests pin everything the backend asserts about the extension to the values in vulkan_core.h.

@Test func vulkanMapsASTCHDRFormatsToTheirSFloatBlockValues() {
    // VK_FORMAT_ASTC_*_SFLOAT_BLOCK = 1000066000 + index, in block-size order (vulkan_core.h, Vulkan 1.4.313).
    let expected: [(TextureFormat, UInt32)] = [
        (.astc4x4HDR, 1_000_066_000), (.astc5x4HDR, 1_000_066_001), (.astc5x5HDR, 1_000_066_002), (.astc6x5HDR, 1_000_066_003),
        (.astc6x6HDR, 1_000_066_004), (.astc8x5HDR, 1_000_066_005), (.astc8x6HDR, 1_000_066_006), (.astc8x8HDR, 1_000_066_007),
        (.astc10x5HDR, 1_000_066_008), (.astc10x6HDR, 1_000_066_009), (.astc10x8HDR, 1_000_066_010), (.astc10x10HDR, 1_000_066_011),
        (.astc12x10HDR, 1_000_066_012), (.astc12x12HDR, 1_000_066_013),
    ]
    for (format, value) in expected { #expect(vulkanASTCHDRFormat(format) == value) }
    // Other compressed and uncompressed formats take the ordinary mapping paths.
    #expect(vulkanASTCHDRFormat(.astc4x4Unorm) == nil)
    #expect(vulkanASTCHDRFormat(.astc12x12UnormSrgb) == nil)
    #expect(vulkanASTCHDRFormat(.bc1RGBAUnorm) == nil)
    #expect(vulkanASTCHDRFormat(.rgba8Unorm) == nil)
}

@Test func vulkanLinksTheASTCHDRFeatureStructureIntoTheDeviceChain() {
    func chainTypes(_ chain: VulkanFeatureChain) -> [UInt32: UInt32] {
        // Walks the pNext chain and returns each structure type with the VkBool32 at offset 16.
        var found: [UInt32: UInt32] = [:]
        var node: UnsafeRawPointer? = UnsafeRawPointer(chain.features2)
        while let current = node {
            found[current.load(as: UInt32.self)] = current.load(fromByteOffset: 16, as: UInt32.self)
            node = current.load(fromByteOffset: 8, as: UnsafeRawPointer?.self)
        }
        return found
    }
    let hdrStructure: UInt32 = 1_000_066_000  // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES
    let chain = VulkanFeatureChain()
    chain.astcHDR = 1
    chain.relink(float16: false, cooperative: false, astcHDR: true)
    #expect(chainTypes(chain)[hdrStructure] == 1)  // present in the chain with the feature switched on
    chain.relink(float16: false, cooperative: false, astcHDR: false)
    #expect(chainTypes(chain)[hdrStructure] == nil)  // left out when the device does not enable it
    // Ray tracing structures that share the chain keep their own types and do not collide with it.
    chain.rayQuery = 1
    chain.relink(float16: false, cooperative: false, rayQuery: true, astcHDR: true)
    let types = chainTypes(chain)
    #expect(types[hdrStructure] == 1)
    #expect(types[1_000_348_013] == 1)  // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR
}

@Test func vulkanEnablesASTCHDRWhenTheExtensionFeatureAndLDRSupportAreAllPresent() {
    #expect(VulkanFeatureSupport.astcHDRExtension == "VK_EXT_texture_compression_astc_hdr")
    var support = VulkanFeatureSupport()
    #expect(!support.canEnableASTCHDR)
    support.astcHDR = true
    support.core[21] = 1  // textureCompressionASTC_LDR
    #expect(!support.canEnableASTCHDR)  // the extension is missing
    support.deviceExtensions.insert(VulkanFeatureSupport.astcHDRExtension)
    #expect(support.canEnableASTCHDR)
    support.core[21] = 0
    #expect(!support.canEnableASTCHDR)  // the HDR formats build on the LDR ones
}
