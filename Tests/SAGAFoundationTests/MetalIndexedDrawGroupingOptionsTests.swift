@_spi(IndexedDrawGrouping) @testable import SAGAMetal
import Testing

@Test
func metalIndexedDrawGroupingCombinesAttachmentPermissions() {
    let color: MetalIndexedDrawGrouping = .color
    let withDepth: MetalIndexedDrawGrouping = [.color, .depthStencil]
    let withExtraColors: MetalIndexedDrawGrouping = [.color, .multipleColorAttachments]
    let withBoth = withDepth.union(withExtraColors)

    #expect(color.allows(depthStencil: false, multipleColorAttachments: false))
    #expect(!color.allows(depthStencil: true, multipleColorAttachments: false))
    #expect(withDepth.allows(depthStencil: true, multipleColorAttachments: false))
    #expect(!withDepth.allows(depthStencil: true, multipleColorAttachments: true))
    #expect(withExtraColors.allows(depthStencil: false, multipleColorAttachments: true))
    #expect(!withExtraColors.allows(depthStencil: true, multipleColorAttachments: true))
    #expect(withBoth.allows(depthStencil: true, multipleColorAttachments: true))
}
