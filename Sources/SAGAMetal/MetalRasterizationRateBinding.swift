#if canImport(Metal)
import Metal
import SAGATypes

/// Binds a Metal rate map to a draw using one layer of its array attachments.
public final class MetalRasterizationRateBinding: RenderRasterizationRateMap {
    let map: any MTLRasterizationRateMap
    public let layer: UInt32
    public let layerCount: UInt32
    let physicalSize: MTLSize

    public init(map: any MTLRasterizationRateMap, layer: UInt32) throws {
        guard let count = UInt32(exactly: map.layerCount), layer < count else {
            throw GPUError.validation("The rasterization rate map has no such layer")
        }
        self.map = map
        self.layer = layer
        self.layerCount = count
        var width = 0
        var height = 0
        for index in 0..<map.layerCount {
            let size = map.physicalSize(layer: index)
            width = max(width, size.width)
            height = max(height, size.height)
        }
        physicalSize = MTLSize(width: width, height: height, depth: 1)
    }
}
#endif
