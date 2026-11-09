/// Emulates cooperative tiles as contiguous row-major elements in a 32-lane wave.
let hlslCooperativeMatrixPrelude = """
struct saga_coop_tile { float4 lo; float4 hi; };
float saga_coop_element(saga_coop_tile tile, uint index, uint perLane) {
    uint owner = index / perLane;
    uint component = index % perLane;
    float4 lo = float4(WaveReadLaneAt(tile.lo.x, owner), WaveReadLaneAt(tile.lo.y, owner),
                       WaveReadLaneAt(tile.lo.z, owner), WaveReadLaneAt(tile.lo.w, owner));
    float4 hi = float4(WaveReadLaneAt(tile.hi.x, owner), WaveReadLaneAt(tile.hi.y, owner),
                       WaveReadLaneAt(tile.hi.z, owner), WaveReadLaneAt(tile.hi.w, owner));
    return component < 4u ? lo[component] : hi[component - 4u];
}
saga_coop_tile saga_coop_muladd(saga_coop_tile a, saga_coop_tile b, saga_coop_tile c,
                                uint rows, uint columns, uint inner) {
    uint count = rows * columns / 32u;
    uint aCount = rows * inner / 32u;
    uint bCount = inner * columns / 32u;
    saga_coop_tile result = c;
    for (uint i = 0u; i < count; ++i) {
        uint index = WaveGetLaneIndex() * count + i;
        float value = i < 4u ? c.lo[i] : c.hi[i - 4u];
        for (uint k = 0u; k < inner; ++k) {
            value += saga_coop_element(a, (index / columns) * inner + k, aCount) *
                     saga_coop_element(b, k * columns + index % columns, bCount);
        }
        if (i < 4u) result.lo[i] = value; else result.hi[i - 4u] = value;
    }
    return result;
}
"""
