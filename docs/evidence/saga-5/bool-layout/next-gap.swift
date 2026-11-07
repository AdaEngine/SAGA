import SAGAShaderIR
import SAGAFrontends
@main struct NextGap {
    static func main() throws {
        let source = "const A = 16u; struct S { @align(A) x: f32, }"
        do { _ = try WGSLFrontend().parse(.wgsl(source)); print("accepted") }
        catch { print(error) }
    }
}
