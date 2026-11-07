import SAGAShaderIR
import SAGAFrontends
@main struct Smoke {
    static func main() throws {
        for (s,stage) in [("#version 450\nlayout(local_size_x=4) in;void main(){uint x=1u;while(x<4u){x=x+1u;}if(x==4u){return;}}",ShaderStage.compute),
                          ("#version 450\nfloat twice(float a){return a*2.0;}void main(){gl_Position=vec4(twice(1.0));}",.vertex),
                          ("#version 460\n#define PLUS(a,b) ((a)+(b))\n#if defined(ENABLED) && ENABLED == 2\nvoid main(){int x=PLUS(1,2);}\n#else\n#error nope\n#endif",.compute)] {
            let m=try GLSLFrontend(options:.init(stage:IRStage(stage))).parse(.glsl(s,stage:stage,defines:["ENABLED":"2"]))
            print("Parsed \(stage) types=\(m.types.count) functions=\(m.functions.count)")
        }
        print("SMOKE three independent shader parsing/lowering calls completed (not official filtered test suite)")
    }
}
