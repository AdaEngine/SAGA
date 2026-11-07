import SAGAShaderIR

struct SPVBlockRecord {
    let label: SPVInstruction
    var instructions: [SPVInstruction] = []
    var phis: [SPVInstruction] = []
    var merge: UInt32?
    var continuing: UInt32?
    var terminator: SPVInstruction?
}
struct SPVValue { let expression: IRHandle<IRExpression>; let type: UInt32 }

struct SPVFunctionLowerer {
    let module: SPVDecoder
    let record: SPVFunctionRecord
    var budget: ShaderBudget
    var blocks: [UInt32: SPVBlockRecord] = [:]
    var order: [UInt32] = []
    var predecessors: [UInt32: Set<UInt32>] = [:]
    var parameters: [IRFunctionArgument] = []
    var expressions: [IRExpression] = []
    var statements: [IRStatement] = []
    var irBlocks: [IRBlock] = []
    var locals: [IRLocalVariable] = []
    var values: [UInt32: SPVValue] = [:]
    var phiPointers: [UInt32: IRHandle<IRExpression>] = [:]
    var visited: Set<UInt32> = []
    var resultType: UInt32 = 0
    var entryLabel: UInt32 = 0
    var deferredPhiReloads: [SPVInstruction] = []

    mutating func work(_ n:Int=1) throws { try budget.consume(.work,amount:n) }
    mutating func node() throws { try budget.consume(.nodes) }
    mutating func expr(_ t:IRHandle<IRType>,_ k:IRExpressionKind,_ i:SPVInstruction) throws -> IRHandle<IRExpression> {
        try node();try work();let h=try IRHandle<IRExpression>(index:expressions.count);expressions.append(.init(type:t,kind:k,location:i.location));return h
    }
    mutating func stmt(_ k:IRStatementKind,_ i:SPVInstruction) throws -> IRHandle<IRStatement> {
        try node();try work();let h=try IRHandle<IRStatement>(index:statements.count);statements.append(.init(kind:k,location:i.location));return h
    }
    mutating func block(_ ss:[IRHandle<IRStatement>],_ i:SPVInstruction) throws -> IRHandle<IRBlock> {
        try node();let h=try IRHandle<IRBlock>(index:irBlocks.count);irBlocks.append(.init(statements:ss,location:i.location));return h
    }
    mutating func value(_ id:UInt32,_ i:SPVInstruction) throws -> SPVValue {
        try work()
        if let v=values[id] { return v }
        if let c=module.constantMap[id] {
            let e=try expr(module.type(c.type,i),.constant(c.handle),i);let v=SPVValue(expression:e,type:c.type);values[id]=v;return v
        }
        if let (g,t)=module.globalMap[id] {
            let e=try expr(module.type(t,i),.globalVariable(g),i);let v=SPVValue(expression:e,type:t);values[id]=v;return v
        }
        throw i.fail("Undefined, forward or non-dominating value %\(id)")
    }
    mutating func typed(_ id:UInt32,_ type:UInt32,_ i:SPVInstruction) throws -> IRHandle<IRExpression> {
        let v=try value(id,i);guard try module.type(v.type,i) == module.type(type,i) else {throw i.fail("Operand type mismatch")};return v.expression
    }
    mutating func result(_ i:SPVInstruction,_ k:IRExpressionKind,_ ss:inout [IRHandle<IRStatement>],emit:Bool=true) throws {
        let t=try module.type(i.a[0],i);let e=try expr(t,k,i)
        values[i.a[1]]=SPVValue(expression:e,type:i.a[0]);if emit {ss.append(try stmt(.emit([e]),i))}
    }
    mutating func lower() throws -> IRFunction {
        let h=record.header,a=h.a;resultType=a[0]
        guard a[2] == 0 else {throw h.fail("Function control flags unavailable",code:.unsupportedConstruct)}
        guard let signature=module.functionTypes[a[3]],signature.result == a[0] else {throw h.fail("Function type/result mismatch")}
        var current:UInt32?
        var count=0
        for i in record.body {
            try work()
            if i.op == 55 {
                try i.count(2)
                guard current == nil,count < signature.parameters.count,signature.parameters[count] == i.a[0] else {throw i.fail("Invalid parameter/order/type")}
                let t=try module.type(i.a[0],i);parameters.append(.init(name:module.names[i.a[1]],type:t,location:i.location))
                let e=try expr(t,.functionArgument(UInt32(count)),i);values[i.a[1]]=SPVValue(expression:e,type:i.a[0]);count+=1;continue
            }
            if i.op == 248 {
                if let current,blocks[current]?.terminator == nil {throw i.fail("Block missing terminator")}
                try i.count(1);current=i.a[0];blocks[i.a[0]]=SPVBlockRecord(label:i);order.append(i.a[0]);continue
            }
            guard let id=current,var b=blocks[id],b.terminator == nil else {throw i.fail("Instruction outside block or after terminator")}
            switch i.op {
            case 245:
                try i.minimum(4);guard i.a.count%2==0,b.instructions.isEmpty,b.merge==nil else {throw i.fail("Invalid OpPhi position/operands")};b.phis.append(i)
            case 247:
                try i.count(2);guard b.merge==nil,i.a[1]==0 else {throw i.fail("Invalid selection merge flags")};b.merge=i.a[0]
            case 246:
                try i.count(3);guard b.merge==nil,i.a[2]==0 else {throw i.fail("Loop flags not implemented",code:.unsupportedConstruct)}
                b.merge=i.a[0];b.continuing=i.a[1]
            case 249,250,251,252,253,254,255:b.terminator=i
            default:
                guard b.merge==nil else {throw i.fail("Merge must immediately precede branch")};b.instructions.append(i)
            }
            blocks[id]=b
        }
        guard count==signature.parameters.count,let first=order.first else {throw h.fail("Missing parameters or function body")};entryLabel=first
        for id in order {
            guard let b=blocks[id],let term=b.terminator else {throw h.fail("Missing terminator")}
            let targets:[UInt32]
            switch term.op {
            case 249:try term.count(1);targets=[term.a[0]]
            case 250:
                guard term.a.count==3 || term.a.count==5 else {throw term.fail("Invalid conditional operands")};targets=[term.a[1],term.a[2]]
                guard b.merge != nil else {throw term.fail("Conditional requires SelectionMerge",code:.unsupportedConstruct)}
            case 253,252,255:try term.count(0);targets=[]
            case 254:try term.count(1);targets=[]
            case 251:throw term.fail("Switch reconstruction not implemented",code:.unsupportedConstruct)
            default:throw term.fail("Unknown terminator")
            }
            if let merge=b.merge {guard blocks[merge] != nil,merge != id,term.op==250 else {throw term.fail("Invalid structured merge")}}
            if let continuing=b.continuing {guard blocks[continuing] != nil,continuing != id,continuing != b.merge else {throw term.fail("Loop continuing shape unavailable",code:.unsupportedConstruct)}}
            for target in targets {try work();guard blocks[target] != nil,target != first else {throw term.fail("Invalid branch target/entry predecessor")};predecessors[target,default:[]].insert(id)}
        }
        for id in order {
            guard let b=blocks[id] else {throw h.fail("Missing block")}
            for p in b.phis {
                let t=try module.type(p.a[0],p);guard let pointer=module.localPointerTypes[t] else {throw p.fail("Phi of pointer not implemented",code:.unsupportedConstruct)}
                var incoming:Set<UInt32>=[]
                for j in stride(from:2,to:p.a.count,by:2) {
                    try work();guard incoming.insert(p.a[j+1]).inserted else {throw p.fail("Duplicate phi predecessor")}
                }
                guard !incoming.isEmpty,incoming==predecessors[id,default:[]] else {throw p.fail("Phi predecessor set mismatch")}
                try node();let local=try IRHandle<IRLocalVariable>(index:locals.count);locals.append(.init(type:t,location:p.location))
                phiPointers[p.a[1]]=try expr(pointer,.localVariable(local),p)
            }
        }
        let body=try region(first,stop:nil,depth:1)
        guard visited.count==blocks.count else {throw h.fail("Unreachable/unstructured blocks not lowered",code:.unsupportedConstruct)}
        let result:IRFunctionResult? = module.voids.contains(a[0]) ? nil : .init(type:try module.type(a[0],h))
        return IRFunction(name:module.names[a[1]],arguments:parameters,result:result,localVariables:IRTable(locals),expressions:IRTable(expressions),statements:IRTable(statements),blocks:IRTable(irBlocks),body:body,location:h.location)
    }
    mutating func edge(from:UInt32,to:UInt32,_ i:SPVInstruction) throws -> [IRHandle<IRStatement>] {
        guard let target=blocks[to] else {throw i.fail("Missing edge target")}
        var ss:[IRHandle<IRStatement>]=[],stores:[(IRHandle<IRExpression>,IRHandle<IRExpression>)]=[]
        // Freeze all incoming loads BEFORE any phi write (parallel assignment semantics).
        for p in target.phis {
            try work();var incoming:UInt32?
            for j in stride(from:2,to:p.a.count,by:2) {try work();if p.a[j+1]==from {incoming=p.a[j]}}
            guard let incoming,let pointer=phiPointers[p.a[1]] else {throw p.fail("Phi missing incoming edge")}
            let v=try typed(incoming,p.a[0],p)
            stores.append((pointer,v))
        }
        for (p,v) in stores {ss.append(try stmt(.store(pointer:p,value:v),i))}
        return ss
    }
    mutating func region(_ start:UInt32,stop:UInt32?,depth:Int) throws -> IRHandle<IRBlock> {
        guard depth<=budget.limits.depth else {throw ShaderFailure.budgetExceeded(.depth)}
        var ss:[IRHandle<IRStatement>]=[];var next:UInt32?=start
        while let id=next,id != stop {
            try work()
            guard let candidate=blocks[id] else {throw record.header.fail("Missing block")}
            if candidate.continuing != nil {
                let loop=try canonicalLoop(id,depth:depth+1)
                ss.append(try stmt(loop,candidate.label))
                let reloads=deferredPhiReloads;deferredPhiReloads=[]
                for p in reloads {guard let ptr=phiPointers[p.a[1]] else {throw p.fail("Missing phi pointer")};try result(p,.load(pointer:ptr),&ss)}
                next=candidate.merge;continue
            }
            guard visited.insert(id).inserted,let b=blocks[id],let term=b.terminator else {throw record.header.fail("Cyclic/nonstructured CFG",code:.unsupportedConstruct)}
            for p in b.phis {
                guard let pointer=phiPointers[p.a[1]] else {throw p.fail("Missing phi spill")}
                try result(p,.load(pointer:pointer),&ss)
            }
            for i in b.instructions {try operation(i,&ss,blockID:id)}
            switch term.op {
            case 249:
                let dest=term.a[0];ss += try edge(from:id,to:dest,term);next=dest
            case 250:
                guard let merge=b.merge else {throw term.fail("Missing merge")}
                let condition=try value(term.a[0],term)
                guard try module.kind(condition.type,term) == .scalar(.bool) else {throw term.fail("Branch condition must be bool")}
                let saved=values
                var yes=try edge(from:id,to:term.a[1],term)
                if term.a[1] != merge {yes.append(try stmt(.block(region(term.a[1],stop:merge,depth:depth+1)),term))}
                let accept=try block(yes,term);values=saved
                var no=try edge(from:id,to:term.a[2],term)
                if term.a[2] != merge {no.append(try stmt(.block(region(term.a[2],stop:merge,depth:depth+1)),term))}
                let reject=try block(no,term);values=saved
                ss.append(try stmt(.ifCondition(condition:condition.expression,accept:accept,reject:reject),term));next=merge
            case 253:
                guard module.voids.contains(resultType) else {throw term.fail("Return without value in nonvoid function")};ss.append(try stmt(.returnValue(nil),term));next=nil
            case 254:
                guard !module.voids.contains(resultType) else {throw term.fail("ReturnValue in void function")};let v=try typed(term.a[0],resultType,term);ss.append(try stmt(.returnValue(v),term));next=nil
            case 252:ss.append(try stmt(.kill,term));next=nil
            case 255:throw term.fail("OpUnreachable cannot be represented faithfully",code:.unsupportedConstruct)
            default:throw term.fail("Unsupported control flow",code:.unsupportedConstruct)
            }
        }
        return try block(ss,blocks[start]?.label ?? record.header)
    }
    // Deliberately narrow canonical while-loop reconstruction. Noncanonical shapes fail,
    // never flatten a back edge or discard a continuing block.
    mutating func canonicalLoop(_ id:UInt32,depth:Int) throws -> IRStatementKind {
        guard depth<=budget.limits.depth else {throw ShaderFailure.budgetExceeded(.depth)}
        guard visited.insert(id).inserted,let h=blocks[id],let term=h.terminator,
              let merge=h.merge,let continuing=h.continuing,term.op==250 else {throw record.header.fail("Invalid canonical loop")}
        let trueExit=term.a[1]==merge,falseExit=term.a[2]==merge
        guard trueExit != falseExit else {throw term.fail("Canonical loop must have one exit edge",code:.unsupportedConstruct)}
        let bodyID=trueExit ? term.a[2]:term.a[1]
        guard bodyID != continuing,bodyID != id else {throw term.fail("Loop body shape unavailable",code:.unsupportedConstruct)}
        let outer=values
        var body:[IRHandle<IRStatement>]=[]
        for p in h.phis {guard let ptr=phiPointers[p.a[1]] else {throw p.fail("Missing phi pointer")};try result(p,.load(pointer:ptr),&body)}
        for i in h.instructions {try operation(i,&body,blockID:id)}
        let condition=try value(term.a[0],term)
        guard try module.kind(condition.type,term) == .scalar(.bool) else {throw term.fail("Loop condition must be bool")}
        let headerValues=values
        var live=try edge(from:id,to:bodyID,term)
        live += try linearLoopPath(bodyID,stop:continuing)
        let liveBlock=try block(live,term)
        values=headerValues
        var exit=try edge(from:id,to:merge,term);exit.append(try stmt(.break,term));let exitBlock=try block(exit,term)
        body.append(try stmt(.ifCondition(condition:condition.expression,accept:trueExit ? exitBlock:liveBlock,reject:trueExit ? liveBlock:exitBlock),term))
        let bodyBlock=try block(body,h.label)
        // Only header-dominating expressions are available in the continuing region.
        // Values produced in a nested body block need general spilling (explicitly rejected).
        values=headerValues
        let cont=try linearLoopPath(continuing,stop:id)
        let contBlock=try block(cont,blocks[continuing]?.label ?? term)
        values=outer
        // Header phi loads inside loop body cannot escape their lexical scope.
        // Reload spills into the outer expression scope after the loop statement.
        // The caller injects these after constructing .loop via deferredPhiReloads.
        deferredPhiReloads=h.phis
        return .loop(body:bodyBlock,continuing:contBlock,breakIf:nil)
    }
    mutating func linearLoopPath(_ start:UInt32,stop:UInt32) throws -> [IRHandle<IRStatement>] {
        var result:[IRHandle<IRStatement>]=[];var id=start
        while id != stop {
            try work()
            guard visited.insert(id).inserted,let b=blocks[id],b.merge==nil,let term=b.terminator,term.op==249 else {throw record.header.fail("Generalized loop CFG not implemented",code:.unsupportedConstruct)}
            for p in b.phis {guard let ptr=phiPointers[p.a[1]] else {throw p.fail("Missing phi pointer")};try self.result(p,.load(pointer:ptr),&result)}
            for i in b.instructions {try operation(i,&result,blockID:id)}
            result += try edge(from:id,to:term.a[0],term);id=term.a[0]
        }
        return result
    }
    mutating func operation(_ i:SPVInstruction,_ ss:inout [IRHandle<IRStatement>],blockID:UInt32) throws {
        let a=i.a;try work()
        switch i.op {
        case 0:try i.count(0)
        case 59:
            guard a.count==3 || a.count==4 else {throw i.fail("Invalid local operands")}
            guard blockID==entryLabel,let p=module.pointers[a[0]],p.storage==7,a[2]==7 else {throw i.fail("Invalid local type/position")}
            let t=try module.type(p.base,i);let initExpr:IRHandle<IRExpression>?
            if a.count==4 {guard module.constantMap[a[3]] != nil else {throw i.fail("Local initializer must be constant")};initExpr=try typed(a[3],p.base,i)} else {initExpr=nil}
            try node();let local=try IRHandle<IRLocalVariable>(index:locals.count);locals.append(.init(name:module.names[a[1]],type:t,initializer:initExpr,location:i.location))
            try result(i,.localVariable(local),&ss,emit:false)
        case 61:
            try i.count(3);let v=try value(a[2],i)
            guard let p=module.pointers[v.type],p.base==a[0] else {throw i.fail("Load pointer/result mismatch")}
            try result(i,.load(pointer:v.expression),&ss)
        case 62:
            try i.count(2);let p=try value(a[0],i)
            guard let pointer=module.pointers[p.type],![2].contains(pointer.storage) else {throw i.fail("Store target not writable")}
            let v=try typed(a[1],pointer.base,i);ss.append(try stmt(.store(pointer:p.expression,value:v),i))
        case 83:
            try i.count(3);let v=try typed(a[2],a[0],i);values[a[1]]=SPVValue(expression:v,type:a[0])
        case 57:
            try i.minimum(3)
            guard let h=module.functionMap[a[2]],h.index<module.functions.count else {throw i.fail("Undefined function")}
            guard h.index < (module.functionMap[record.header.a[1]]?.index ?? 0) else {throw i.fail("Recursive/forward calls need topological remapping",code:.unsupportedConstruct)}
            let target=module.functions[h.index].header
            guard let sig=module.functionTypes[target.a[3]],sig.result==a[0],sig.parameters.count==a.count-3 else {throw i.fail("Call signature mismatch")}
            var args:[IRHandle<IRExpression>]=[]
            for (j,p) in sig.parameters.enumerated() {args.append(try typed(a[j+3],p,i))}
            let r:IRHandle<IRExpression>?
            if module.voids.contains(a[0]) {r=nil} else {let e=try expr(module.type(a[0],i),.callResult(h),i);values[a[1]]=SPVValue(expression:e,type:a[0]);r=e}
            ss.append(try stmt(.call(function:h,arguments:args,result:r),i))
        case 65:try accessChain(i,&ss)
        case 80:
            try i.minimum(2);let t=try module.type(a[0],i);let components=try componentTypes(t,i)
            guard a.count-2 == components.count else {throw i.fail("CompositeConstruct arity")}
            var es:[IRHandle<IRExpression>]=[]
            for (j,id) in a.dropFirst(2).enumerated() {
                let v=try value(id,i);guard try module.type(v.type,i)==components[j] else {throw i.fail("CompositeConstruct type mismatch")};es.append(v.expression)
            }
            try result(i,.compose(type:t,components:es),&ss)
        case 81:
            try i.minimum(4);let v=try value(a[2],i);var t=try module.type(v.type,i),e=v.expression
            for index in a.dropFirst(3) {
                try work();t=try indexedType(t,index,i)
                e=try expr(t,.accessIndex(base:e,index:index),i);ss.append(try stmt(.emit([e]),i))
            }
            guard t == module.typeMap[a[0]] else {throw i.fail("CompositeExtract result type mismatch")}
            values[a[1]]=SPVValue(expression:e,type:a[0])
        case 124:
            try i.count(3);let v=try value(a[2],i);let t=try module.type(a[0],i)
            guard try module.byteSize(t,i)==module.byteSize(module.type(v.type,i),i) else {throw i.fail("Bitcast width mismatch")}
            try result(i,.bitcast(expression:v.expression,type:t),&ss)
        case 126,127,168,200:
            try i.count(3);let v=try typed(a[2],a[0],i)
            let k=try module.kind(a[0],i)
            let scalar:IRScalar
            switch k {case .scalar(let s),.vector(_,let s):scalar=s;default:throw i.fail("Unary operand shape invalid")}
            guard (i.op==126 && scalar.kind == .sint) || (i.op==127 && scalar.kind == .float) || (i.op==168 && scalar.kind == .bool) || (i.op==200 && [.sint,.uint].contains(scalar.kind)) else {throw i.fail("Unary type mismatch")}
            let op:IRUnaryOperator=i.op==168 ? .logicalNot : i.op==200 ? .bitwiseNot:.negate
            try result(i,.unary(op:op,expression:v),&ss)
        case 169:
            try i.count(5);let c=try value(a[2],i)
            guard try module.kind(c.type,i) == .scalar(.bool) else {throw i.fail("Select vector conditions not implemented",code:.unsupportedConstruct)}
            let yes=try typed(a[3],a[0],i),no=try typed(a[4],a[0],i)
            try result(i,.select(condition:c.expression,accept:yes,reject:no),&ss)
        default:try binary(i,&ss)
        }
    }
    mutating func componentTypes(_ t:IRHandle<IRType>,_ i:SPVInstruction) throws -> [IRHandle<IRType>] {
        switch module.types[t.index].kind {
        case .vector(let n,let s):
            try work(module.types.count)
            guard let j=module.types.indices.first(where:{module.types[$0].kind == .scalar(s)}) else {throw i.fail("Missing scalar type")}
            return Array(repeating:try IRHandle(index:j),count:Int(n.rawValue))
        case .structure(let ms,_):try work(ms.count);return ms.map(\.type)
        case .array(let b,.constant(let n),_):
            guard UInt64(n)<=UInt64(i.a.count) else {throw i.fail("Composite arity")};try work(Int(n));return Array(repeating:b,count:Int(n))
        default:throw i.fail("Composite type not implemented",code:.unsupportedConstruct)
        }
    }
    mutating func indexedType(_ t:IRHandle<IRType>,_ index:UInt32,_ i:SPVInstruction) throws -> IRHandle<IRType> {
        try work()
        switch module.types[t.index].kind {
        case .structure(let ms,_):guard UInt64(index)<UInt64(ms.count) else {throw i.fail("Member index out of bounds")};return ms[Int(index)].type
        case .array(let base,let n,_):if case .constant(let count)=n {guard index<count else {throw i.fail("Array index out of bounds")}};return base
        case .vector(let n,let scalar):
            guard index<UInt32(n.rawValue) else {throw i.fail("Vector index out of bounds")};try work(module.types.count)
            guard let j=module.types.indices.first(where:{module.types[$0].kind == .scalar(scalar)}) else {throw i.fail("Missing scalar type")};return try IRHandle(index:j)
        default:throw i.fail("Indexed type unavailable",code:.unsupportedConstruct)
        }
    }
    mutating func accessChain(_ i:SPVInstruction,_ ss:inout [IRHandle<IRStatement>]) throws {
        let a=i.a;try i.minimum(4);let base=try value(a[2],i)
        guard let pointer=module.pointers[base.type],let resultPointer=module.pointers[a[0]],pointer.storage==resultPointer.storage else {throw i.fail("AccessChain pointer storage mismatch")}
        var t=try module.type(pointer.base,i),e=base.expression
        for (j,indexID) in a.dropFirst(3).enumerated() {
            let index=try value(indexID,i)
            let ik=try module.kind(index.type,i)
            guard ik == .scalar(.u32) || ik == .scalar(.i32) else {throw i.fail("Index must be i32/u32")}
            let next:IRHandle<IRType>
            if let c=module.constantMap[indexID]?.unsigned {next=try indexedType(t,c,i)}
            else {
                switch module.types[t.index].kind {
                case .array(let b,_,_):next=b
                default:throw i.fail("Dynamic access for this type unavailable",code:.unsupportedConstruct)
                }
            }
            let last=j==a.count-4
            let pt:IRHandle<IRType>
            if last {pt=try module.type(a[0],i);guard try module.type(resultPointer.base,i)==next else {throw i.fail("AccessChain result pointee mismatch")}}
            else {
                try work(module.types.count)
                let space=try module.storage(pointer.storage,i)
                guard let found=module.types.indices.first(where:{module.types[$0].kind == .pointer(base:next,space:space)}) else {throw i.fail("Missing intermediate pointer type",code:.unsupportedConstruct)}
                pt=try IRHandle(index:found)
            }
            e=try expr(pt,.access(base:e,index:index.expression),i);ss.append(try stmt(.emit([e]),i));t=next
        }
        values[a[1]]=SPVValue(expression:e,type:a[0])
    }
    mutating func binary(_ i:SPVInstruction,_ ss:inout [IRHandle<IRStatement>]) throws {
        try i.count(4);let a=i.a,l=try value(a[2],i),r=try value(a[3],i)
        let lk=try module.kind(l.type,i)
        guard try module.type(l.type,i)==module.type(r.type,i) else {throw i.fail("Binary operands differ")}
        let scalar:IRScalar;let size:IRVectorSize?
        switch lk {case .scalar(let s):scalar=s;size=nil;case .vector(let n,let s):scalar=s;size=n;default:throw i.fail("Binary aggregate operation unavailable",code:.unsupportedConstruct)}
        let op:IRBinaryOperator,compare:Bool
        switch i.op {
        case 128,129:op = .add;compare=false
        case 130,131:op = .subtract;compare=false
        case 132,133:op = .multiply;compare=false
        case 134,135,136:op = .divide;compare=false
        case 137,138:op = .modulo;compare=false
        case 139,140,141:throw i.fail("Signed/float modulo variants require special lowering",code:.unsupportedConstruct)
        case 164,170,180:op = .equal;compare=true
        case 165,171:op = .notEqual;compare=true
        case 182:throw i.fail("Ordered floating not-equal requires NaN guard lowering",code:.unsupportedConstruct)
        case 172,173,186:op = .greater;compare=true
        case 174,175,190:op = .greaterEqual;compare=true
        case 176,177,184:op = .less;compare=true
        case 178,179,188:op = .lessEqual;compare=true
        case 166:op = .logicalOr;compare=false
        case 167:op = .logicalAnd;compare=false
        case 194,195:op = .shiftRight;compare=false
        case 196:op = .shiftLeft;compare=false
        case 197:op = .inclusiveOr;compare=false
        case 198:op = .exclusiveOr;compare=false
        case 199:op = .and;compare=false
        default:throw i.fail("Unsupported function operation \(i.op)",code:.unsupportedConstruct)
        }
        // Signedness and ordered floating comparisons are preserved, not coerced silently.
        let expected:IRScalarKind?
        switch i.op {
        case 129,131,133,136,180,182,184,186,188,190:expected = .float
        case 134,137,172,174,176,178,194:expected = .uint
        case 135,138,173,175,177,179,195:expected = .sint
        case 164,165,166,167:expected = .bool
        default:expected=nil
        }
        if let expected {guard scalar.kind==expected else {throw i.fail("Opcode signedness/kind mismatch")}}
        else {guard [.uint,.sint].contains(scalar.kind) else {throw i.fail("Integer opcode on noninteger")}}
        let resultKind=try module.kind(a[0],i)
        let expectedKind:IRTypeKind = compare ? size.map { .vector(size:$0,scalar:.bool) } ?? .scalar(.bool) : lk
        guard resultKind==expectedKind else {throw i.fail("Binary result type mismatch")}
        try result(i,.binary(op:op,left:l.expression,right:r.expression),&ss)
    }
}
