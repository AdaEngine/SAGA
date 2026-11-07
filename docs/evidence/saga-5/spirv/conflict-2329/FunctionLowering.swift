import SAGAShaderIR

struct SPVBlockRecord { let label: SPVInstruction; var code: [SPVInstruction] = [] }
struct SPVValue {
    let type: UInt32
    let expression: IRHandle<IRExpression>?
    let spill: IRHandle<IRLocalVariable>?
    let block: UInt32?
}

/// Conservative structured acyclic CFG lowering; computed SSA results spill at their definition.
struct SPVFunctionLowerer {
    let module: SPVDecoder
    let record: SPVFunctionRecord
    var budget: ShaderBudget
    var expressions: [IRExpression] = []
    var statements: [IRStatement] = []
    var blocks: [IRBlock] = []
    var locals: [IRLocalVariable] = []
    var values: [UInt32: SPVValue] = [:]
    var source: [UInt32: SPVBlockRecord] = [:]
    var order: [UInt32] = []
    var predecessors: [UInt32: Set<UInt32>] = [:]
    var dominators: [UInt32: Set<UInt32>] = [:]
    var visited: Set<UInt32> = []
    var arguments: [IRFunctionArgument] = []
    var currentBlock: UInt32 = 0

    mutating func charge() throws { try budget.consume(.work) }
    mutating func expression(_ type: IRHandle<IRType>, _ kind: IRExpressionKind, _ i: SPVInstruction) throws -> IRHandle<IRExpression> {
        try charge(); try budget.consume(.nodes)
        let h = try IRHandle<IRExpression>(index: expressions.count)
        expressions.append(.init(type: type, kind: kind, location: i.location)); return h
    }
    mutating func statement(_ kind: IRStatementKind, _ i: SPVInstruction, into body: inout [IRHandle<IRStatement>]) throws {
        try charge(); try budget.consume(.nodes)
        let h = try IRHandle<IRStatement>(index: statements.count)
        statements.append(.init(kind: kind, location: i.location)); body.append(h)
    }
    mutating func block(_ body: [IRHandle<IRStatement>], _ i: SPVInstruction) throws -> IRHandle<IRBlock> {
        try charge(); try budget.consume(.nodes)
        let h = try IRHandle<IRBlock>(index: blocks.count)
        blocks.append(.init(statements: body, location: i.location)); return h
    }
    mutating func local(_ type: IRHandle<IRType>, _ i: SPVInstruction) throws -> IRHandle<IRLocalVariable> {
        try charge(); try budget.consume(.nodes)
        let h = try IRHandle<IRLocalVariable>(index: locals.count)
        locals.append(.init(type: type, location: i.location)); return h
    }
    func pointerType(_ base: IRHandle<IRType>, _ i: SPVInstruction) throws -> IRHandle<IRType> {
        guard let t = module.spillPointerTypes[base] else { throw i.fail("Missing internal spill pointer type") }; return t
    }
    mutating func read(_ id: UInt32, _ i: SPVInstruction, into body: inout [IRHandle<IRStatement>], at edge: UInt32? = nil) throws -> (IRHandle<IRExpression>, UInt32) {
        try charge()
        if let c = module.constantMap[id] { return (try expression(module.type(c.type,i), .constant(c.handle),i),c.type) }
        if let g = module.globalMap[id] { return (try expression(module.type(g.1,i), .globalVariable(g.0),i),g.1) }
        guard let v = values[id] else { throw i.fail("Undefined value %\(id)") }
        if let definition = v.block {
            guard dominators[edge ?? currentBlock]?.contains(definition) == true else { throw i.fail("Value %\(id) does not dominate use") }
        }
        if let slot = v.spill {
            let t = try module.type(v.type,i), p = try expression(pointerType(module.type(v.type,i),i),.localVariable(slot),i)
            let e = try expression(t,.load(pointer:p),i); try statement(.emit([e]),i,into:&body); return(e,v.type)
        }
        guard let e=v.expression else { throw i.fail("Void result used as value") }
        if let definition=v.block, definition != (edge ?? currentBlock) { throw i.fail("Cross-block pointer SSA unavailable",code:.unsupportedConstruct) }
        return(e,v.type)
    }
    mutating func publish(_ id:UInt32,type:UInt32,value:IRHandle<IRExpression>,_ i:SPVInstruction,into body:inout [IRHandle<IRStatement>]) throws {
        let t=try module.type(type,i)
        if case .pointer=module.types[t.index].kind { values[id]=SPVValue(type:type,expression:value,spill:nil,block:currentBlock);return }
        let slot=try local(t,i), p=try expression(pointerType(t,i),.localVariable(slot),i)
        try statement(.store(pointer:p,value:value),i,into:&body)
        values[id]=SPVValue(type:type,expression:nil,spill:slot,block:currentBlock)
    }
    func successors(_ b:SPVBlockRecord) throws -> [UInt32] {
        guard let i=b.code.last else {throw b.label.fail("Block has no terminator")}
        switch i.op {
        case 249:try i.count(1);return [i.a[0]]
        case 250:try i.count(3);return [i.a[1],i.a[2]]
        case 253,252,255:try i.count(0);return []
        case 254:try i.count(1);return []
        default:throw i.fail("Unsupported or missing terminator",code:.unsupportedConstruct)
        }
    }
    mutating func lower() throws -> IRFunction {
        let h=record.header,a=h.a
        guard a[2]==0, let sig=module.functionTypes[a[3]],sig.result==a[0] else {throw h.fail("Function control/signature unsupported or mismatched")}
        var parameters:[SPVInstruction]=[],last:UInt32?
        for i in record.body {
            try charge()
            if i.op==55 {guard last==nil else {throw i.fail("Parameter after label")};try i.count(2);parameters.append(i);continue}
            if i.op==248 {last=i.a[0];order.append(i.a[0]);source[i.a[0]]=SPVBlockRecord(label:i);continue}
            guard let id=last,var b=source[id] else {throw i.fail("Instruction outside block")}
            if let tail=b.code.last,[249,250,251,252,253,254,255].contains(tail.op) {throw i.fail("Instruction after terminator")}
            b.code.append(i);source[id]=b
        }
        guard parameters.count==sig.parameters.count,let first=order.first else {throw h.fail("Parameter count or function body missing")}
        for (n,p) in parameters.enumerated() {
            guard p.a[0]==sig.parameters[n] else {throw p.fail("Parameter type mismatch")}
            let t=try module.type(p.a[0],p)
            arguments.append(.init(name:module.names[p.a[1]],type:t,location:p.location))
            values[p.a[1]]=SPVValue(type:p.a[0],expression:try expression(t,.functionArgument(UInt32(n)),p),spill:nil,block:nil)
        }
        for id in order {
            guard let b=source[id] else {throw h.fail("Invalid block table")}
            for target in try successors(b) {try charge();guard source[target] != nil,target != first else {throw b.label.fail("Invalid branch target")};predecessors[target,default:[]].insert(id)}
        }
        let universe=Set(order)
        for id in order {try budget.consume(.work,amount:universe.count);dominators[id]=id==first ? [first]:universe}
        var changed=true
        while changed {
            changed=false
            for id in order.dropFirst() {
                try charge();let ps=predecessors[id,default:[]]
                guard !ps.isEmpty else {throw source[id]?.label.fail("Unreachable block") ?? h.fail("Missing block")}
                var d=universe
                for p in ps {try budget.consume(.work,amount:d.count);d.formIntersection(dominators[p,default:[]])}
                d.insert(id);if d != dominators[id] {dominators[id]=d;changed=true}
            }
        }
        for id in order {
            guard let b=source[id] else {throw h.fail("Missing block")};var afterPhi=false
            for i in b.code {
                try charge();if i.op != 245 {afterPhi=true;continue}
                guard !afterPhi,i.a.count>=4,i.a.count%2==0 else {throw i.fail("Malformed or misplaced Phi")}
                var seen:Set<UInt32>=[]
                for n in stride(from:3,to:i.a.count,by:2) {try charge();guard seen.insert(i.a[n]).inserted else {throw i.fail("Duplicate Phi predecessor")}}
                guard seen==predecessors[id,default:[]] else {throw i.fail("Phi predecessor list mismatch")}
                let t=try module.type(i.a[0],i)
                if case .pointer=module.types[t.index].kind {throw i.fail("Pointer Phi unsupported",code:.unsupportedConstruct)}
                values[i.a[1]]=SPVValue(type:i.a[0],expression:nil,spill:try local(t,i),block:id)
            }
        }
        let body=try region(first,stop:nil,depth:1)
        guard visited==universe else {throw h.fail("Unreachable or unstructured blocks",code:.unsupportedConstruct)}
        let result:IRFunctionResult? = module.voids.contains(sig.result) ? nil:.init(type:try module.type(sig.result,h))
        return IRFunction(name:module.names[a[1]],arguments:arguments,result:result,localVariables:IRTable(locals),expressions:IRTable(expressions),statements:IRTable(statements),blocks:IRTable(blocks),body:body,location:h.location)
    }
    mutating func edge(from:UInt32,to:UInt32,_ i:SPVInstruction,into body:inout [IRHandle<IRStatement>]) throws {
        guard let b=source[to] else {throw i.fail("Missing target block")}
        var copies:[(IRHandle<IRExpression>,IRHandle<IRExpression>)]=[]
        // Capture ALL incoming values before ANY Phi edge store (parallel copies).
        for phi in b.code.prefix(while:{$0.op==245}) {
            try charge();guard let v=values[phi.a[1]],let slot=v.spill else {throw phi.fail("Missing Phi spill")}
            var incoming:UInt32?
            for n in stride(from:3,to:phi.a.count,by:2) where phi.a[n]==from {incoming=phi.a[n-1]}
            guard let incoming else {throw phi.fail("Missing Phi incoming edge")}
            let(value,type)=try read(incoming,phi,into:&body,at:from);guard type==v.type else {throw phi.fail("Phi type mismatch")}
            let p=try expression(pointerType(module.type(v.type,phi),phi),.localVariable(slot),phi);copies.append((p,value))
        }
        for(p,v) in copies {try statement(.store(pointer:p,value:v),i,into:&body)}
    }
    mutating func region(_ start:UInt32,stop:UInt32?,depth:Int) throws -> IRHandle<IRBlock> {
        guard depth<=min(128,budget.limits.depth) else {throw ShaderFailure.budgetExceeded(.depth)}
        var body:[IRHandle<IRStatement>]=[],next:UInt32?=start
        let origin=source[start]?.label ?? record.header
        while let id=next,id != stop {
            try charge();guard visited.insert(id).inserted,let b=source[id],let term=b.code.last else {throw origin.fail("Cycle or overlapping region",code:.unsupportedConstruct)}
            currentBlock=id;var merge:UInt32?,variablesFinished=false
            for(n,i) in b.code.dropLast().enumerated() {
                try charge();if i.op != 59 {variablesFinished=true}
                switch i.op {
                case 245:break
                case 247:try i.count(2);guard i.a[1]==0,n==b.code.count-2,source[i.a[0]] != nil else {throw i.fail("Invalid SelectionMerge")};merge=i.a[0]
                case 246:throw i.fail("Loop reconstruction not implemented",code:.unsupportedConstruct)
                default:try instruction(i,firstBlock:id==order.first && !variablesFinished,into:&body)
                }
            }
            switch term.op {
            case 249:guard merge==nil else {throw term.fail("SelectionMerge without conditional")};try edge(from:id,to:term.a[0],term,into:&body);next=term.a[0]
            case 250:
                guard let merge,merge != id,term.a[1] != term.a[2] else {throw term.fail("Unstructured conditional",code:.unsupportedConstruct)}
                let(c,t)=try read(term.a[0],term,into:&body);guard try module.kind(t,term) == .scalar(.bool) else {throw term.fail("Branch condition not bool")}
                let accept=try branch(from:id,target:term.a[1],stop:merge,depth:depth+1,term)
                let reject=try branch(from:id,target:term.a[2],stop:merge,depth:depth+1,term)
                try statement(.ifCondition(condition:c,accept:accept,reject:reject),term,into:&body);next=merge
            case 253:guard module.voids.contains(record.header.a[0]),merge==nil else {throw term.fail("Return without required value")};try statement(.returnValue(nil),term,into:&body);next=nil
            case 254:let(v,t)=try read(term.a[0],term,into:&body);guard t==record.header.a[0],merge==nil else {throw term.fail("Return type mismatch")};try statement(.returnValue(v),term,into:&body);next=nil
            default:throw term.fail("Unsupported terminator",code:.unsupportedConstruct)
            }
        }
        return try block(body,origin)
    }
    mutating func branch(from:UInt32,target:UInt32,stop:UInt32,depth:Int,_ i:SPVInstruction) throws -> IRHandle<IRBlock> {
        var body:[IRHandle<IRStatement>]=[];try edge(from:from,to:target,i,into:&body)
        if target != stop {let child=try region(target,stop:stop,depth:depth);try statement(.block(child),i,into:&body)}
        return try block(body,i)
    }
    mutating func instruction(_ i:SPVInstruction,firstBlock:Bool,into body:inout [IRHandle<IRStatement>]) throws {
        let a=i.a
        switch i.op {
        case 0:try i.count(0)
        case 59:
            guard firstBlock,a.count==3 || a.count==4,let p=module.pointers[a[0]],p.storage==7,a[2]==7 else {throw i.fail("Invalid local variable")}
            let t=try module.type(p.base,i),slot=try local(t,i),ptr=try expression(module.type(a[0],i),.localVariable(slot),i)
            values[a[1]]=SPVValue(type:a[0],expression:ptr,spill:nil,block:nil)
            if a.count==4 {let(v,vt)=try read(a[3],i,into:&body);guard vt==p.base,module.constantMap[a[3]] != nil else {throw i.fail("Bad local initializer")};try statement(.store(pointer:ptr,value:v),i,into:&body)}
        case 62:
            try i.count(2);let(p,t)=try read(a[0],i,into:&body),(v,vt)=try read(a[1],i,into:&body)
            guard let pt=module.pointers[t],pt.base==vt,[6,7,4,12].contains(pt.storage) else {throw i.fail("Store type/access mismatch")}
            try statement(.store(pointer:p,value:v),i,into:&body)
        case 61:
            try i.count(3);let(p,t)=try read(a[2],i,into:&body);guard let pt=module.pointers[t],pt.base==a[0] else {throw i.fail("Load type mismatch")}
            let v=try expression(module.type(a[0],i),.load(pointer:p),i);try statement(.emit([v]),i,into:&body);try publish(a[1],type:a[0],value:v,i,into:&body)
        case 83:
            try i.count(3);let(v,t)=try read(a[2],i,into:&body);guard t==a[0] else {throw i.fail("CopyObject type mismatch")};try publish(a[1],type:t,value:v,i,into:&body)
        case 57:
            try i.minimum(3)
            guard let fn=module.functionMap[a[2]],let own=module.functionMap[record.header.a[1]],fn.index<own.index,
                  let sig=module.functionTypes[module.functions[fn.index].header.a[3]],sig.result==a[0],sig.parameters.count==a.count-3 else {throw i.fail("Forward/recursive call or signature unsupported",code:.unsupportedConstruct)}
            var args:[IRHandle<IRExpression>]=[]
            for(j,id) in a.dropFirst(3).enumerated() {let(v,t)=try read(id,i,into:&body);guard t==sig.parameters[j] else {throw i.fail("Call argument mismatch")};args.append(v)}
            if module.voids.contains(a[0]) {try statement(.call(function:fn,arguments:args,result:nil),i,into:&body);values[a[1]]=SPVValue(type:a[0],expression:nil,spill:nil,block:currentBlock)}
            else {let v=try expression(module.type(a[0],i),.callResult(fn),i);try statement(.call(function:fn,arguments:args,result:v),i,into:&body);try publish(a[1],type:a[0],value:v,i,into:&body)}
        case 65,80,81,124:throw i.fail("Access/composite/bitcast lowering unavailable",code:.unsupportedConstruct)
        case 126,127,168,200:
            try i.count(3);let(v,t)=try read(a[2],i,into:&body);guard t==a[0] else {throw i.fail("Unary type mismatch")}
            let s=try numericScalar(t,i),op:IRUnaryOperator
            switch i.op {
            case 126:guard s.kind == .sint else {throw i.fail("SNegate requires sint")};op = .negate
            case 127:guard s.kind == .float else {throw i.fail("FNegate requires float")};op = .negate
            case 168:guard s.kind == .bool else {throw i.fail("LogicalNot requires bool")};op = .logicalNot
            default:guard s.kind == .uint || s.kind == .sint else {throw i.fail("Not requires integer")};op = .bitwiseNot
            }
            let e=try expression(module.type(t,i),.unary(op:op,expression:v),i);try statement(.emit([e]),i,into:&body);try publish(a[1],type:t,value:e,i,into:&body)
        case 169:
            try i.count(5);let(c,ct)=try read(a[2],i,into:&body),(v,t)=try read(a[3],i,into:&body),(r,rt)=try read(a[4],i,into:&body)
            guard try module.kind(ct,i) == .scalar(.bool),t==rt,t==a[0] else {throw i.fail("Select type mismatch")}
            let e=try expression(module.type(t,i),.select(condition:c,accept:v,reject:r),i);try statement(.emit([e]),i,into:&body);try publish(a[1],type:t,value:e,i,into:&body)
        default:try binary(i,into:&body)
        }
    }
    func numericScalar(_ id:UInt32,_ i:SPVInstruction) throws -> IRScalar {
        switch try module.kind(id,i) {case .scalar(let s),.vector(_,let s):return s;default:throw i.fail("Expected scalar/vector")}
    }
    mutating func binary(_ i:SPVInstruction,into body:inout [IRHandle<IRStatement>]) throws {
        try i.count(4);let a=i.a
        let(l,lt)=try read(a[2],i,into:&body),(r,rt)=try read(a[3],i,into:&body)
        guard lt==rt else {throw i.fail("Binary operand mismatch")}
        let s=try numericScalar(lt,i),op:IRBinaryOperator,required:IRScalarKind?
        var comparison=false
        switch i.op {
        case 128:op = .add;required=nil
        case 129:op = .add;required = .float
        case 130:op = .subtract;required=nil
        case 131:op = .subtract;required = .float
        case 132:op = .multiply;required=nil
        case 133:op = .multiply;required = .float
        case 134:op = .divide;required = .uint
        case 135:op = .divide;required = .sint
        case 136:op = .divide;required = .float
        case 137:op = .modulo;required = .uint
        case 138:op = .modulo;required = .sint
        case 164:op = .equal;required = .bool;comparison=true
        case 165:op = .notEqual;required = .bool;comparison=true
        case 166:op = .logicalOr;required = .bool
        case 167:op = .logicalAnd;required = .bool
        case 170:op = .equal;required=nil;comparison=true
        case 171:op = .notEqual;required=nil;comparison=true
        case 172:op = .greater;required = .uint;comparison=true
        case 173:op = .greater;required = .sint;comparison=true
        case 174:op = .greaterEqual;required = .uint;comparison=true
        case 175:op = .greaterEqual;required = .sint;comparison=true
        case 176:op = .less;required = .uint;comparison=true
        case 177:op = .less;required = .sint;comparison=true
        case 178:op = .lessEqual;required = .uint;comparison=true
        case 179:op = .lessEqual;required = .sint;comparison=true
        case 180:op = .equal;required = .float;comparison=true
        case 184:op = .less;required = .float;comparison=true
        case 186:op = .greater;required = .float;comparison=true
        case 188:op = .lessEqual;required = .float;comparison=true
        case 190:op = .greaterEqual;required = .float;comparison=true
        case 197:op = .inclusiveOr;required=nil
        case 198:op = .exclusiveOr;required=nil
        case 199:op = .and;required=nil
        default:throw i.fail("Unsupported arithmetic opcode \(i.op)",code:.unsupportedConstruct)
        }
        guard required.map({$0==s.kind}) ?? (s.kind == .sint || s.kind == .uint) else {throw i.fail("Opcode/operand kind mismatch")}
        if comparison {
            let result=try module.kind(a[0],i),operand=try module.kind(lt,i),expected:IRTypeKind
            if case .vector(let n,_)=operand {expected = .vector(size:n,scalar:.bool)} else {expected = .scalar(.bool)}
            guard result==expected else {throw i.fail("Comparison shape mismatch")}
        } else {guard a[0]==lt else {throw i.fail("Binary result mismatch")}}
        let e=try expression(module.type(a[0],i),.binary(op:op,left:l,right:r),i)
        try statement(.emit([e]),i,into:&body);try publish(a[1],type:a[0],value:e,i,into:&body)
    }
}
