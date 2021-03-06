#===
ファイルの読み込み
===#
# include(".jl")

# export save

#===
共通型，関数
・全ての構造体はOptimizerの子型になる
===#

export SGD,Momentum,Adam,update!

#===SGD==============#
mutable struct SGD <: Optimizer
    learning_rate
    function SGD(;learning_rate=0.001)
        new(learning_rate)
    end
end
function fit!(this::SGD,model::Models)
    model.common.optimizer = this
end
function update!(this::SGD,model::Models)
    cnt = 1
    @inbounds for i in 1:length(model.common.layers)
        grads = model.common.layers[i].grads
        if grads === nothing
            continue
        end
        for j in 1:length(grads)
            grad = grads[j]
            model.common.layers[i].params[j] .-= this.learning_rate*grad
            cnt+=1
        end
    end
    nothing
end
function convert_to_cu!(this::SGD,model::Models)
    return nothing
end
function convert_to_array!(this::SGD,model::Models)
    return nothing
end

#===Momentum=========#
mutable struct Momentum <: Optimizer
    learning_rate
    vs #速度
    momentum
    function Momentum(;learning_rate=0.001,momentum=0.8)
        new(learning_rate,[],momentum)
    end
end
function fit!(this::Momentum,model::Models)
    @inbounds for i in 1:length(model.common.layers)
        params = model.common.layers[i].params
        if params === nothing
            continue
        end
        for j in 1:length(params)
            param = params[j]
            s = copy(param) * 0
            append!(this.vs,[s])
        end
    end
    model.common.optimizer = this
    nothing
end
function update!(this::Momentum,model::Models)
    cnt = 1
    @inbounds for i in 1:length(model.common.layers)
        grads = model.common.layers[i].grads
        if grads === nothing
            continue
        end
        for j in 1:length(grads)
            grad = grads[j]
            this.vs[cnt] = this.momentum * this.vs[cnt] - this.learning_rate * grad
            model.common.layers[i].params[j] .+= this.vs[cnt]
            cnt+=1
        end
    end
    nothing
end
function convert_to_cu!(this::Momentum,model::Models)
    this.vs = cu.(this.vs)
    return nothing
end
function convert_to_array!(this::Momentum,model::Models)
    this.vs = Array.(this.vs)
    return nothing
end

#====Adam============#
mutable struct Adam <: Optimizer
    learning_rate
    p1
    p2
    e
    vs #momentam
    ss #RMSProp
    function Adam(;learning_rate=0.001,p1=0.95,p2=0.99,e=10^(-12))
        new(learning_rate,p1,p2,e,[],[])
    end
end
function fit!(this::Adam,model::Models)
    @inbounds for i in 1:length(model.common.layers)
        params = model.common.layers[i].params
        if params === nothing
            continue
        end
        for j in 1:length(params)
            param = params[j]
            s = copy(param) * 0
            append!(this.vs,[s])
            append!(this.ss,[s])
        end
    end
    model.common.optimizer = this
    nothing
end
function update!(this::Adam,model::Models)
    cnt = 1
    @inbounds for i in 1:length(model.common.layers)
        grads = model.common.layers[i].grads
        if grads === nothing
            continue
        end
        for j in 1:length(grads)
            grad = grads[j]
            v = this.p1 .* this.vs[cnt] .+ (1-this.p1).*grad
            s = this.p2 .* this.ss[cnt] .+ (1-this.p2).*(grad .^2)
            model.common.layers[i].params[j] .-= this.learning_rate*(v./sqrt.(this.e .+s))
            this.vs[cnt] = v
            this.ss[cnt] = s 
            cnt+=1
        end
    end
    nothing
end
function convert_to_cu!(this::Adam,model::Models)
    this.vs = cu.(this.vs)
    this.ss = cu.(this.ss)
    return nothing
end
function convert_to_array!(this::Adam,model::Models)
    this.vs = Array.(this.vs)
    this.ss = Array.(this.ss)
    return nothing
end