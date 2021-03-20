#===
ファイルの読み込み
===#
# include(".jl")

# export save

#===
共通型，関数
・全ての構造体はOptimizerの子型になる
===#

export Adam,update!

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
        for j in 1:length(model.common.layers[i].params)
            params = model.common.layers[i].params[j]
            (length(params)==0) && continue
            s = copy(params) * 0
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
        for j in 1:length(model.common.layers[i].grads)
            grads = model.common.layers[i].grads[j]
            (length(grads)==0) && continue
            v = this.p1 .* this.vs[cnt] .+ (1-this.p1).*grads
            s = this.p2 .* this.ss[cnt] .+ (1-this.p2).*(grads .^2)
            model.common.layers[i].params[j] .-= this.learning_rate*(v./sqrt.(this.e .+s))
            this.vs[cnt] = v
            this.ss[cnt] = s 
            cnt+=1
        end
    end
    nothing
end