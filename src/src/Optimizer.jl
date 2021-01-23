module Optimizer

#=====
params
======#
abstract type Optimizer_params end
mutable struct SGD_params <: Optimizer_params
    learning_rate::Float64 
end
mutable struct Adam_params <: Optimizer_params
    learning_rate::Float64
    p1
    p2
    e
    vs #momentam
    ss #RMSProp
end

#=========
コンストラクタ
==========#
function SGD(params::Array)
    return SGD_params(params[1])    
end
function Adam(params::Array)
    return Adam_params(params[1],params[2],params[3],params[4],params[5],params[6])
end

#=========
実行
==========#
function update(this::SGD_params,model)
    for i in 1:length(model.params)
        # layers[i].W = layers[i].W -this.learning_rate*layers[i].dW
        model.params[i] .-= this.learning_rate*model.grads[i]
    end
end
function update(this::Adam_params,model) #layer->model
    for i in 1:length(model.params)
        v = this.p1 .* this.vs[i] .+ (1-this.p1).*model.grads[i]
        s = this.p2 .* this.ss[i] .+ (1-this.p2).*(model.grads[i] .^2)
        model.params[i] .-= this.learning_rate*(v./sqrt.(this.e .+s))
        this.vs[i] = v
        this.ss[i] = s 
    end
end

end