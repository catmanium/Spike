module Optimizer

#====Adam============#
mutable struct Adam_struct
    learning_rate
    p1
    p2
    e
    vs #momentam
    ss #RMSProp
end
function Adam(model,option=Dict())
    gpu = model.option["GPU"]

    #デフォルト値
    learning_rate = haskey(option,"learning_rate") ? option["learning_rate"] : 0.001
    p1 = haskey(option,"p1") ? option["p1"] : 0.95
    p2 = haskey(option,"p2") ? option["p2"] : 0.99
    e  = haskey(option,"e") ? option["e"] : 10^(-12)

    vs = []
    ss = []
    for param in model.params
        v = gpu ? CUDA.zeros(size(param)) : zeros(Float64,size(param))
        s = gpu ? CUDA.zeros(size(param)) : zeros(Float64,size(param))
        append!(vs,[v])
        append!(ss,[s])
    end

    model.optimizer = Adam_struct(learning_rate,p1,p2,e,vs,ss) 
end
function update(this::Adam_struct,model)
    for i in 1:length(model.params)
        v = this.p1 .* this.vs[i] .+ (1-this.p1).*model.grads[i]
        s = this.p2 .* this.ss[i] .+ (1-this.p2).*(model.grads[i] .^2)
        model.params[i] .-= this.learning_rate*(v./sqrt.(this.e .+s))
        this.vs[i] = v
        this.ss[i] = s 
    end
end

end