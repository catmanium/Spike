module Layers

using UnPack,CUDA,LinearAlgebra
#=================================
・params,gradsは共通で持たせる
・add_レイヤ：コンストラクタは配列で返す．
・Float32で統一
==================================#

UnionFloatArray = Union{Array{Float64,2},CuArray{Float32,2},Nothing}

#=====Affine==============#
mutable struct Affine
    params::Array{UnionFloatArray,1} #W,b
    grads::Array{UnionFloatArray,1} #dW,db
    in::UnionFloatArray
    Affine(params,grads,in) = new(params,grads,in)
end
function forward(this::Affine,in)
    this.in = in
    return in * this.params[1] .+ this.params[2]
end
function backward(this::Affine,din)
    mul!(this.grads[1],(this.in)',din)
    this.grads[2] .= sum(din,dims=1)
    re = zeros(size(din,1),size(this.params[1],1))
    mul!(re,din,(this.params[1])')
    return re
end
function add_Affine(pre_neuron,neurons,option)
    gpu = option["GPU"]
    layers = []
    @inbounds for neuron in neurons
        W = gpu ? CUDA.randn(pre_neuron,neuron) : randn(pre_neuron,neuron)
        b = gpu ? CUDA.zeros(1,neuron) : zeros(1,neuron)
        params = [W,b]
        dW = gpu ? CUDA.zeros(size(W)) : zeros(size(W))
        db = gpu ? CUDA.zeros(size(b)) : zeros(size(b))
        grads = [dW,db]
        in = gpu ? CUDA.zeros(size(b)) : zeros(neuron,pre_neuron)
        pre_neuron = neuron
        append!(layers,[Affine(params,grads,in)])
    end
    return layers
end
function reset(this::Affine)
    return 0
end

#=====uni_LSTM==============#
mutable struct uni_LSTM
    params::Array{UnionFloatArray,1} #Wx,Wh,b
    grads::Array{UnionFloatArray,1} #dWx,dWh,db
    cache::Array{UnionFloatArray,1} #x, h_prev, h_next,i,f,g,o,c_next
    uni_LSTM(params,grads) = new(params,grads,[nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing])
end
function forward(this::uni_LSTM,x,h_prev,c_prev)
    Wx, Wh, b = this.params
    N, H = size(h_prev)
    A = x * Wx + h_prev * Wh .+ b

    #slice
    f = @view A[:, begin:H]
    g = @view A[:, H+1:2*H]
    i = @view A[:, 2*H+1:3*H]
    o = @view A[:, 3*H+1:end]

    f = 1.0 ./ (1.0 .+ exp.(-f)) #Sigmoid
    g = tanh.(g)
    i = 1.0 ./ (1.0 .+ exp.(-i))
    o = 1.0 ./ (1.0 .+ exp.(-o))

    c_next = f .* c_prev + g .* i

    h_next = o .* tanh.(c_next)

    # this.cache = [x, h_prev, c_prev, i, f, g, o, c_next]
    this.cache[1] = x
    this.cache[2] = h_prev
    this.cache[3] = c_prev
    this.cache[4] = i
    this.cache[5] = f
    this.cache[6] = g
    this.cache[7] = o
    this.cache[8] = c_next

    return h_next, c_next
end
function backward(this::uni_LSTM,dh_next,dc_next)
    Wx, Wh, b = this.params
    x, h_prev, c_prev, i, f, g, o, c_next = this.cache

    tanh_c_next = tanh.(c_next)

    ds = dc_next + (dh_next .* o) .* (1 .- tanh_c_next.^2)

    dc_prev = ds .* f

    di = ds .* g
    df = ds .* c_prev
    d_o = dh_next .* tanh_c_next
    dg = ds .* i

    di .*= i .* (1 .- i)
    df .*= f .* (1 .- f)
    d_o .*= o .* (1 .- o)
    dg .*= (1 .- g.^2)

    dA = hcat(df,dg,di,d_o)

    # dWh = h_prev' * dA
    # dWx = x' * dA
    # db = sum(dA,dims=1)

    # this.grads[1] .= dWx
    # this.grads[2] .= dWh
    mul!(this.grads[1],x',dA)
    mul!(this.grads[2],h_prev',dA)
    this.grads[3] .= sum(dA,dims=1)

    dx = zeros(size(dA,1),size(Wx,1))
    dh_prev = zeros(size(dA,1),size(Wh,1))
    mul!(dx,dA,Wx')
    mul!(dh_prev,dA,Wh')
    # dx = dA * Wx'
    # dh_prev = dA * Wh'

    return dx, dh_prev, dc_prev
end
function add_uni_LSTM(pre_neuron,neurons,option)
    gpu = option["GPU"]
    layers = []
    @inbounds for neuron in neurons
        Wx = gpu ? CUDA.randn(pre_neuron,4*neuron)/sqrt(pre_neuron) : randn(pre_neuron,4*neuron)/sqrt(pre_neuron)
        Wh = gpu ? CUDA.randn(neuron,4*neuron)/sqrt(neuron) : randn(neuron,4*neuron)/sqrt(neuron)
        b = gpu ? CUDA.zeros(1,4*neuron) : zeros(1,4*neuron)
        params = [Wx,Wh,b]

        dWx = gpu ? CUDA.zeros(size(Wx)) : zeros(size(Wx))
        dWh = gpu ? CUDA.zeros(size(Wh)) : zeros(size(Wh))
        db = gpu ? CUDA.zeros(size(b)) : zeros(size(b))
        grads = [dWx,dWh,db]

        pre_neuron = neuron
        append!(layers,[uni_LSTM(params,grads)])
    end
    return layers
end
function reset(this::uni_LSTM)
    return 0
end

UnionLSTMLayers = Union{uni_LSTM,Nothing}
#====LSTM===================#
mutable struct LSTM
    params::Array{UnionFloatArray,1} #Wx,Wh,b
    grads::Array{UnionFloatArray,1} #dWx,dWh,db
    layers::Array{UnionLSTMLayers,1} #複数のLSTMレイヤを管理
    h::UnionFloatArray #次のエポックに引き継ぐ
    c::UnionFloatArray
    dh::UnionFloatArray
    option::Dict{String,Any}
    LSTM(params,grads,option::Dict{String,Any}) = new(
        params,
        grads,
        [nothing],
        nothing,
        nothing,
        nothing,
        option
    )
end
function forward(this::LSTM,xs)
    #sequenceに応じて出力を変える(hs{N,T,H},h{N,H})
    #{N,1,H}のように3次元で固定する場合，他レイヤの仕様変更が必要

    @unpack stateful, out_sequence, padding, GPU = this.option
    Wx,Wh,b = this.params
    params = [Wx,Wh,b]
    dWx = GPU ? CUDA.zeros(size(Wx)) : zeros(size(Wx))
    dWh = GPU ? CUDA.zeros(size(Wh)) : zeros(size(Wh))
    db = GPU ? CUDA.zeros(size(b)) : zeros(size(b))
    grads = [dWx,dWh,db]
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    H = size(Wh,1)

    hs = GPU ? CUDA.zeros((N,T,H)) : zeros(Float64,(N,T,H))

    if !stateful || this.h === nothing
        this.h = GPU ? CUDA.zeros((N,H)) : zeros(Float64,(N,H))
    end

    if !stateful || this.c === nothing
        this.c = GPU ? CUDA.zeros((N,H)) : zeros(Float64,(N,H))
    end

    if this.layers[1] === nothing
        empty!(this.layers)
        @inbounds for t in 1:T
            rnn_layer = uni_LSTM(params,grads)
            this.h, this.c = forward(rnn_layer,xs[:,t,:],this.h,this.c)
            hs[:,t,:] = this.h
            push!(this.layers,rnn_layer)
        end
    else
        @inbounds for t in 1:T
            this.layers[t].params = params
            this.layers[t].grads = grads
            this.h, this.c = forward(this.layers[t],xs[:,t,:],this.h,this.c)
            hs[:,t,:] = this.h
        end
    end


    return out_sequence ? hs : this.h #this.h{N,H}
end
        #後で改善
function backward(this::LSTM,dhs)
    gpu = this.option["GPU"]
    Wx, Wh, b = this.params
    D, H = size(Wx)
    
    #dhsの次元数から処理を分岐
    if ndims(dhs) == 2
        #many to one (dhs が (N,H))
        N, H = size(dhs)
        T = length(this.layers)
        dxs = gpu ? CUDA.zeros((N,T,D)) : zeros(Float64,(N,T,D))
        #代入する勾配
        grads = copy(this.grads)
        grads -= grads #全て0に
        
        dh = dhs
        dc = gpu ? CUDA.zeros(size(dh)) : zeros(Float64,size(dh))
        
        @inbounds for t in T:-1:1
            rnn_layer = this.layers[t]
            dx, dh, dc = backward(rnn_layer,dh,dc)
            dxs[:,t,:] = dx

            #各レイヤの勾配合算 
            grads += rnn_layer.grads
        end

        #TimeRNN,Modelsの勾配更新-> Model.gradsとメモリ共有しているため同時に更新される
        this.grads[1][:] = grads[1][:]
        this.grads[2][:] = grads[2][:]
        this.grads[3][:] = grads[3][:]

        this.dh = dh
        return dxs

    elseif ndims(dhs) == 3
        #many to many (dhs が (N,T,H))
        N, T, H = size(dhs)
        dxs = gpu ? CUDA.zeros((N,T,D)) : zeros(Float64,(N,T,D))

        dh = gpu ? CUDA.zeros(size(dhs[:,1,:])) : zeros(Float64,size(dhs[:,1,:]))
        dc = gpu ? CUDA.zeros(size(dhs[:,1,:])) : zeros(Float64,size(dhs[:,1,:]))
        @inbounds for t in T:-1:1
            rnn_layer = this.layers[t]
            dx, dh, dc = backward(rnn_layer,dhs[:,t,:]+dh, dc)
            dxs[:,t,:] = dx

            #勾配合算 -> Model.gradsとメモリ共有しているため同時に更新される
            for i in 1:3
                this.grads[i] .+= rnn_layer.grads[i]
            end
        end
        this.dh = dh
        return dxs
    end
end
function add_LSTM(pre_neuron,neurons,arg_option)
    gpu = arg_option["GPU"]
    #最後のレイヤ以外はoptionにかかわらずシーケンス
    layers = []
    option = copy(arg_option)
    option["out_sequence"] = true
    @inbounds for neuron in neurons
        Wx = gpu ? CUDA.randn(pre_neuron,4*neuron)/sqrt(pre_neuron) : randn(pre_neuron,4*neuron)/sqrt(pre_neuron)
        Wh = gpu ? CUDA.randn(neuron,4*neuron)/sqrt(neuron) : randn(neuron,4*neuron)/sqrt(neuron)
        b = gpu ? CUDA.zeros(1,4*neuron) : zeros(1,4*neuron)
        params = [Wx,Wh,b]

        dWx = gpu ? CUDA.zeros(size(Wx)) : zeros(size(Wx))
        dWh = gpu ? CUDA.zeros(size(Wh)) : zeros(size(Wh))
        db = gpu ? CUDA.zeros(size(b)) : zeros(size(b))
        grads = [dWx,dWh,db]

        pre_neuron = neuron
        append!(layers,[LSTM(params,grads,option)])
    end

    layers[end].option = arg_option

    return layers
end
function reset(this::LSTM)
    this.c = nothing
    this.h = nothing
end

#====Sigmoid===========================#
mutable struct Sigmoid
    params
    grads
    s::UnionFloatArray
    Sigmoid() = new([],[],nothing)
end
function forward(this::Sigmoid,in)
    s = 1.0 ./ (1.0 .+ exp.(-in))
    this.s = s
    return s
end
function backward(this::Sigmoid,din)
    return (1.0 .- this.s) .* this.s
end
function add_Sigmoid(pre_neuron,neurons,arg_option)
    layers = []
    @inbounds for neuron in neurons
        pre_neuron = neuron
        append!(layers,[Sigmoid()])
    end
    return layers
end
function reset(this::Sigmoid)
end
#====Sigmoid_with_loss=================#
mutable struct Sigmoid_with_loss
    params
    grads
    s::UnionFloatArray #スコア
    t::UnionFloatArray
    Sigmoid_with_loss() = new([],[],nothing,nothing)
end
function forward(this::Sigmoid_with_loss,in)
    s = 1.0 ./ (1.0 .+ exp.(-in))
    delta = 1e-7
    this.s = s
    if this.t === nothing
        #ただの推論
        return s
    end
    l = -this.t .* log.(delta.+s) - (1 .-this.t) .* log.(1+delta .-s)
    return l
end
function backward(this::Sigmoid_with_loss,din)
    return this.s - this.t
end
function add_Sigmoid_with_loss()
    return Sigmoid_with_loss()
end
function reset(this::Sigmoid_with_loss)
    this.t = nothing
end

#====MSE==============================#
mutable struct Mean_Squared_Error
    params
    grads
    s::UnionFloatArray
    t::UnionFloatArray
    Mean_Squared_Error() = new([],[],nothing,nothing)
end
function forward(this::Mean_Squared_Error,in)
    this.s = in
    if this.t === nothing
        return this.s
    end
    
    return sum(((this.t-in).^2),dims=2)./size(this.t,2)
    # return sum.(((this.t-in).^2))/length(this.t)
end
function backward(this::Mean_Squared_Error,din)
    return (this.s - this.t).*(2/size(this.t,2))
end
function add_Mean_Squared_Error()
    return Mean_Squared_Error()
end
function reset(this::Mean_Squared_Error)
    this.t = nothing
    return 0
end

end