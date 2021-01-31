module Layers

using UnPack,CUDA
#=================================
・params,gradsは共通で持たせる
・add_レイヤ：コンストラクタは配列で返す．
・
==================================#

#=====Affine==============#
mutable struct Affine
    params #W,b
    grads #dW,db
    in
    Affine(params,grads) = new(params,grads,[])
end
function forward(this::Affine,in)
    this.in = in
    return in * this.params[1] .+ this.params[2]
end
function backward(this::Affine,din)
    this.grads[1] .= (this.in)' * din
    this.grads[2] .= sum(din,dims=1)
    return din * (this.params[1])'
end
function add_Affine(pre_neuron,neurons,option)
    gpu = option["GPU"]
    layers = []
    for neuron in neurons
        W = gpu ? CUDA.randn(pre_neuron,neuron) : randn(pre_neuron,neuron)
        b = gpu ? CUDA.zeros(1,neuron) : zeros(1,neuron)
        params = [W,b]
        dW = gpu ? CUDA.zeros(size(W)) : zeros(Float64,size(W))
        db = gpu ? CUDA.zeros(size(b)) : zeros(size(b))
        grads = [dW,db]
        pre_neuron = neuron
        append!(layers,[Affine(params,grads)])
    end
    return layers
end
function reset(this::Affine)
    return 0
end

#=====uni_LSTM==============#
mutable struct uni_LSTM
    params #Wx,Wh,b
    grads #dWx,dWh,db
    cache #x, h_prev, h_next,i,f,g,o,c_next
    uni_LSTM(params,grads) = new(params,grads,[])
end
function forward(this::uni_LSTM,x,h_prev,c_prev)
    Wx, Wh, b = this.params
    N, H = size(h_prev)
    A = x * Wx + h_prev * Wh .+ b

    #slice
    f = A[:, begin:H]
    g = A[:, H+1:2*H]
    i = A[:, 2*H+1:3*H]
    o = A[:, 3*H+1:end]

    f = 1.0 ./ (1.0 .+ exp.(-f)) #Sigmoid
    g = tanh.(g)
    i = 1.0 ./ (1.0 .+ exp.(-i))
    o = 1.0 ./ (1.0 .+ exp.(-o))

    c_next = f .* c_prev + g .* i

    h_next = o .* tanh.(c_next)

    this.cache = [x, h_prev, c_prev, i, f, g, o, c_next]

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

    dWh = h_prev' * dA
    dWx = x' * dA
    db = sum(dA,dims=1)

    this.grads[1] .= dWx
    this.grads[2] .= dWh
    this.grads[3] .= db

    dx = dA * Wx'
    dh_prev = dA * Wh'

    return dx, dh_prev, dc_prev
end
function add_uni_LSTM(pre_neuron,neurons,option)
    gpu = option["GPU"]
    layers = []
    for neuron in neurons
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

#====LSTM===================#
mutable struct LSTM
    params #Wx,Wh,b
    grads #dWx,dWh,db
    layers #複数のRNNレイヤを管理
    h #次のエポックに引き継ぐ
    c
    dh
    option
    LSTM(params,grads,option) = new(
        params,
        grads,
        nothing,
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
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    H = size(Wh,1)

    this.layers = []
    hs = GPU ? CUDA.zeros((N,T,H)) : zeros(Float64,(N,T,H))

    if !stateful || this.h === nothing
        this.h = GPU ? CUDA.zeros((N,H)) : zeros(Float64,(N,H))
    end

    if !stateful || this.c === nothing
        this.c = GPU ? CUDA.zeros((N,H)) : zeros(Float64,(N,H))
    end

    for t in 1:T
        rnn_layer = uni_LSTM(Wx,Wh,b)
        this.h, this.c = forward(rnn_layer,xs[:,t,:],this.h,this.c)
        hs[:,t,:] = this.h
        push!(this.layers,rnn_layer)
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
        
        dh = copy(dhs)
        dc = gpu ? CUDA.zeros(size(dh)) : zeros(Float64,size(dh))
        
        for t in T:-1:1
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
        for t in T:-1:1
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
    for neuron in neurons
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
#====Sigmoid_with_loss=================#
mutable struct Sigmoid_with_loss
    params
    grads
    s::Array #スコア
    t::Array
    Sigmoid_with_loss() = new([],[],[],[])
end
function forward(this::Sigmoid_with_loss,in)
    s = 1.0 ./ (1.0 .+ exp.(-in))
    this.s = s
    if length(this.t) == 0
        #ただの推論
        return s
    end
    l = -this.t .* log.(s) - (1 .-this.t) .* log.(1 .-s)
    return l
end
function backward(this::Sigmoid_with_loss,din)
    return this.s - this.t
end
function add_Sigmoid_with_loss()
    return Sigmoid_with_loss()
end
function reset(this::Sigmoid_with_loss)
    this.t = []
end

end