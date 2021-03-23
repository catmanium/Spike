#===
・サブ配列の転置は，相手側を転置して，積を再転置する方が早い
・similar()は，計算に用いらない場合に使う（値の置換など）
===#


#===
ファイルの読み込み
===#
# include("seqential.jl")

export Affine,LSTM,Sigmoid_with_loss

#===
共通型，関数
・全てのレイヤはLayersの子型になる
===#

#=====Affine==============#
mutable struct Affine <: Layers
    params #W,b
    grads #dW,db
    in
    gpu_flg
    function Affine(input,output)
        W = randn((input,output))
        b = zeros((1,output))
        dW = zeros(size(W))
        db = zeros(size(b))
        in = zeros(output,input)
        new([W,b],[dW,db],in,false)
    end
end
function forward!(this::Affine,in)
    W,b = this.params
    this.in = in
    in * W .+ b
end
function backward!(this::Affine,din)
    # mul!(this.grads[1],(this.in)',din)
    this.grads[1] = (din' * this.in)'    
    this.grads[2] = sum(din,dims=1)
    din * this.params[1]'
end
function reset!(this::Affine)
    nothing
end


#====uni_LSTM=============#
mutable struct uni_LSTM <: Layers
    params #W,b
    grads #dW,db
    x
    h_prev
    c_prev
    i
    f
    g
    o
    c_next
    gpu_flg
    function uni_LSTM(params,grads)
        new(params,grads,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,false)
    end
end
function forward!(this::uni_LSTM,x,h_prev,c_prev)
    #x{N,D}, h_prev{N,H}, c_prev{N,H}
    Wx, Wh, b = this.params #{D*4H,H*4H,1*4H}
    N = size(x,1)
    H = size(h_prev,2)

    #N,4H
    A = x * Wx + h_prev * Wh .+ b

    #それぞれ{N,H}
    f = 1.0 ./ (1.0 .+ exp.(-(view(A,:,1:H)))) #Sigmoid
    g = tanh.(view(A,:,H+1:2*H))
    i = 1.0 ./ (1.0 .+ exp.(-(view(A,:,2*H+1:3*H))))
    o = 1.0 ./ (1.0 .+ exp.(-(view(A,:,3*H+1:size(A,2)))))

    c_next = f .* c_prev + g .* i #N,H

    h_next = o .* tanh.(c_next) #N,H

    this.x = x
    this.h_prev = h_prev #N,H
    this.c_prev = c_prev #N,H
    this.i = i
    this.f = f
    this.g = g
    this.o = o
    this.c_next = c_next

    return h_next, c_next
end
function backward!(this::uni_LSTM,dh_next,dc_next)
    Wx, Wh, b = this.params
    i = this.i
    f = this.f
    g = this.g
    o = this.o

    tanh_c_next = tanh.(this.c_next)
    ds = dc_next + (dh_next .* o) .* (1 .- tanh_c_next.^2)

    dc_prev = ds .* f

    di = ds .* g .* i .* (1 .- i)
    df = ds .* this.c_prev .* f .* (1 .- f)
    d_o = dh_next .* tanh_c_next .* o .* (1 .- o)
    dg = ds .* i .* (1 .- g.^2)

    dA = hcat(df,dg,di,d_o)

    # mul!(this.grads[1],copy(this.x'),dA)
    this.grads[1] = (dA' * this.x)'
    this.grads[2] = this.h_prev' * dA
    # mul!(this.grads[2],this.h_prev',dA)

    this.grads[3] .= sum(dA,dims=1)

    # mul!(this.dx,dA,Wx')
    # mul!(this.dh_prev,dA,Wh')
    dx = dA * Wx'
    dh_prev = dA * Wh'

    return dx, dh_prev, dc_prev
end
function reset!(this::uni_LSTM)
    nothing
end


#====LSTM===================#
mutable struct LSTM
    params #Wx,Wh,b
    grads #dWx,dWh,db
    layers #複数のLSTMレイヤを管理
    h #次のエポックに引き継ぐ
    c
    dh
    stateful
    out_sequence
    dxs
    gpu_flg
    function LSTM(input,output;stateful=true,out_sequence=true)
        Wx = randn(input,4*output)/sqrt(input)
        Wh = randn(output,4*output)/sqrt(output)
        b =  zeros(1,4*output)

        dWx = zeros(size(Wx))
        dWh = zeros(size(Wh))
        db =  zeros(size(b))
        new(
            [Wx,Wh,b],
            [dWx,dWh,db],
            nothing,
            nothing,
            nothing,
            nothing,
            stateful,
            out_sequence,
            nothing,
            false
        )
    end
end
function forward!(this::LSTM,xs)
    stateful = this.stateful
    gpu_flg = this.gpu_flg
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    H = size(this.params[2],1) #Wh{H,4H}

    hs = similar(xs,N,T,H)

    if !stateful || this.h === nothing
        this.h = gpu_flg ? CUDA.zeros(N,H) : zeros(N,H)
    end
    if !stateful || this.c === nothing
        this.c = gpu_flg ? CUDA.zeros(N,H) : zeros(N,H)
    end

    #uni_LSTMの初期化
    uni_layer = uni_LSTM(this.params,this.grads.*0)
    this.layers = fill(uni_layer,T)

    @inbounds @simd for t in 1:T
        this.h, this.c = forward!(this.layers[t],view(xs,:,t,:),this.h,this.c)
        hs[:,t,:] = this.h
    end

    return this.out_sequence ? hs : this.h #this.h{N,H}
end
function backward!(this::LSTM,dhs)
    Wx, Wh, b = this.params
    D = size(Wx,1) #{D,4H}
    N = size(dhs,1) #dhs{N,H} || {N,T,H}
    T = length(this.layers)
    this.dxs = similar(dhs,(N,T,D))
    this.grads *= 0 #勾配初期化
        
    if this.out_sequence
        backward_dims_3!(this,dhs,T) #dhs{N,T,H}
    else
        backward_dims_2!(this,dhs,T) #dhs{N,H}
    end

    return this.dxs
end
function backward_dims_2!(this::LSTM,dh,T)
    dc = this.gpu_flg ? CUDA.zeros(size(dh)) : zeros(size(dh))
    @inbounds @simd for t in T:-1:1
        uni_layer = this.layers[t]
        this.dxs[:,t,:], dh, dc = backward!(uni_layer,dh,dc)

        #勾配合算 
        this.grads += uni_layer.grads
    end
    this.dh = dh
    nothing
end
function backward_dims_3!(this::LSTM,dhs,T)
    this.dh = this.gpu_flg ? CUDA.zeros(size(dhs[:,1,:])) : zeros(size(dhs[:,1,:]))
    dc = this.gpu_flg ? CUDA.zeros(size(dhs[:,1,:])) : zeros(size(dhs[:,1,:]))
    @inbounds @simd for t in T:-1:1
        uni_layer = this.layers[t]
        this.dxs[:,t,:], this.dh, dc = backward!(uni_layer,view(dhs,:,t,:)+this.dh, dc)

        #勾配合算 
        this.grads += uni_layer.grads
    end
    nothing
end
function reset!(this::LSTM)
    this.c = nothing
    this.h = nothing
end


#====Sigmoid_with_loss=================#
mutable struct Sigmoid_with_loss
    params
    grads
    s #スコア
    t
    gpu_flg
    Sigmoid_with_loss() = new([],[],nothing,nothing,false)
end
function forward!(this::Sigmoid_with_loss,in)
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
function backward!(this::Sigmoid_with_loss,din)
    return this.s - this.t
end
function reset!(this::Sigmoid_with_loss)
    this.t = nothing
end