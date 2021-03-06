#===
・サブ配列の転置は，相手側を転置して，積を再転置する方が早い
・similar()は，計算に用いらない場合に使う（値の置換など）
===#


#===
ファイルの読み込み
===#
# include("seqential.jl")

export Affine,LSTM,Sigmoid_with_loss,Dropout

#===
共通型，関数
・全てのレイヤはLayersの子型になる
・forward!(this,x,learn_flg)
・backward!(this,din)
・gpu_flg は add_layer!()で処理してる
===#

#===
計算
・計算式をつなげる場合は@.を付ける
====#

#===
cacheは各structを作成し，保持
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
        new([W,b],[dW,db],nothing,false)
    end
end
function forward!(this::Affine,in,learn_flg)
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
function convert_to_cu!(this::Affine)
    this.params = cu.(this.params)
    this.grads = cu.(this.grads)
    this.in = nothing
    this.gpu_flg = true
    nothing
end
function convert_to_array!(this::Affine)
    this.params = Array.(this.params)
    this.grads = Array.(this.grads)
    this.in = nothing
    this.gpu_flg = false
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
    cache
    function uni_LSTM(params,grads,cache,gpu_flg)
        new(params,grads,nothing,nothing,nothing,nothing,nothing,
        nothing,nothing,nothing,gpu_flg,cache)
    end
end
function forward!(this::uni_LSTM,x,h_prev,c_prev)
    #x{N,D}, h_prev{N,H}, c_prev{N,H}
    Wx, Wh, b = this.params #{D*4H,H*4H,1*4H}
    gpu_flg = this.gpu_flg
    N = size(x,1)
    H = size(h_prev,2)
    
    @unpack A,x_Wx,h_Wh = this.cache

    #N,4H
    if A === nothing || size(A) != (N,4H)
        A = gpu_flg ? CUDA.zeros(N,4H) : zeros(N,4H)
        x_Wx = gpu_flg ? CUDA.zeros(N,4H) : zeros(N,4H)
        h_Wh = gpu_flg ? CUDA.zeros(N,4H) : zeros(N,4H)
        this.cache.A = A
        this.cache.x_Wx = x_Wx
        this.cache.h_Wh = h_Wh
    end

    mul!(x_Wx,x,Wx)
    mul!(h_Wh,h_prev,Wh)
    A .= x_Wx .+ h_Wh .+ b

    #それぞれ{N,H}
    this.f = 1 ./ (1 .+ exp.(-(view(A,:,1:H)))) #Sigmoid
    this.g = tanh.(view(A,:,H+1:2*H))
    this.i = 1 ./ (1 .+ exp.(-(view(A,:,2*H+1:3*H))))
    this.o = 1 ./ (1 .+ exp.(-(view(A,:,3*H+1:size(A,2)))))

    c_next = this.f .* c_prev .+ this.g .* this.i #N,H

    h_next = this.o .* tanh.(c_next) #N,H

    this.x = x
    this.h_prev = h_prev #N,H
    this.c_prev = c_prev #N,H
    this.c_next = c_next

    return h_next, c_next
end
function backward!(this::uni_LSTM,dh_next,dc_next)
    @unpack A = this.cache #dA

    gpu_flg = this.gpu_flg
    Wx, Wh, b = this.params
    i = this.i
    f = this.f
    g = this.g
    o = this.o

    tanh_c_next = tanh.(this.c_next)
    ds = dc_next .+ (dh_next .* o) .* (1 .- tanh_c_next.^2)

    dc_prev = ds .* f

    N,H = size(i)
    #N,4H
    if A === nothing || size(A) != (N,4H)
        A = gpu_flg ? CUDA.zeros(N,4H) : zeros(N,4H)
        this.cache.A = A
    end
    
    A[:,2H+1:3H] = ds .* g .* i .* (1 .- i)
    A[:,1:H] = ds .* this.c_prev .* f .* (1 .- f)
    A[:,3H+1:4H] = dh_next .* tanh_c_next .* o .* (1 .- o)
    A[:,H+1:2H] = ds .* i .* (1 .- g.^2)

    # this.A .= hcat(df,dg,di,d_o)

    this.grads[1] = (A' * this.x)'
    this.grads[2] = this.h_prev' * A

    this.grads[3] .= sum(A,dims=1)

    dx = A * Wx'
    dh_prev = A * Wh'

    return dx, dh_prev, dc_prev
end
function reset!(this::uni_LSTM)
    nothing
end


#====LSTM===================#
#===
Dropoutを引数に渡すと，時系列方向にも適応できる(p.265)

===#
mutable struct LSTM_cache
    A
    x_Wx
    h_Wh
    function LSTM_cache()
        new(nothing,nothing,nothing)
    end
end
mutable struct LSTM
    params #Wx,Wh,b
    grads #dWx,dWh,db
    layers #複数のLSTMレイヤを管理
    h #次のエポックに引き継ぐ
    c
    stateful
    out_sequence
    dxs
    dropout
    gpu_flg
    cache
    function LSTM(input,output;stateful=true,out_sequence=true,dropout=nothing)
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
            stateful,
            out_sequence,
            nothing,
            dropout,
            false,
            LSTM_cache()
        )
    end
end
function forward!(this::LSTM,xs,learn_flg)
    stateful = this.stateful
    gpu_flg = this.gpu_flg
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    H = size(this.params[2],1) #Wh{H,4H}

    hs = similar(xs,N,T,H)

    if !stateful || this.h === nothing || typeof(this.h)
        this.h = gpu_flg ? CUDA.zeros(N,H) : zeros(N,H)
    end
    if !stateful || this.c === nothing
        this.c = gpu_flg ? CUDA.zeros(N,H) : zeros(N,H)
    end

    #uni_LSTMの初期化
    this.layers = Array{uni_LSTM}(undef,T)

    #Dropoutのマスクのシャッフル(生成)
    if this.dropout !== nothing
        make_mask!(this.dropout,(N,H))
    end

    @inbounds for t in 1:T
        this.layers[t] = uni_LSTM(this.params,this.grads.*0,this.cache,gpu_flg)
        this.h, this.c = forward!(this.layers[t],view(xs,:,t,:),this.h,this.c)
        hs[:,t,:] = this.h #次レイヤへの伝播用
        if this.dropout !== nothing
            #dropoutをかけるのはuni_LSTMに渡す値のみ(次レイヤへは別のdropoutを使う)
            this.h = forward!(this.dropout,this.h,learn_flg,false)
            this.c = forward!(this.dropout,this.c,learn_flg,false)
        end
    end

    return this.out_sequence ? hs : hs[:,end,:] #hs{N,T,H}
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
    @inbounds for t in T:-1:1
        uni_layer = this.layers[t]
        this.dxs[:,t,:], dh, dc = backward!(uni_layer,dh,dc)
        #勾配合算 
        this.grads += uni_layer.grads
        #dropoutの誤差逆伝播
        if this.dropout !== nothing
            dh = backward!(this.dropout,dh)
            dc = backward!(this.dropout,dc)
        end
    end
    nothing
end
function backward_dims_3!(this::LSTM,dhs,T)
    dh = this.gpu_flg ? CUDA.zeros(size(dhs[:,1,:])) : zeros(size(dhs[:,1,:]))
    dc = this.gpu_flg ? CUDA.zeros(size(dhs[:,1,:])) : zeros(size(dhs[:,1,:]))
    @inbounds @simd for t in T:-1:1
        uni_layer = this.layers[t]
        this.dxs[:,t,:], dh, dc = backward!(uni_layer,view(dhs,:,t,:)+dh, dc)
        #勾配合算 
        this.grads += uni_layer.grads
        #dropoutの誤差逆伝播
        if this.dropout !== nothing
            dh = backward!(this.dropout,dh)
            dc = backward!(this.dropout,dc)
        end
    end
    nothing
end
function reset!(this::LSTM)
    this.c = nothing
    this.h = nothing
end
function convert_to_cu!(this::LSTM)
    this.params = cu.(this.params)
    this.grads = cu.(this.grads)
    this.gpu_flg = true
    this.layers = nothing
    this.dxs = nothing
    #==cache==#
    if this.cache.A !== nothing
        this.cache.A = cu(this.cache.A)
        this.cache.x_Wx = cu(this.cache.x_Wx)
        this.cache.h_Wh = cu(this.cache.h_Wh) 
    end

    if this.c !== nothing
        this.c = cu(this.c)
        this.h = cu(this.h)
    end
    if this.dropout !== nothing
        convert_to_cu!(this.dropout)
    end
    nothing
end
function convert_to_array!(this::LSTM)
    this.params = Array.(this.params)
    this.grads = Array.(this.grads)
    this.gpu_flg = false
    this.layers = nothing
    this.dxs = nothing
    #==cache==#
    if this.cache.A !== nothing
        this.cache.A = Array(this.cache.A)
        this.cache.x_Wx = Array(this.cache.x_Wx)
        this.cache.h_Wh = Array(this.cache.h_Wh)      
    end
    if this.c !== nothing
        this.c = Array(this.c)
        this.h = Array(this.h)
    end
    if this.dropout !== nothing
        convert_to_array!(this.dropout)
    end
    nothing
end


#====Sigmoid_with_loss=================#
mutable struct Sigmoid_with_loss
    params
    grads
    s #スコア
    t
    gpu_flg
    Sigmoid_with_loss() = new(nothing,nothing,nothing,nothing,false)
end
function forward!(this::Sigmoid_with_loss,in,learn_flg)
    s = 1 ./ (1 .+ exp.(-in))
    delta = 1e-7
    this.s = s
    if !learn_flg
        #ただの推論
        return s
    end
    l = -this.t .* log.(delta.+s) .- (1 .-this.t) .* log.(1+delta .-s)
    return l
end
function backward!(this::Sigmoid_with_loss,din)
    return this.s .- this.t
end
function reset!(this::Sigmoid_with_loss)
    this.t = nothing
end
function convert_to_cu!(this::Sigmoid_with_loss)
    this.gpu_flg = true
    this.s = nothing
    this.t = nothing
    nothing
end
function convert_to_array!(this::Sigmoid_with_loss)
    this.gpu_flg = false
    this.s = nothing
    this.t = nothing
    nothing
end

#====dropout===========================#
#===
・LSTMに適応する場合
    通常：LSTMの出力に，異なるmaskをかける
        LSTM
        ↑
        Dropout(0.5) dims=3
        ↑
        LSTM

    変分：LSTM内部に共通のmask，外部も共通mask(生成したmaskの[:,1,:]を使う)
    (内部でdropoutを使用したLSTMの入出力で使われる)
        LSTM(Dropout(0.5)) dims=2
        ↑
        Dropout(0.5,true) dims=2,各時刻に同じmask
        ↑
        LSTM(Dropout(0.5)) dims=2

===#
mutable struct Dropout
    params
    grads
    mask
    ratio
    gpu_flg
    variational_flg #変分=>T方向へ
    function Dropout(ratio,variational_flg=false)
        new(nothing,nothing,nothing,ratio,false,variational_flg)
    end
end
function forward!(this::Dropout,x,learn_flg,shuffle_flg=true)
    if !learn_flg
        return  x .*= (1.0 - this.ratio) #推論
    end

    #LSTM内部のT方向の処理はshuffleしない（代わりに計算前に行う）
    if shuffle_flg || size(this.mask) != size(x)
        make_mask!(this,size(x))
    end

    return x .*= this.mask #学習
end
function backward!(this::Dropout,dx)
    return dx .*= this.mask #通常
end
function reset!(this::Dropout)
    nothing
end
function convert_to_cu!(this::Dropout)
    this.gpu_flg = true
    if this.mask !== nothing
        this.mask = cu(this.mask)
    end
    nothing
end
function convert_to_array!(this::Dropout)
    this.gpu_flg = false
    if this.mask !== nothing
        this.mask = Array(this.mask)
    end
    nothing
end
function make_mask!(this::Dropout,s)
    #新しく作る
    if this.mask === nothing || s != size(this.mask)
        if this.variational_flg
            this.mask = rand(Float64,s[1],1,s[3]) #変分
        else
            this.mask = rand(Float64,s) #通常
        end
        this.mask = map(x->(x.>this.ratio ? 1.0 : 0.0),this.mask)
        if this.gpu_flg
            this.mask = cu(this.mask)
        end
    else
    #shuffle
        if this.gpu_flg
            this.mask = Array(this.mask)
        end
        this.mask = shuffle(this.mask)
        if this.gpu_flg
            this.mask = cu(this.mask)
        end
    end
end