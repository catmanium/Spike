export predict!,backward!,add_layer!,add_optimizer!,get_layers,get_params,get_loss,get_now_epoch,save_model,load_model,reset!
#===
共通型，関数
・全てのモデルはModelCommonをcommonに持ち，Modelsの子型になる
===#

mutable struct ModelCommon
    layers
    optimizer
    epoch
    now_epoch
    min_epoch
    learn_io
    learn_plot
    loss
    min_loss
    std_params #Dict("name"=>[avg,variance,std])
    norm_params #Dict("name"=>[max,min])
    gpu_flg

    function ModelCommon(gpu_flg)
        new(
            nothing,
            nothing,
            0,
            0,
            0,
            IOBuffer(),
            nothing,
            [],
            0.0,
            Dict(),
            Dict(),
            gpu_flg
        )
    end
end

#=========
dataは全て{N,D}の形でここに渡す.
cuは引数の前にしておく
=========#
function predict!(model::Models,data,learn_flg=false)
    out = data

    @inbounds @simd for layer in model.common.layers
        out = forward!(layer,out,learn_flg)
    end

    return out
end

function backward!(model::Models)
    dout = 0
    @inbounds @simd for layer in reverse(model.common.layers)
        dout = backward!(layer,dout)
    end
end

function add_layer!(model::Models,layers)
    if model.common.gpu_flg
        @inbounds for i in 1:length(layers)
            convert_to_cu!(layers[i])
        end
    end
    model.common.layers = layers
    nothing
end

function add_optimizer!(model::Models,optimizer)
    fit!(optimizer,model)
end

function reset!(model::Models)
    @inbounds @simd for i in 1:length(model.common.layers)
        reset!(model.common.layers[i])
    end
end

function get_layers(model::Models,n=nothing)
    (n===nothing) ? model.common.layers : model.common.layers[n]
end
function get_now_epoch(model::Models)
    model.common.now_epoch
end
function get_loss(model::Models)
    model.common.loss
end
function get_params(model::Models,n=nothing)
    if n===nothing
        re = []
        @inbounds for i in 1:length(model.common.layers)
            @simd for j in 1:length(model.common.layers[i].params)
                append!(re,[model.common.layers[i].params[j]])
            end
        end
        return re
    else
        return model.common.layers[n].params
    end
end

function save_model(model::Models,path)
    #パラメータ全てをArrayに直す必要がある
    @inbounds for i in 1:length(model.common.layers)
        convert_to_array!(model.common.layers[i])
    end
    save(path,"model",Dict("content"=>model))
    nothing
end
function load_model(path,gpu_flg=false)
    d = load(path)
    model = d["model"]["content"]
    if gpu_flg
        @inbounds for i in 1:length(model.common.layers)
            convert_to_cu!(model.common.layers[i])
        end
    end
    return model
end
#===
ファイルの読み込み
===#
include("sequential.jl")