include("common.jl")

mutable struct LSTM_struct <: Sequence
    params
    grads
    layers
    optimizer
    padding
    LSTM_struct() = new([],[],[],nothing,nothing)
end

function LSTM(;layer_and_neurons=[],loss_layer=nothing,optimizer="Adam",option=Dict())
    #デフォルト
    m_option = Dict{String,Any}()
    m_option["stateful"] = haskey(option,"stateful") ? option["stateful"] : true
    m_option["out_sequence"] = haskey(option,"out_sequence") ? option["out_sequence"] : false
    m_option["padding"] = haskey(option,"padding") ? option["padding"] : false
    m_option["GPU"] = haskey(option,"GPU") ? option["GPU"] : false

    model = LSTM_struct()
    format_model(model,layer_and_neurons,loss_layer,m_option)
    return model
end

