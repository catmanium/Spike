include("common.jl")

mutable struct LSTM_struct
    params
    grads
    layers
    optimizer
    padding
    LSTM_struct() = new([],[],[],nothing,nothing)
end

function LSTM(;layer_and_neurons=[],loss_layer=nothing,option=Dict())
    if isempty(option)
        #デフォルト設定
        option = Dict(
            "stateful" => false,
            "out_sequence" => false,
            "padding" => nothing
        )
    end
    model = LSTM_struct()
    format_model(model,layer_and_neurons,loss_layer,option)
    return model
end