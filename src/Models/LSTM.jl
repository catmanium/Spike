include("../src/Layers.jl")

mutable struct LSTM_struct
    params
    grads
    layers
    optimizer
    padding
end

function LSTM(layer_and_neurons,loss,out_sequences)
    layers = []

    #レイヤの生成
    pre_neuron = layer_and_neurons["input"]
    for l_n in layer_and_neurons[2:end] #inputの除外
        layer = l_n[1]
        neurons = l_n[2]

        for neuron in neurons
            func_layer = string("Layers.add_$(layer)($(pre_neuron),$(neuron))")
            pre_neuron = neuron
        end

        append!(layers,eval(func_layer))
    end
end