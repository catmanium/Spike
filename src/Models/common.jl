include("../src/Layers.jl")

function format_model(model,layer_and_neurons,loss_layer,option)
    pre_neuron = layer_and_neurons["input"]
    for l_n in layer_and_neurons
        if l_n[1] == "input" continue end
        layer_name = l_n[1]
        neurons = l_n[2]
        func_layer = nothing

        for neuron in neurons
            func_layer = string("Layers.add_$(layer_name)($(pre_neuron),$(neuron),$(option))")
            pre_neuron = neuron
        end

        layer = eval(Meta.parse(func_layer))
        @show typeof(layer)
        append!(model.layers,layer)
        append!(model.params,[layer.params])
        append!(model.grads,[layer.grads])
    end

    append!(model.layers,eval(string("Layers.add_$(loss_layer)()")))

    return layers
end