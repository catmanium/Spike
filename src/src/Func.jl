module Func

function clip_grads(grads,max_norm)
    total_norm = 0
    
    for i in 1:length(grads)
        total_norm += sum(grads[i].^2)
    end
    total_norm = sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1
        for i in 1:length(grads)
            grads[i] .*= rate
        end
    end

end

function make_layers(layers)
    layers_tmp =[]

    
    return layers_tmp
end


end
