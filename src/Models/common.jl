include("../src/Layers.jl")
include("../src/Optimizer.jl")

abstract type Sequence end
abstract type not_Sequence end

function format_model(model,layer_and_neurons,loss_layer,option)
    pre_neuron = layer_and_neurons["input"]
    for l_n in layer_and_neurons
        if l_n[1] == "input" continue end
        layer_name = l_n[1]
        neurons = l_n[2]
        func_layer = nothing
        
        func_layer = string("Layers.add_$(layer_name)($(pre_neuron),$(neurons),$(option))")
        pre_neuron = neurons[end]       
       
        layers = eval(Meta.parse(func_layer))
        append!(model.layers,layers)

        for layer in layers
            append!(model.params,layer.params)
            append!(model.grads,layer.grads)
        end
    end

    func_loss_layer = string("Layers.add_$(loss_layer)()")
    push!(model.layers,eval(Meta.parse(func_loss_layer)))

    if option["GPU"]
        model.params = cu.(model.params)
        model.grads = cu.(model.grads)
        for i in 1:length(model.layers)
            model.layers[i].params = cu.(model.layers[i].params)
            model.layers[i].grads = cu.(model.layers[i].grads)
        end
    end

    return true
end

function reset(model)
    for layer in model.layers
        Layers.reset(layer)
    end
end

function predict(model,data)
    out = data
    for layer in model.layers
        out = Layers.forward(layer,out)
    end
    return out
end

function backward(model)
    dout = 0
    for layer in reverse(model.layers)
        dout = Layers.backward(layer,dout)
    end
end

function learn(model::Sequence;max_epoch,window_size,data,t_data)
    D = size(data,3) #データ次元数
    T = window_size #RNNレイヤ数
    N = size(data,1) #バッチ数
    max_ite = size(data,2)÷T #イテレーション数
    loss_list = [] #avg_lossのリスト
    
    for epoch in 1:max_epoch
        ite_total_loss = 0 #損失合計
        avg_loss = 0 #1エポックの平均損失
        st = 0 #data切り取り位置
        ed = 0
        for ite in 1:max_ite
            #ミニバッチ作成
            st = Int(1+(ite-1)*T)
            ed = Int(T*ite)
            xs = data[:,st:ed,:]
            t = t_data[:,ite,:]
            #順伝播
            model.layers[end].t = t #教師データ挿入
            loss = predict(model,xs)
            #ite毎の平均損失を加算
            ite_total_loss += sum(loss)/length(loss)
            #逆伝播
            backward(model)
            #勾配クリッピング
            # append!(grads_list,this.grads[1][1])
            #更新
            Optimizer.update(model.optimizer,model)
        end

        avg_loss = ite_total_loss/max_ite
        append!(loss_list,avg_loss)
       
        if epoch == 1 || epoch%(max_epoch÷10) == 0
            println("ep.$epoch : Loss :　",avg_loss)
        end
    end

    reset(model)

    return loss_list
end

