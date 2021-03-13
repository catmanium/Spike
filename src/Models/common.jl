include("../src/Layers.jl")
include("../src/Optimizer.jl")

abstract type Sequence end
abstract type not_Sequence end

function format_model(model,layer_and_neurons,loss_layer,option)
    pre_neuron = layer_and_neurons[1][2] #input
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

    return true
end

function reset(model)
    for layer in model.layers
        Layers.reset(layer)
    end
end

function predict(model,data)
    if model.option["GPU"]
        data = cu(data)
    end

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


#===============================================
verification(model,struct)の形にする
    保存したい値はstructへ
    返り値は学習終了の判断に使うbool

    表示を追加する場合は，モデルにIOBuffer()を追加
    そこにverification()内で文字列を追加
================================================#
function learn(model::Sequence;max_epoch,window_size,data,t_data,verification=nothing,verification_params=nothing,notebook=false)
    if model.option["GPU"]
        data = cu(data)
        t_data = cu(t_data)
    end

    D = size(data,3) #データ次元数
    T = window_size #RNNレイヤ数
    N = size(data,1) #バッチ数
    max_ite = size(data,2)÷T #イテレーション数
    min_avg_loss = 0 #最小損失
    min_epoch = 0
    continue_flg = true
    p = Progress(max_epoch*max_ite,1,"Progress : ")

    model.max_epoch = max_epoch
    model.loss = zeros(max_epoch)
    model.window_size = window_size

    println("epoch: 0 | loss: 0 | ")
    
    for epoch in 1:max_epoch
        ite_total_loss = 0 #損失合計
        avg_loss = 0 #1エポックの平均損失
        st = 0 #data切り取り位置
        ed = 0
        model.epoch = epoch
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
            #更新
            Optimizer.update(model.optimizer,model)

            next!(p)
        end

        reset(model)

        avg_loss = ite_total_loss/max_ite
        model.loss[epoch] = avg_loss

        print(model.learn_io,"\n","\e[0F","\e[2K","epoch: $(epoch) | loss: $(avg_loss) | ")

        #検証
        if verification!=nothing
            continue_flg = verification(model,verification_params)
            reset(model)
        end

        if notebook 
            IJulia.clear_output(true)
            plot(model.learn_plot) |> display
        else
            print("\e[2K","\e[1F","\e[2K")
        end
        take!(model.learn_io) |> String |> println

        # min_avg_loss，その時のep,今のep,avg_loss
        if min_avg_loss > avg_loss || epoch==1
            min_avg_loss = avg_loss
            min_epoch = epoch
        end
       

    end

    return 0
end

function model_save(model,path="")
    #paramsとnorm,stdのみ
    d = Dict(
    "params" => Array.(model.params),
    "std_params" => model.std_params,
    "norm_params" => model.norm_params
    )
    
    save("$path","model",d)

end

function model_load(model,path)
    gpu = model.option["GPU"]
    d = load("$path")
    for i in 1:length(model.params)
        model.params[i] .= gpu ? cu(d["model"]["params"][i]) : d["model"]["params"][i]
    end
    model.std_params = d["model"]["std_params"]
    model.norm_params = d["model"]["norm_params"]
end

function disp(model,notebook)

    
end