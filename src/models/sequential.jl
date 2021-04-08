export Sequential,learn!


mutable struct Sequential <: Models
    common
    window_size
    Sequential(gpu_flg=false)=new(ModelCommon(gpu_flg),0)
end

function learn!(model::Models;data,t_data,window_size,max_epoch,verification_params=nothing,verification=nothing,notebook=false)
    D = size(data,3) #データ次元数
    T = window_size #内部ユニット数
    N = size(data,1) #バッチ数
    max_ite = size(t_data,2) #イテレーション数
    continue_flg = true
    p = Progress(max_epoch*max_ite,1,"Progress : ")
    model.common.epoch = max_epoch
    model.common.loss = zeros(max_epoch)
    model.window_size = window_size

    @inbounds for epoch in 1:max_epoch
        ite_total_loss = 0 #損失合計
        avg_loss = 0 #1エポックの平均損失
        model.common.now_epoch = epoch
        @inbounds for ite in 1:max_ite
            #ミニバッチ作成
            st = Int(1+(ite-1)*T) #data切り取り位置
            ed = Int(T*ite)
            xs = view(data,:,st:ed,:)
            t = view(t_data,:,ite,:)
            #順伝播
            model.common.layers[end].t = t #教師データ挿入
            loss = predict!(model,xs,true)
            #ite毎の平均損失を加算
            ite_total_loss += sum(loss)/length(loss)
            #逆伝播
            backward!(model)
            #更新
            update!(model.common.optimizer,model)

            next!(p)
        end

        reset!(model)

        avg_loss = ite_total_loss/max_ite
        model.common.loss[epoch] = avg_loss

        if notebook 
            print(model.common.learn_io,"\n","\e[0F","\e[2K","epoch: ",epoch," | loss: ",avg_loss," | ")
        else
            print(model.common.learn_io,"\n","epoch:",epoch," | ","loss:",avg_loss," | ")
        end

        #検証
        if verification!=nothing
            continue_flg = verification(model,verification_params)
            reset!(model)
        end

        if notebook 
            IJulia.clear_output(true)
            plot(model.common.learn_plot) |> display
        else
            print(model.common.learn_io,"\e[2F") #上(プログレスバーの先頭へ)
        end

        take!(model.common.learn_io) |> String |> println

        # min_avg_loss，その時のep,今のep,avg_loss
        if model.common.min_loss > avg_loss || epoch==1
            model.common.min_loss = avg_loss
            model.common.min_epoch = epoch
        end
       

    end

    return 0
end