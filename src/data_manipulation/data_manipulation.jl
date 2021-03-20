export shaping_rnn,make_sequential_batch,get_rate_of_change,standardization!,decode_standardization!,normalization!,decode_normalization

#================
データ整形
=================#
function shaping_rnn(data,N)

    X,D = size(data,1),size(data,2)
    re_data = []

    #データの補充
    # if size(data,1)%N != 0
    #     #(n,D)分，paddingで埋める
    #     n = N - size(data,1)%N
    #     padding_arr = fill(padding,n,D)
    #     data = vcat(data,padding_arr) #補充
    #     X,D = size(data)
    # end
    
    # re_data = reshape(data,Int(X/N),N,D)

    #補充じゃなく，はみ出たデータは切り捨て
    re_data = reshape(data[1:X-X%N],Int(X÷N),N,D)
    re_data = permutedims(re_data,(2,1,3)) #軸の入れ替え

    return re_data
end

function make_sequential_batch(data,N,window_size)
    #T = window * 任意の数 + 1
    T_a = size(data,1) ÷ N
    n = (T_a - 1) ÷ window_size
    T = window_size * n + 1
    D = size(data,2)
    ed = N * T #はみ出したデータは削る

    re_data = reshape(data[1:ed,:],(T,N,D))
    return permutedims(re_data,(2,1,3))  #軸の入れ替え
end

function get_rate_of_change(array)
    rate_arr = zeros(length(array)-1)
    for i in 1:length(array)-1
        rate = (array[i+1]-array[i])/array[i]
        rate_arr[i] = rate
    end
    #行列化
    rate_arr = reshape(rate_arr,(length(rate_arr),1))
    return rate_arr
end

function standardization!(model::Models,data,name)
    if haskey(model.common.std_params,name)
        avg,variance,std = model.common.std_params[name]

        return (data.-avg)/std
    end

    avg = sum(data)/length(data)
    variance = sum((data.-avg).^2)/length(data)
    std = sqrt(variance)

    std_data = (data.-avg)/std

    model.common.std_params[name] = [avg,variance,std]

    return std_data
end
function decode_standardization(model::Models,data,name)
    avg,variance,std = model.common.std_params[name]

    return data .* std .+ avg
end

function normalization!(model::Models,data,name)
    if haskey(model.common.norm_params,name)
        max,min = model.common.norm_params[name]

        return (data.-min)./(max-min)
    end

    max = findmax(data)[1]
    min = findmin(data)[1]
    
    model.common.norm_params[name] = [max,min]

    if max == min return zeros(size(data)) end

    norm_data = (data.-min)./(max-min)

    return norm_data
end
function decode_normalization(model::Models,data,name)
    max,min = model.common.norm_params[name]

    return data .* (max-min) .+ min
end