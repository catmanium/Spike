#================
データ整形
=================#
function shaping_rnn(data,N)
    #re_data (N, X/N, D)
    #ite毎に切り出して使用する
    #Tは2軸目のサイズに応じて後から決める

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

function standardization(data,disp=false)
    avg = sum(data)/length(data)
    variance = sum((data.-avg).^2)/length(data)
    std = sqrt(variance)

    data_norm = (data.-avg)/std

    if disp
        println("avg : $avg")
        println("variance : $variance")
        println("standard deviation : $std")
    end

    return data_norm
end