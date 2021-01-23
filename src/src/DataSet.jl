module DataSet

function two_squared(n)
    x = rand(0:100,n)*0.01
    y = rand(0:100,n)*0.01
    _t_data = [] #正解データ（上1か下0か）

    for i in 1:n
        if x[i]^2 >= y[i]
            #下にある
            append!(_t_data,0)
        else
            append!(_t_data,1)
        end
    end

    _data = hcat(x,y)

    return _data,_t_data
end

function noise_sin(n)
    data,t_data = normal_sin(n,interval=0.05)
    t_data += rand(size(t_data,1))*0.2

    return data,t_data
end

function normal_sin(n;interval=0.1)
    data = Array(-pi*2*n : interval : pi*2*n)
    t_data = sin.(data)

    return data,t_data
end


end