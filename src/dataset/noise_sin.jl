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