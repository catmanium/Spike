module Spike
    using CUDA,UnPack
    #全てのモデルファイルをincludeする ※後日

    include("Models/MLP.jl")
    include("Models/LSTM.jl")

    include("dataset/noise_sin.jl")
    include("dataset/shaping_data.jl")


end
