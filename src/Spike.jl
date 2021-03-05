module Spike
    using Pkg
    Pkg.resolve()

    using CUDA,UnPack,CSV,DataFrames,JSON,PyCall,JLD2,FileIO,ProgressMeter,Plots,IJulia

    #全てのモデルファイルをincludeする ※後日

    include("Models/MLP.jl")
    include("Models/LSTM.jl")

    include("dataset/noise_sin.jl")
    include("dataset/shaping_data.jl")


end
