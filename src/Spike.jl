module Spike

    #全てのモデルファイルをincludeする ※後日

    include("Models/MLP.jl")
    include("Models/LSTM.jl")

    export MLP,LSTM


end
