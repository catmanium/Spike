module Spike

using Pkg
Pkg.resolve()

#===
共通パッケージ
===#

using CUDA,UnPack,CSV,DataFrames,JSON,JLD2,FileIO,ProgressMeter,Plots,IJulia,LinearAlgebra

#===
全ファイル共通
===#
abstract type Models end
abstract type Layers end
abstract type Optimizer end

#===
各種ファイルの読み込み
===#

include("models/models.jl")
include("layers/layers.jl")
include("optimizer/optimizer.jl")
include("data_manipulation/data_manipulation.jl")


end
