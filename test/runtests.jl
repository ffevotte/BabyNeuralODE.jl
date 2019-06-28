using BabyNeuralODE
using Test

data = read_halfmoons("../data/halfmoons.dat", 2000);

# no middle layer
train(data, 0);

# Dense middle layer
train(data, 1);

# ODE middle layer
train(data, 2);


# using Plots
#
# t1 = []
# t2 = []
# for _ in 1:5
#     push!(t1, Essais.train(data, 1))
#     plot()
#     plot!(t1, label="Dense", color=:red)
#     plot!(t2, label="ODE", color=:blue)
#     savefig("plot.pdf")
#
#     push!(t2, Essais.train(data, 2))
#     plot()
#     plot!(t1, label="Dense", color=:red)
#     plot!(t2, label="ODE", color=:blue)
#     savefig("plot.pdf")
# end
