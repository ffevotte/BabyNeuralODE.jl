module BabyNeuralODE

export read_halfmoons, train

using Flux
using Random

function ode_solve(fun, u, t1, N)
    δt = t1/N
    for _ in 1:N
        u += fun(u)*δt
    end
    u
end

function read_halfmoons(fname, n)
    to_float(x) = parse(Float64, x)
    open(fname) do f
        map(readlines(f)) do l
            (x1, x2, y) = split(l) .|> to_float
            ([x1, x2], y)
        end
    end
end

function test(m, data)
    acc = 0
    for (x, y) in data
        (p0, p1) = m(x)
        yy = p0>p1 ? 0 : 1
        if yy == y
            acc += 1
        end
    end
    acc / length(data)
end

function train(data, typ)
    Random.shuffle!(data)
    ndata = length(data)
    ntrain = Int(round(0.8*ndata))
    ntest = ndata - ntrain

    nhidden = 16

    w2 = param(zeros(nhidden, nhidden))
    b2 = param(zeros(nhidden))
    odelayer(x) = ode_solve(x->(w2*x .+ b2), x, 1, 10)

    if typ == 1
        layer2 = Dense(nhidden, nhidden, σ)
    elseif typ == 2
        layer2 = odelayer
    else
        layer2 = x->x
    end

    m = Chain(
        Dense(2, nhidden, σ),
        layer2,
        Dense(nhidden, 2, σ),
        softmax
    )

    function loss(x, y)
        y = Flux.onehot(y, 0:1)
        Flux.crossentropy(m(x), y)
    end

    pm = Flux.params(m, w2, b2)
    opt = Flux.Optimise.ADAM()

    println("epoch: ",  0,
            "\tloss: ", sum(loss(d...) for d in data),
            "\ttest: ", test(m, data))

    batchsize = 400

    acc = Float64[]
    for epoch in 1:100
        Random.shuffle!(@view data[1:ntrain])

        i = 1
        while i<ntrain
            Flux.train!(loss, pm, data[i:(i+batchsize-1)], opt)
            i += batchsize
        end

        t = test(m, data[ntrain+1:end])
        println("epoch: ",  epoch,
                "\tloss: ", sum(loss(data[i]...) for i in 1:ntrain),
                "\ttest: ", t)
        push!(acc, t)
        t > 0.999 && break
    end

    acc
end

end # module
