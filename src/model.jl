using Flux


struct NCAModel
    layers
    fire_rate::Float64
    alive_thresh::Float64
end
Flux.@functor NCAModel
Flux.trainable(m::NCAModel) = (m.layers,)
NCAModel(layers) = NCAModel(layers, 0.5, 0.1)

function NCAModel(n_channels::Integer; n_filters::Integer=3, hidden_size::Integer=128,
                  fire_rate::Float64=0.5, alive_thresh::Float64=0.1) 
    layers = Chain(
        Conv((3,3), n_channels => n_channels * n_filters, pad=SamePad(), groups=n_channels),
        Conv((1,1), n_channels * n_filters => hidden_size, relu),
        Conv((1,1), hidden_size => n_channels, bias=false)
    )
    NCAModel(layers, fire_rate, alive_thresh)
end

function SobelNCAModel(n_channels::Integer; hidden_size::Integer=128,
                  fire_rate::Float64=0.5, alive_thresh::Float64=0.1)
    dx = ([1, 2, 1] .* [-1, 0, 1]') ./ 8
    dy = dx'
    I = [1 0 0; 0 1 0; 0 0 1]
    w = repeat(cat(dx, dy, I, dims=3), outer=[1,1,n_channels])
    w = reshape(w, (3, 3, 1, n_channels * 3))
    w = convert(Array{Float32}, w)

    layers = Chain(
        Conv(w, zeros(Float32, 3 * n_channels), x -> x, stride=1, pad=SamePad(), groups=n_channels),
        Conv((1,1), n_channels * 3 => hidden_size, relu),
        Conv((1,1), hidden_size => n_channels, bias=false)
    )
    NCAModel(layers, fire_rate, alive_thresh)
end

function alive_mask(x, threshold; dims=2:2)
    maxpool = MaxPool((3, 3), pad=SamePad(), stride=1)
    maxpool(selectdim(x, 3, dims)) .> threshold
end

function apply_fire_rate(x, fire_rate::AbstractFloat)
    W, H, _, B = size(x)
    random_arr = rand((W, H, 1, B))
    ifelse.(random_arr .> fire_rate, x, 0)
end

function (m::NCAModel)(x; steps=1)
    D = ndims(x)
    dims, C, N = 1:D - 2, D - 1, D
    for step in 1:steps
        pre_life_mask = alive_mask(x, m.alive_thresh)

        △ = m.layers(x)

        △ = apply_fire_rate(△, m.fire_rate)
        x = x + △
        life_mask = alive_mask(x, m.alive_thresh) .&& pre_life_mask
        x = ifelse.(life_mask, x, 0.0f32)
    end
    x
end

function one_epoch(model, optim, pool::Pool{T, N}, n_samples::Integer, seed; mod_sample::Function=(x, y) -> (x,y)) where {T, N}
    loss = 0.0
    NCA.withsample(pool, n_samples) do s, t
        NCA.reset_empty!(s)
        s, t = mod_sample(s, t)
        s[:, :, :, 1] .= seed

        update, loss, grads = let y_hat
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(s, steps=rand(10:20))
                Flux.mse(y_hat[:, :, pool.c_to_comp, :], t)
            end
            y_hat, loss, grads
        end
        Flux.update!(optim, model, grads[1])
        global loss = loss
        update
    end
    loss
end


function with_one_epoch(mod_sample::Function, model, optim, pool::Pool{T, N}, n_samples::Integer, seed) where {T, N}
    one_epoch(model, optim, pool, n_samples, seed, mod_sample=mod_sample)
end