using Random
using Flux, MLUtils


"M == N or M == N - 1"
struct Pool{T, N} 
    states::AbstractArray{T, N}
    target::AbstractArray{T, N}
    c_to_comp
    function Pool(states::AbstractArray{T, N}, targets::AbstractArray{T, M}, c_to_comp=1) where {T, N, M}
        targets = copy(targets)
        if M == N - 1
            targets = unsqueeze(targets, dims=ndims(targets) + 1)
        end
        if size(states, ndims(states)) > size(targets, ndims(targets))
            p_size = size(states, ndims(states))
            chunks = size(targets, ndims(targets))
            target_idx = chunk(ones(Integer, p_size), chunks) .* (1:chunks)
            target_idx = [(target_idx...)...]
            # Create array of views to avoid copying memory
            targets = view(targets, [:; for i in 1:ndims(targets)-1]..., target_idx)
        end
        new{T, N}(states, targets, c_to_comp)
    end
end

function Pool(shape, targets::AbstractArray{T, N}, seed::AbstractArray{T}, c_to_comp=1) where {T, N}
    states = zeros(T, shape)
    states .= seed
    Pool(states, targets, c_to_comp)
end

function Pool(shape, targets::AbstractArray{T, N}, c_to_comp=1; seed_channels=1:1) where {T, N}
    seed = make_seed(T, shape[1:end-1], seed_channels=seed_channels)
    Pool(shape, targets, seed, c_to_comp)
end

function make_seed(type::Type, shape; seed_channels=:)
    state = zeros(type, (shape))
    middle = CartesianIndex(map(x -> ceil(Int32, x/2), shape[1:end-1]))
    state[middle, seed_channels] .= 1.0
    state
end

function make_seed(shape; seed_channels=:)
    make_seed(Float32, shape, seed_channels=seed_channels)
end

function reset_empty!(states::AbstractArray{T, N},
                        seed::AbstractArray{T, M}=make_seed(T, size(states)[1:end-1])) where {T, N, M}
    shape, samples = size(states)[1:end - 1], size(states)[end]

    z = zeros(T, shape)
    for i in 1:samples
        slice = selectdim(states, N, i)
        if slice == z
            slice .= seed
        end
    end
end

"Returns a view of a sampled pool. This means the return can ALTER the original."
function sample(pool::Pool{T, N}, n::Integer;) where {T, N}
    perm = Random.randperm(size(pool.states, 4))
    perm = perm[1:n]
    states = selectdim(pool.states, ndims(pool.states), perm)

    target = selectdim(pool.target, ndims(pool.target), perm)
    target = selectdim(target, ndims(target) - 1, pool.c_to_comp)
    states, target
end

function withsample(f, pool::Pool{T, N}, n::Integer;) where {T, N}
    states, target = sample(pool, n)
    update = f(states, target)
    states .= update
end
