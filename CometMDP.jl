using DifferentialEquations
using LinearAlgebra: norm
using LinearAlgebra: dot
using LinearAlgebra: cross
using Plots
using Distributions
using Flux
using Zygote
using IJulia
using VegaLite
using ReinforcementLearning

###################
# Comet Environment
###################

struct CometEnvParams{T}
    # Comet Specific Parameters
    N::Int # Number of mascons
    M::T # Mass of comet (kg)
    Rcomet::T # Radius of comet (m)
    Rcoma::T # Radius of coma (m)
    mascons::Vector{Vector{T}} # mascons
    hull::Vector{Vector{T}} # Surface of comet
    # Spacecraft Specific Parameters
    max_propellant_mass::T # Propellant mass (kg)
    dry_mass::T # Dry mass (kg)
    Isp::T # Specific impulse (s)
    # Time interval and max steps
    dt::Int
    H::Int
end

Base.show(io::IO, params::CometEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(CometEnvParams)], ","),
)

# Stores comet environment parameters, values can be changed
function CometEnvParams(;
    T=Float64,
    N = 100,
    M = 1.0*10^16,
    Rcomet = 1000.0,
    Rcoma = 50000.0)
    mascons = [rand(Normal(0f0,Rcomet),2) for _ in 1:N]
    hull = create_hull(mascons)
    max_propellant_mass = 10.0
    dry_mass = 140.0
    Isp = 230.0
    dt = 50
    H = 400
    CometEnvParams{T}(
        N,
        M,
        Rcomet,
        Rcoma,
        mascons,
        hull,
        max_propellant_mass,
        dry_mass,
        Isp,
        dt,
        H,
    )
end

# Struct for the environment
mutable struct CometEnv{T} <: AbstractEnv
    params::CometEnvParams{T} 
    state::Vector{T} # Agents state
    action::Vector{T} # Agents most recent action
    done::Bool # Is the state terminal
    t::Int # Elapsed time
end

# Creates comet environment
function CometEnv(;T=Float64)
    params = CometEnvParams(;T=T)
    env = CometEnv(params, zeros(T,5), zeros(T,2), false, 0)
    reset!(env)
    env
end

# State space
function RLBase.state_space(env::CometEnv)
    (-Inf .. Inf) × (-Inf .. Inf) × (-Inf .. Inf) × (-Inf .. Inf) × (0.0 .. env.params.max_propellant_mass) × (0.0 .. env.params.H)
end

# Action space, max ΔV is restricted to 1/100 of available ΔV
function RLBase.action_space(env::CometEnv)
    dV = 9.81*env.params.Isp*log((env.params.dry_mass+env.params.max_propellant_mass)/env.params.dry_mass)
    (0.0 .. dV/100)
end

# Reward function
function RLBase.reward(env::CometEnv)
    s = env.state
    a = env.action
    m0 = env.params.dry_mass+env.params.max_propellant_mass - s[5] # Current spacecraft mass
    dV = norm(a) # ΔV for the action
    ve = 9.81*env.params.Isp # Exhaust velocity
    m_consumed = m0 - m0/exp(dV/ve) # Δm
    if norm(s[1:2]) <= env.params.Rcoma
        # If s/c is inside coma, positive reward
        return -m_consumed/env.params.max_propellant_mass + 4/env.params.H
    else
        return -m_consumed/env.params.max_propellant_mass
    end
end

# Checks if state is terminal
function RLBase.is_terminated(env::CometEnv)
    env.done
end

# Get state
function RLBase.state(env::CometEnv)
    env.state
end

# Resets environment
function RLBase.reset!(env::CometEnv)
    θ = 2*π*rand() # Agent is initialized to random point on the coma
    x = env.params.Rcoma*cos(θ)
    y = env.params.Rcoma*sin(θ)
    pos = [x ; y]
    vel = [0.0,0.0] # Initial velocity is 0
    env.state = [pos ; vel ; 0.0 ; 0.0] # Initial consumed propellant mass and time are 0
    env.done = false
    env.t = 0.0
    nothing
end

function RLBase.act!(env::CometEnv, a)
    env.action = a
    _step!(env, a)
end

# Advances state based on action
function _step!(env,a)
    env.t += env.params.dt
    env.state[6] += env.params.dt # Updates t
    env.state[5] += mass_update(env,a) # Updates mass
    env.state[1:4] = coord_update(env,a) # Updates coordinates
    # Checks if state is terminal
    if is_inside(env.params.hull,env.state[1:2]) || env.state[6] >= env.params.H*env.params.dt || env.state[5] >= env.params.max_propellant_mass
        env.done = true
    end
    nothing
end

# Updates mass using rocket equation
function mass_update(env,a)
    s = env.state
    Isp = env.params.Isp # Specific impulse (s)
    ve = 9.81*Isp # Exhaust velocity (m/s)
    dV = norm(a) # Delta-V (m/s)
    m0 = env.params.dry_mass + env.params.max_propellant_mass - s[5]
    return m0*(1-1/exp(dV/ve)) # Updated mass (kg)
end

# Differential equation for propagating state forward
function mascon_dynamics(Xk,comet,t)
    G = 6.67430*10^-11
    mascons,M = comet[1], comet[2]
    μ = G*M/length(mascons)
    a = [0.0,0.0]
    pos = Xk[1:2]
    # Sums over all mascons to compute acceleration
    for i in eachindex(mascons)
        r = mascons[i] - pos
        a += μ*r/norm(r)^3
    end
    return [Xk[3],Xk[4],a[1],a[2]]
end

function prop_traj(env,DE,X0)
    tspan = (0.0,env.params.dt)
    p = (env.params.mascons,env.params.M)
    traj = ODEProblem(DE,X0,tspan,p)
    sol = solve(traj)
    return sol
end

function coord_update(env,a)
    Xk = env.state[1:4]
    Xk[3:4] += a
    sol = prop_traj(env,mascon_dynamics,Xk)
    return sol.u[end]
end

# 2D cross product for hull wrapping algorithm
function direction(a,b,c)
    result = (b[1]-a[1])*(c[2]-a[2])-(c[1]-a[1])*(b[2]-a[2]) # Cross product
    return result < 0
end

# Checks if agents position is inside the comet using ray casting
function is_inside(hull,r)
    xp,yp = r[1],r[2]
    count = 0
    for i in eachindex(hull)
        if i == length(hull)
            x1,y1 = hull[i][1],hull[i][2]
            x2,y2 = hull[1][1],hull[1][2]
        else
            x1,y1 = hull[i][1],hull[i][2]
            x2,y2 = hull[i+1][1],hull[i+1][2]
        end
        if y2 != y1
            if (yp < y1) != (yp < y2) && xp < (x1 + ((yp-y1)/(y2-y1))*(x2-x1))
                count += 1
            end
        end
    end
    return count%2 == 1
end

# Wraps mascons in a surface
function create_hull(coords)
    leftmost_x,index = findmin(coord[1] for coord in coords)
    leftmost_pt = coords[index]
    current_pt = leftmost_pt
    hull = Vector{Vector{Float64}}()
    while true
        push!(hull,current_pt)
        j = rand(1:10)
        if coords[j] == current_pt
            if j == 10
                j += -1
            else
                j += 1
            end
        end
        next_pt = coords[j]
        for check_pt in coords
            if direction(current_pt,next_pt,check_pt)
                next_pt = check_pt
            end
        end
        j = findfirst(==(current_pt), coords)
        current_pt = next_pt
        if current_pt == hull[1]
            break
        end
    end
    return hull
end

# Creates plot of comet environment and trajectories
function create_plot(env::CometEnv,τs)
    coords,hull,coma = env.params.mascons,env.params.hull,env.params.Rcoma
    coords_plot = zeros(length(coords),2)
    hull_plot = zeros(length(hull)+1,2)
    coma_plot = zeros(1000,2)
    hull_plot[end,:] += hull[1]

    for i in eachindex(coords)
        coords_plot[i,:] += coords[i]
    end

    for i in eachindex(hull)
        hull_plot[i,:] += hull[i]
    end

    for i in 1:1000
        θ = 0.001*i
        coma_plot[i,1] += coma*sin(2*π*θ)
        coma_plot[i,2] += coma*cos(2*π*θ)
    end

    p = scatter(xlabel="x (m)", ylabel="y (m)")
    scatter!(p,coords_plot[:,1],coords_plot[:,2],label="mascons")
    plot!(p,coma_plot[:,1],coma_plot[:,2],label="coma")
    for τ in τs
        Xs = zeros(length(τ),2)
        for i in eachindex(τ)
            Xs[i,:] += τ[i][1][1:2]
        end
        plot!(p,Xs[:,1],Xs[:,2],label=false)
        scatter!(p,[Xs[end,1]],[Xs[end,2]],markershape=:x,mc=:red,label=false)
    end
    display(p)
end


function heuristic_policy()
    # Only works for postion Rcoma*[cos(0),sin(0)]
    return [-1, 1.5]
end

##############################
# Proximal Policy Optimization
##############################

function select_action(s,max_ΔV,π_net)
    # Get mean and variance for ΔV and θ
    X = abs.(π_net(s))
    μ_ΔV,σ_ΔV,μ_θ,σ_θ = X[1],X[2],X[3],X[4]

    dist_ΔV = Normal(μ_ΔV,σ_ΔV)
    dist_θ = Normal(μ_θ,σ_θ)

    # Abs random samples so action is legal
    ΔV = abs(rand(dist_ΔV))*max_ΔV
    θ = abs(rand(dist_θ))*2*π

    # Return action
    return [ΔV,θ]
end

# Runs trajectory rollouts
function ac_trajectory(env,π_net)
    reset!(env)
    τ = []
    while !is_terminated(env)
        max_ΔV = maximum(action_space(env))
        s = state(env)
        a = select_action(s,max_ΔV,π_net)
        r = reward(env)
        push!(τ,(s[1:6],a,r))
        act!(env,a[1]*[cos(a[2]),sin(a[2])])
    end
    return τ
end

# Best trajectory takes the mean of the ΔV and θ distributions, policy is no longer stochastic
function best_trajectory(env,π_net)
    reset!(env)
    τ = []
    while !is_terminated(env)
        max_ΔV = maximum(action_space(env))
        s = state(env)
        X = abs.(π_net(s))
        ΔV = abs(X[1])*max_ΔV
        θ = abs(X[3])*2*π
        a = ΔV*[cos(θ),sin(θ)]
        r = reward(env)
        push!(τ,(s[1:6],a,r))
        act!(env,a)
    end
    return τ
end

# Update policy and value networks
function run_epochs(env,π_net,V_net,α,ϵ,epochs,trajectories)
    # Run trajectories
    τs = [ac_trajectory(env,π_net) for _ in 1:trajectories]
    step = [length(τ) for τ in τs]
    data_V = Vloss_data(τs)
    data_ΔV = Vobj_data(τs,π_net,V_net,ϵ,env)

    # Copy old policy network
    πp_net = π_net

    Losses = Vector{Float32}()
    Rewards = Vector{Float32}()
    Steps = Vector{Int64}()
    for i in 1:epochs
        if i == 1
            push!(Steps,sum(step))
        else
            push!(Steps,sum(step)+Steps[end])
        end
        # Updates value network every other epoch
        if i%2 == 1
            Flux.Optimise.train!(Vloss,V_net, data_V, Flux.setup(ADAM(α), V_net))
        end
        push!(Losses,avg_loss(V_net,data_V))
        push!(Rewards,evaluate_policy(env,πp_net,20))
        Flux.Optimise.train!(Vobj, πp_net, data_ΔV, Descent(-α))
    end
    return πp_net,V_net,Losses,Rewards,Steps
end

# Runs training loops
function train_networks(env,π_net,V_net,α,ϵ,epochs,trajectories,training_loops)
    Loss_Curve = []
    Reward_Curve = []
    Total_Steps = Vector{Float32}()
    for i in 1:training_loops
        π_net,V_net,Losses,Rewards,Steps = run_epochs(env,π_net,V_net,α,ϵ,epochs,trajectories)
        Loss_Curve = [Loss_Curve ; Losses]
        Reward_Curve = [Reward_Curve ; Rewards]
        if i == 1
            Total_Steps = [Total_Steps ; Steps]
        else
            Total_Steps = [Total_Steps ; Steps .+ Total_Steps[end]]
        end
        println(evaluate_policy(env,π_net,10))
        println(i)
    end
    return π_net,V_net,Loss_Curve,Reward_Curve,Total_Steps
end

# Loss function for value network
function Vloss(V_net,s,r)
    L = 0.0
    γ = 0.99
    for i in eachindex(r)
        # Mean squared error
        L += ((V_net(s[:,i])[1] - sum(γ^(k-1)*r[k] for k in eachindex(r[i:end])))^2)/2
    end
    return L
end

# Format value network data for train!
function Vloss_data(τs)
    data = Vector{Tuple{Matrix{Float32}, Vector{Float32}}}()
    for τ in τs
        s_matrix = zeros(Float32,6,length(τ))
        r_vector = Vector{Float32}()
        for i in eachindex(τ)
            s_matrix[:,i] += τ[i][1]
            push!(r_vector,τ[i][3])
        end
        push!(data,(s_matrix,r_vector))
    end
    return data
end

# Compute average loss over the dataset
function avg_loss(V_net,data)
    L = 0.0
    for i in eachindex(data)
        s,r = data[i][1],data[i][2]
        L += Vloss(V_net,s,r)
    end
    return L/length(data)
end

# Evaluate the current policy, takes mean of ΔV and θ
# Computes average discounted reward over N trajectories
function evaluate_policy(env,π_net,N)
    total_reward = 0
    γ = 0.99
    for _ in 1:N
        reset!(env)
        R = 0
        k = 0
        while !is_terminated(env)
            max_ΔV = maximum(action_space(env))
            s = state(env)
            X = abs.(π_net(s))
            ΔV = abs(X[1])*max_ΔV
            θ = abs(X[3])*2*π
            r = reward(env)
            act!(env,ΔV*[cos(θ),sin(θ)])
            R += γ^(k)*r
            k += 1
        end
        total_reward += R
    end
    return total_reward/N
end

# Objective function for PPO
function Vobj(πp_net,s,a,πθ,Aθ,ϵ,scale)
    Lclip = 0.0
    for i in eachindex(πθ)
        Xp = abs.(πp_net(s[:,i]))
        distps = (Normal(Xp[1],Xp[2]),Normal(Xp[3],Xp[4]))
        # probability of taking an action is the mutliple of the individual probability distributions of ΔV and θ
        πθp = pdf(distps[1],a[1,i]/scale[1,i])*pdf(distps[2],a[2,i]/scale[2,i])
        rθ = πθp/πθ[i]
        Lclip += min(rθ*Aθ[i], clamp(rθ,1-ϵ[i],1+ϵ[i])*Aθ[i])
    end
    return Lclip
end

# Format data for PPO objective function
function Vobj_data(τs,π_net,V_net,ϵ,env)
    γ = 0.99
    max_ΔV = maximum(action_space(env))
    scale_vector = [max_ΔV,2*π]
    data = Vector{Tuple{Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32}, Matrix{Float64}}}()
    for τ in τs
        s_matrix = zeros(Float32,6,length(τ))
        a_matrix = zeros(Float32,2,length(τ))
        πθ_vector = Vector{Float32}()
        Aθ_vector = Vector{Float32}()
        ϵ_vector = Vector{Float32}()
        # k_vector = Vector{Int64}()
        scale_matrix = zeros(Float32,2,length(τ))
        for i in eachindex(τ)
            s_matrix[:,i] += τ[i][1]
            a_matrix[:,i] += τ[i][2]
            X = abs.(π_net(τ[i][1]))
            dists = [Normal(X[1],X[2]),Normal(X[3],X[4])]
            push!(πθ_vector,pdf(dists[1],τ[i][2][1]/scale_vector[1])*pdf(dists[2],τ[i][2][2]/scale_vector[2]))
            if i == length(τ)
                # push!(Aθ_vector,sum(r for (s,a,r) in τ[i:end]) - V_net(τ[i][1])[1])
                push!(Aθ_vector,τ[i][3] - V_net(τ[i][1])[1])
            else
                # push!(Aθ_vector,sum(r for (s,a,r) in τ[i:end]) - V_net(τ[i][1])[1])
                push!(Aθ_vector,τ[i][3] + γ*V_net(τ[i+1][1])[1] - V_net(τ[i][1])[1])
            end
            push!(ϵ_vector,ϵ)
            scale_matrix[:,i] += scale_vector
        end
        push!(data,(s_matrix,a_matrix,πθ_vector,Aθ_vector,ϵ_vector,scale_matrix))
    end
    return data
end

env = CometEnv(T=Float32)
s = state(env)

π_net = Chain(
            Dense(6, 30, sigmoid),
            Dense(30, 30, sigmoid),
            Dense(30, 4),
            )

V_net = Chain(
            Dense(6, 30, sigmoid),
            Dense(30, 30, sigmoid),
            Dense(30, 1),
            )

# Hyperparameters
ϵ = 0.1
α = 0.0001
epochs = 1
trajectories = 1
training_loops = 10

# Evaluate starting policy
println(evaluate_policy(env,π_net,40))

# Train the networks
π_net,V_net,Loss_Curve,Reward_Curve,Total_Steps = train_networks(env,π_net,V_net,α,ϵ,epochs,trajectories,training_loops)

# Plot learning curves and trajectories
total_epochs = [i for i in 1:Int(epochs*training_loops)]
p = plot(xlabel="Total Epochs",ylabel="Value Network Loss")
plot!(p,total_epochs,Loss_Curve,label=false)
display(p)
q = plot(xlabel="Total Steps",ylabel="Cumulative Reward")
plot!(q,Total_Steps,Reward_Curve,label=false)
display(q)
τs = [best_trajectory(env,π_net) for _ in 1:10]
println(sum(τ[end][1][5] for τ in τs)/10)
create_plot(env,τs)