using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")

currentDir = @__DIR__
cd(currentDir)

# Constants
g = 9.81
ρ = 1025.
σ = 1e-2
Ca0 = 1e-3
Ca1 = 1e4
δ = 1e-1


quadra(x)         = x⋅x # quadratic loss
expo(x)           = 1 - exp(- x⋅x) # exponetial loss
cauchy(x)         = log(1 + x⋅x) # Cauchy loss
huber(x)          = VALUE(∂0(x⋅x)) < δ ? 0.5 * x⋅x : δ * (sqrt(x⋅x) - 0.5*δ) # Huber loss
pseudo_huber(x)   = δ^2 * (sqrt(1 + x⋅x/(δ^2)) - 1) # Pseudo huber loss
scaled_quadra(x)  = sqrt(1 + x⋅x) # Custom
ch(x)             = cosh(x) # Cosh
logch(x)          = log(ch(x)) # Logcosh


loss_name = "quadra"
loss_function = quadra

# Test parameters
max_disp = 1
verbose = true

println("=== Cost Surface Visualization (γ₀ vs X DOFs) ===")

vec3(v,ind) = SVector{3}(v[i] for i∈ind)
@functor with() zeromotion(x,t) = x[1]

nsteps = 10
Δtᵢₙᵥ = 0.01
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()

# Materials
x1_D        = 0.306
x1_Dh       = 0.306
x1_Area = (x1_D)^2*π/4
x1_ρₛ       = 7859.45
x1_EA       = 2.4681e09
x1_μ        = x1_Area*x1_ρₛ
x1_w        = x1_μ*g - π*x1_D^2/4*ρ*g
x1_Caₜ      = 1.0 * ρ * π*x1_Dh^2/4
x1_Caₙ      = 1.4 * ρ * π*x1_Dh^2/4
x1_Cqₜ      = 0.5 * 0.5 * ρ * x1_Dh
x1_Cqₙ      = 2.6 * 0.5 * ρ * x1_Dh
x1_Clₙ      = 0.0 * 0.5 * ρ * x1_Dh
x1_Clₜ      = 0.0 * 0.5 * ρ * x1_Dh
x1_mat = AxisymmetricBarCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=x1_Caₜ, Clₜ=x1_Clₜ, Cqₜ=x1_Cqₜ, Caₙ=x1_Caₙ, Clₙ=x1_Clₙ, Cqₙ=x1_Cqₙ)

x2_D        = 0.25
x2_Dh       = 0.25
x2_Area     = (x2_D)^2*π/4
x2_ρₛ       = 1222.32
x2_EA       = 3.44e08
x2_μ        = x2_Area*x2_ρₛ
x2_w        = x2_μ*g - π*x2_D^2/4*ρ*g
x2_Caₜ      = 0.0 * ρ * π*x2_Dh^2/4
x2_Caₙ      = 1.0 * ρ * π*x2_Dh^2/4
x2_Cqₜ      = 0.0 * 0.5 * ρ * x2_Dh
x2_Cqₙ      = 1.6 * 0.5 * ρ * x2_Dh
x2_Clₙ      = 0.0 * 0.5 * ρ * x2_Dh
x2_Clₜ      = 0.0 * 0.5 * ρ * x2_Dh
x2_mat = AxisymmetricBarCrossSection(EA=x2_EA, μ=x2_μ, w=x2_w, Caₜ=x2_Caₜ, Clₜ=x2_Clₜ, Cqₜ=x2_Cqₜ, Caₙ=x2_Caₙ, Clₙ=x2_Clₙ, Cqₙ=x2_Cqₙ)

nel = [5, 23, 12, 7]
segLength = [150., 414., 250., 150.]
xSection = [x1_mat, x2_mat, x2_mat, x1_mat]
nseg = length(nel)

waterDepth = 200.
fairleadDepth = 10.
offsetHorizontal = 1000. - sum(segLength) - 58.75
offsetDownwards = -waterDepth + fairleadDepth
prestrechStaticAnalysis = sum(segLength) * 0.01

# Forward solve (to get measured strain)
model = Model(:testline)
topNode = addnode!(model, [0., 0., -fairleadDepth])
nodeList, elementList, anodeList = MeshLineGauge(model, topNode, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

@functor with(offsetHorizontal, prestrechStaticAnalysis) xMotionBottom(x,t) = x[1] - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))
@functor with(offsetDownwards) zMotionBottom(x,t) = x[1] - ((min(t,-5.)+10)/5 * offsetDownwards)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

x_disp = max_disp * inverseLoadSteps ./ inverseLoadSteps[end]
x_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., x_disp))
@functor with() xMotionTop(x,t) = x[1] - x_disp_interp(t)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

@functor with() buoygravForce_topchain(t) = ((min(t,-5.)+10)/5) * (-3 + 0.) * 1e3 * g
@functor with() buoygravForce_midpolyester(t) = ((min(t,-5.)+10)/5) * (-3 + 10.) * 1e3 * g
@functor with() buoygravForce_bottomchain(t) = ((min(t,-5.)+10)/5) * (-3 + 15.) * 1e3 * g
addelement!(model, DofLoad, [nodeList[2][1]], field=:t3, value=buoygravForce_topchain)
addelement!(model, DofLoad, [nodeList[nseg-1][1]], field=:t3, value=buoygravForce_midpolyester)
addelement!(model, DofLoad, [nodeList[nseg][1]], field=:t3, value=buoygravForce_bottomchain)

Kv = 14000.
[addelement!(model, SoilContact, [nodeList[nseg][idxNod]], z₀=-waterDepth, Kh=0.0, Kv=Kv, Ch=0., Cv=0.0) for idxNod in 1:length(nodeList[nseg])]

initialstate = initialize!(model)
staticLoadSteps = (-10:0.1:0)
staticStates = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)
stateForward = solve(SweepX{2}; initialstate=staticStates[end], time=inverseLoadSteps, verbose=false, maxiter=60)
req = @request εₐₓ
out = getresult(stateForward, req, [elementList[1]])
strain = [out[idxEl].εₐₓ for idxEl in 1:size(out,2)]

# Now return the inverse model setup
static_bias = 0.015

model_inv = Model(:testline)
topNode_inv = addnode!(model_inv, [0., 0., -fairleadDepth])
nodeList_inv, elementList_inv, anodeList_inv = MeshLineGauge(model_inv, topNode_inv, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

addelement!(model_inv, DofLoad, [nodeList_inv[2][1]], field=:t3, value=buoygravForce_topchain)
addelement!(model_inv, DofLoad, [nodeList_inv[nseg-1][1]], field=:t3, value=buoygravForce_midpolyester)
addelement!(model_inv, DofLoad, [nodeList_inv[nseg][1]], field=:t3, value=buoygravForce_bottomchain)

[addelement!(model_inv, SoilContact, [nodeList_inv[nseg][idxNod]], z₀=-waterDepth, Kh=0.0, Kv=Kv, Ch=0., Cv=0.0) for idxNod in 1:length(nodeList_inv[nseg])]

@functor with() costA(a) = loss_function(sqrt(Ca0) * (a - 0.))
@functor with() costAother(a) = loss_function(sqrt(Ca1) * (a - 0.))
addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₀, cost=costA)
addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₁, cost=costAother)

measured_strain_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain .+ static_bias))
element1 = elementList_inv[1]

@functor with(measured_strain_interp, element1, model_inv, σ) function straincost(X,U,A,t)
    elestraingauge = model_inv.eleobj[element1]
    elebar = elestraingauge.eleobj
    
    Xvec = X[1]
    uᵧ₁ = vec3(Xvec, 1:3)
    uᵧ₂ = vec3(Xvec, 4:6)
    
    tg = elebar.tgₘ + uᵧ₂ - uᵧ₁
    L = √(tg[1]^2+tg[2]^2+tg[3]^2)
    ε_val = L/elebar.Lₛ - 1
    
    εₚ = elestraingauge.ηₙ[1] * ((1 + A[2]) * ε_val + A[1])
    εₘ_val = measured_strain_interp(t)
    Δε = εₚ - εₘ_val
    cost_val = loss_function(Δε / σ)
    
    return cost_val
end

addelement!(model_inv, DofCost, [nodeList_inv[1][1], nodeList_inv[1][2], anodeList_inv];
    xinod=(1,1,1,2,2,2), xfield=(:t1,:t2,:t3,:t1,:t2,:t3),
    ainod=(3,3), afield=(:γ₀,:γ₁),
    cost=straincost)

initialstate_inv = initialize!(model_inv)
staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

# Extract static positions for node 2
nodes = [nodeList_inv[1][1], nodeList_inv[1][2]]
fields = [:t1, :t2, :t3]
Xvec_static = Float64[]
for node in nodes
    for field in fields
        val = getdof(staticStates_inv[end]; class=:X, field=field, nodID=[node])[1]
        push!(Xvec_static, val)
    end
end

# Static positions for node 2 (last 3 DOFs)
static_t1_node2 = Xvec_static[4]
static_t2_node2 = Xvec_static[5]
static_t3_node2 = Xvec_static[6]
println("Static equilibrium positions:")
println("  t1_node2 = $static_t1_node2")
println("  t2_node2 = $static_t2_node2")
println("  t3_node2 = $static_t3_node2")

# DOF variations (centered around static positions)
dof_variations = [1.5, 2., 1.5]  # For t1, t2, t3
dof_names = ["t1_node2", "t2_node2", "t3_node2"]
static_dofs = [static_t1_node2, static_t2_node2, static_t3_node2]

# Range for γ₀ parameter
γ₀_range = range(-0.1, 0.1, length=21)

# DOF ranges (centered around static values)
dof_ranges = [
    range(static_t1_node2 - dof_variations[1], static_t1_node2 + dof_variations[1], length=21),
    range(static_t2_node2 - dof_variations[2], static_t2_node2 + dof_variations[2], length=21),
    range(static_t3_node2 - dof_variations[3], static_t3_node2 + dof_variations[3], length=21)
]

# Compute cost at different time steps
time_indices = [1, div(length(inverseLoadSteps), 4), div(length(inverseLoadSteps), 2), length(inverseLoadSteps)]

for t_idx in time_indices
    t_val = inverseLoadSteps[t_idx]
    println("\n--- Time step $t_idx, t = $t_val ---")
    
    # Get state at this time step
    temp_state = solve(SweepX{2}; initialstate=staticStates_inv[end], time=collect(inverseLoadSteps[1:t_idx]), verbose=false, maxiter=20)
    state_at_t = temp_state[end]
    
    # For each DOF, create a 2D cost surface with γ₀
    for dof_idx in 1:3
        println("\n  --- $(dof_names[dof_idx]) vs γ₀, t = $t_val ---")
        
        costs = zeros(length(γ₀_range), length(dof_ranges[dof_idx]))
        
        for (i, γ₀) in enumerate(γ₀_range)
            for (j, dof_val) in enumerate(dof_ranges[dof_idx])
                try
                    # Get X dofs from state at this time
                    Xvec = Float64[]
                    for node in nodes
                        for field in fields
                            val = getdof(state_at_t; class=:X, field=field, nodID=[node])[1]
                            push!(Xvec, val)
                        end
                    end
                    
                    # Modify the specific DOF
                    Xvec[3 + dof_idx] = dof_val
                    
                    X_dofs = (Xvec,)
                    γ₁_val = 0.0  # Keep γ₁ fixed
                    A_dofs = SVector(γ₀, γ₁_val)
                    
                    cost_val = VALUE(straincost(X_dofs, nothing, A_dofs, t_val)) .+ VALUE(costA(A_dofs[1])) .+ VALUE(costAother(A_dofs[2]))
                    costs[i, j] = cost_val
                    
                catch e
                    costs[i, j] = NaN
                    verbose && println("    Error at γ₀=$γ₀, $(dof_names[dof_idx])=$dof_val: $e")
                end
            end
        end
        
        # Find minimum
        min_cost, min_idx = findmin(costs)
        min_i, min_j = Tuple(min_idx)
        println("    Min cost = $min_cost at γ₀ = $(γ₀_range[min_i]), $(dof_names[dof_idx]) = $(dof_ranges[dof_idx][min_j])")
        
        # Plot contours
        fig = Figure(size=(800, 700))
        ax = Axis(fig[1,1], xlabel="γ₀ (static bias)", ylabel=dof_names[dof_idx], 
                    title="Cost surface: γ₀ vs $(dof_names[dof_idx]), max_disp = $max_disp, t = $t_val")
        contourf!(ax, γ₀_range, dof_ranges[dof_idx], costs, levels=20)
        Colorbar(fig[1,2], label="Cost")
        # Mark the minimum
        scatter!(ax, [γ₀_range[min_i]], [dof_ranges[dof_idx][min_j]], color=:red, markersize=10, label="Min")
        axislegend(ax)
        save("figs/cost_surface_gamma0_$(dof_names[dof_idx])_max_disp_$(max_disp)_t_$(t_idx)_"*loss_name*".png", fig)
        println("    Saved: figs/cost_surface_gamma0_$(dof_names[dof_idx])_max_disp_$(max_disp)_t_$(t_idx).png")
    end
end

println("\nVisualization complete!")
