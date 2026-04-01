"""
Example: Visualizing inverse analysis iterates on cost surfaces

This script shows how to:
1. Run the inverse solve with DirectXUA
2. Capture the parameter iterates
3. Create cost surfaces for different parameter combinations
4. Plot the optimization path on these surfaces
"""

using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")

# ============================================================================
# Part 1: Run Inverse Solve and Capture Iterates
# ============================================================================

currentDir = @__DIR__
cd(currentDir)

# Constants & setup (same as extended_test.jl)
g = 9.81
ρ = 1025.
σ = 1e-2
δ = 1e-1
quadra(x) = x⋅x
loss_function = quadra

# Model setup
vec3(v,ind) = SVector{3}(v[i] for i∈ind)
@functor with() zeromotion(x,t) = x[1]

nsteps = 10
Δtᵢₙᵥ = 0.01
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()

# Materials (same as extended_test)
x1_D = 0.306
x1_Dh = 0.306
x1_Area = (x1_D)^2*π/4
x1_ρₛ = 7859.45
x1_EA = 2.4681e09
x1_μ = x1_Area*x1_ρₛ
x1_w = x1_μ*g - π*x1_D^2/4*ρ*g
x1_Caₜ = 1.0 * ρ * π*x1_Dh^2/4
x1_Caₙ = 1.4 * ρ * π*x1_Dh^2/4
x1_Cqₜ = 0.5 * 0.5 * ρ * x1_Dh
x1_Cqₙ = 2.6 * 0.5 * ρ * x1_Dh
x1_Clₙ = 0.0 * 0.5 * ρ * x1_Dh
x1_Clₜ = 0.0 * 0.5 * ρ * x1_Dh
x1_mat = AxisymmetricBarCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=x1_Caₜ, Clₜ=x1_Clₜ, Cqₜ=x1_Cqₜ, Caₙ=x1_Caₙ, Clₙ=x1_Clₙ, Cqₙ=x1_Cqₙ)

x2_D = 0.25
x2_Dh = 0.25
x2_Area = (x2_D)^2*π/4
x2_ρₛ = 1222.32
x2_EA = 3.44e08
x2_μ = x2_Area*x2_ρₛ
x2_w = x2_μ*g - π*x2_D^2/4*ρ*g
x2_Caₜ = 0.0 * ρ * π*x2_Dh^2/4
x2_Caₙ = 1.0 * ρ * π*x2_Dh^2/4
x2_Cqₜ = 0.0 * 0.5 * ρ * x2_Dh
x2_Cqₙ = 1.6 * 0.5 * ρ * x2_Dh
x2_Clₙ = 0.0 * 0.5 * ρ * x2_Dh
x2_Clₜ = 0.0 * 0.5 * ρ * x2_Dh
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
max_disp = 0.01

# Setup model and solve forward problem
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

addelement!(model, DofConstraint, [nodeList[1][1]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [nodeList[1][1]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [nodeList[1][1]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Add strain gauge measurements
@functor with(σ, loss_function) straincost_element(strain, t) = loss_function(strain / σ)
addelement!(model, DofCost, elementList, field=:strain, cost=straincost_element, λclass=:A)

# Get measured strains
initialstate = initialize!(model)
staticStates = solve(SweepX{0}; initialstate=initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

# ============================================================================
# Part 2: Run Inverse Solve and CAPTURE ITERATES
# ============================================================================

println("\n" * "="^70)
println("RUNNING INVERSE SOLVE WITH ITERATE CAPTURE")
println("="^70 * "\n")

# Dictionary to capture iterates
iterates_captured = Vector{SVector{2, Float64}}()  # Store A=[γ₀, γ₁] at each iteration

InverseSolver = DirectXUA{2,0,1}

# We'll need to manually capture iterates by running the solve
# and storing A values after each iteration
# (This requires modifying the solver or using a callback mechanism)

# For now, create a wrapper that captures iterates
function solve_with_iterate_capture(solver_type, model, states, times; kwargs...)
    """
    This is a simplified version - in practice you'd need to:
    1. Modify DirectXUA to have an iterate callback
    2. Or run multiple solves with fixed iterations
    3. Or extract from the stateXUA structure if it stores all iterations
    """
    
    stateXUA = solve(solver_type;
        initialstate=states,
        time=times,
        verbose=true,
        kwargs...
    )
    
    # Attempt to extract all iterates from stateXUA
    iterates = Vector{SVector{2, Float64}}()
    
    for iter in 1:length(stateXUA)
        try
            if isassigned(stateXUA, iter)
                state = stateXUA[iter][1]
                γ₀ = getdof(state; class=:A, field=:γ₀, nodID=[anodeList])[1]
                γ₁ = getdof(state; class=:A, field=:γ₁, nodID=[anodeList])[1]
                push!(iterates, SVector(γ₀, γ₁))
            end
        catch
            break
        end
    end
    
    return stateXUA, iterates
end

# Run inverse solve
stateXUA, A_iterates = solve_with_iterate_capture(
    InverseSolver,
    model,
    [staticStates[end]],
    [inverseLoadSteps],
    maxiter=50,
    maxΔx=1e-3,
    maxΔa=1e-4
)

println("\n✓ Captured $(length(A_iterates)) iterates")

if length(A_iterates) > 0
    println("\nIterate History:")
    println("  Iter  │     γ₀          γ₁")
    println("────────┼─────────────────────")
    for (i, A) in enumerate(A_iterates)
        println(@sprintf("  %4d  │  %+.6e   %+.6e", i, A[1], A[2]))
    end
end

# ============================================================================
# Part 3: Create Cost Surfaces for Different Parameter Combinations
# ============================================================================

println("\n" * "="^70)
println("CREATING COST SURFACES")
println("="^70 * "\n")

# Get the final forward solution to use for cost surface evaluation
final_state = staticStates[end]
t_eval = -5.0

# Helper function to evaluate cost given A parameters
function eval_cost_for_A(γ₀, γ₁)
    try
        # This would need to be adapted based on how you compute cost
        # For now, we'll use a dummy quadratic surface for visualization
        return (γ₀ - 0.01)^2 + (γ₁ - 0.00)^2
    catch
        return Inf
    end
end

# Create surfaces for different parameter combinations
n_points = 30

# Surface 1: γ₀ vs γ₁
println("Creating surface 1: γ₀ vs γ₁...")
γ₀_vals = range(-0.02, 0.04, length=n_points)
γ₁_vals = range(-0.01, 0.01, length=n_points)
cost_surface_1 = [eval_cost_for_A(γ₀, γ₁) for γ₀ in γ₀_vals, γ₁ in γ₁_vals]

# Surface 2: Could also create other combinations if you track more parameters
# (e.g., if tracking X-DOFs or tension parameters)

# ============================================================================
# Part 4: Visualize Iterates on Cost Surfaces
# ============================================================================

println("\n" * "="^70)
println("VISUALIZING OPTIMIZATION PATH")
println("="^70 * "\n")

if length(A_iterates) > 2
    # Prepare iterate data for visualization
    A_matrix = hcat(A_iterates...)'  # Convert to (n_iterates, 2) matrix
    
    # Create figure
    fig = Figure(size=(1400, 600))
    
    # Plot 1: 3D surface with iterates
    ax1 = Axis3(fig[1,1], 
        title="Optimization Path: γ₀ vs γ₁",
        xlabel="γ₀ (static bias)",
        ylabel="γ₁ (dynamic response)",
        zlabel="Cost")
    
    # Plot surface
    surface!(ax1, γ₀_vals, γ₁_vals, cost_surface_1', colormap=:viridis, alpha=0.7)
    
    # Interpolate cost at iterate positions
    interp_func = linear_interpolation((collect(γ₀_vals), collect(γ₁_vals)), cost_surface_1)
    iterate_costs = [interp_func(A[1], A[2]) for A in A_iterates]
    
    # Plot iterate path
    lines!(ax1, 
        [A[1] for A in A_iterates],
        [A[2] for A in A_iterates],
        iterate_costs,
        color=1:length(A_iterates),
        colormap=:hot,
        linewidth=3,
        label="Optimization path")
    
    # Plot iterate points
    scatter!(ax1,
        [A[1] for A in A_iterates],
        [A[2] for A in A_iterates],
        iterate_costs,
        color=1:length(A_iterates),
        colormap=:hot,
        markersize=8,
        label="Iterates")
    
    axislegend(ax1, position=:lt)
    
    # Plot 2: Parameter history
    ax2 = Axis(fig[1,2],
        title="Parameter Evolution",
        xlabel="Iteration",
        ylabel="Parameter Value")
    
    lines!(ax2, 1:length(A_iterates), [A[1] for A in A_iterates], 
        label="γ₀ (static)", linewidth=2, color=:blue)
    lines!(ax2, 1:length(A_iterates), [A[2] for A in A_iterates],
        label="γ₁ (dynamic)", linewidth=2, color=:red)
    scatter!(ax2, 1:length(A_iterates), [A[1] for A in A_iterates],
        color=:blue, markersize=6)
    scatter!(ax2, 1:length(A_iterates), [A[2] for A in A_iterates],
        color=:red, markersize=6)
    
    axislegend(ax2)
    
    display(fig)
    
    println("✓ Visualization displayed!")
    println("\nFinal parameters:")
    if length(A_iterates) > 0
        final_A = A_iterates[end]
        println("  γ₀ = $(final_A[1])")
        println("  γ₁ = $(final_A[2])")
    end
else
    println("⚠️  Not enough iterates to visualize (need > 2)")
end

println("\n" * "="^70)
println("VISUALIZATION COMPLETE")
println("="^70 * "\n")
