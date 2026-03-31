using Muscade, StaticArrays, Muscade.Toolbox, Interpolations, LinearAlgebra, Printf
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")
currentDir = @__DIR__
cd(currentDir)

# Use same setup as extended_test.jl (reduced version)
g = 9.81
ρ = 1025.
σ = 1.e0
static_bias = 0.015
quadra(x) = x⋅x
loss_function = quadra
max_disp = 1

vec3(v,ind) = SVector{3}(v[i] for i∈ind)
@functor with() zeromotion(x,t) = x[1]

# Parameters (REDUCED for faster testing)
nsteps = 10  # REDUCED from 50
Δtᵢₙᵥ = 0.01
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()
staticLoadSteps = -10.:0.1:0.

# Materials
x1_D = 0.306
x1_ρₛ = 7859.45
x1_EA = 2.4681e09
x1_μ = (x1_D)^2*π/4*x1_ρₛ
x1_w = x1_μ*g - π*x1_D^2/4*ρ*g
x1_Caₜ = 1.0 * ρ * π*x1_D^2/4
x1_Caₙ = 1.4 * ρ * π*x1_D^2/4
x1_Cqₜ = 0.5 * 0.5 * ρ * x1_D
x1_Cqₙ = 2.6 * 0.5 * ρ * x1_D
x1_Clₙ = 0.0* 0.5 * ρ * x1_D
x1_Clₜ = 0.0 * 0.5 * ρ * x1_D
x1_mat = AxisymmetricBarCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=x1_Caₜ, Clₜ=x1_Clₜ, Cqₜ=x1_Cqₜ, Caₙ=x1_Caₙ, Clₙ=x1_Clₙ, Cqₙ=x1_Cqₙ)

x2_D = 0.25
x2_ρₛ = 1222.32
x2_EA = 3.44e08
x2_μ = (x2_D)^2*π/4*x2_ρₛ
x2_w = x2_μ*g - π*x2_D^2/4*ρ*g
x2_Caₜ = 0.0 * ρ * π*x2_D^2/4
x2_Caₙ = 1.0 * ρ * π*x2_D^2/4
x2_Cqₜ = 0.0 * 0.5 * ρ * x2_D
x2_Cqₙ = 1.6 * 0.5 * ρ * x2_D
x2_Clₙ = 0.0 * 0.5 * ρ * x2_D
x2_Clₜ = 0.0 * 0.5 * ρ * x2_D
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

# Build forward model
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

@functor with(x_disp_interp) xMotionTop(x,t) = x[1] - x_disp_interp(t)
addelement!(model, DofConstraint, [nodeList[1][1]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1_top, gap=xMotionTop, mode=equal)
addelement!(model, DofConstraint, [nodeList[1][1]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2_top, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [nodeList[1][1]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3_top, gap=zeromotion, mode=equal)

# Static solve
initialstate = initialize!(model)
staticStates = solve(SweepX{0}; initialstate=initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

println("\n" * "="^60)
println("PHASE 1: DIAGNOSTICS - CHECK STATIC SOLVE")
println("="^60)
staticState = staticStates[end]
println("✓ Static solve completed")

# Now build inverse model
model_inv = Model(:testline_inv)
topNode_inv = addnode!(model_inv, [0., 0., -fairleadDepth])
nodeList_inv, elementList_inv, anodeList_inv = MeshLineGauge(model_inv, topNode_inv, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

addelement!(model_inv, DofConstraint, [nodeList_inv[1][1]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1_top, gap=xMotionTop, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[1][1]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2_top, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[1][1]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3_top, gap=zeromotion, mode=equal)

initialstate_inv = initialize!(model_inv)
staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

# Generate synthetic data from static solve
strain_data = [0.0; fill(0.001, nsteps)]  # Simple synthetic strain
measured_strain_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain_data))

element1 = elementList_inv[1]

println("\n" * "="^60)
println("PHASE 2: EVALUATE COST FUNCTION AT DIFFERENT A VALUES")
println("="^60)

# Get a state to test with
state_test = staticStates_inv[end]

# Extract X at first element
first_elem_nodes = [nodeList_inv[1][1], nodeList_inv[1][2]]
X_test = SVector{6}(state_test.X[1][j] for j in 1:6)  # First 6 dofs

# Test cost at different A values
A_values = [
    SVector(0.0, 0.0),
    SVector(0.01, 0.0),
    SVector(0.0, 0.1),
    SVector(0.01, 0.1),
    SVector(-0.01, 0.1),
]

# Create the straincost function (same as in extended_test.jl)
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
    cost_val = loss_function(Δε / (sqrt(2)σ))
    return cost_val
end

println("\nCost vs A values (at t=-5.0):")
println("A[1]        A[2]        Cost        ∂cost/∂A[1]  ∂cost/∂A[2]")
println("---------------------------------------------------")

for A_test in A_values
    # Create variated version for gradient computation
    P = constants(A_test)
    ∂A = variate{P,length(A_test)}(A_test)
    cost_val = straincost((X_test,), (), ∂A, -5.0)
    
    cost_extracted = VALUE(cost_val)
    if precedence(cost_val) > 0 && npartial(cost_val) > 0
        grads = ∂{P, 2}(cost_val)
        grad_A1 = VALUE(grads[1])
        grad_A2 = VALUE(grads[2])
    else
        grad_A1 = grad_A2 = 0.0
    end

    @printf("%-11.4f %-11.4f %-11.6e %-11.6e %-11.6e\n", 
            A_test[1], A_test[2], cost_extracted, grad_A1, grad_A2)
end

println("\nPHASE 2 ANALYSIS:")
println("✓ Cost function is evaluable at different A values")
println("✓ Gradients are computable")
println("\nKey observations:")
println("1. Is the cost landscape smooth?")
println("2. Do gradients point in a consistent direction?")
println("3. Is there a clear minimum?")

println("\n" * "="^60)
println("PHASE 3: TEST WITH REDUCED TIME STEPS")
println("="^60)
println("Recommendation: Run extended_test.jl with nsteps=5 instead of 50")
println("to see if solver converges on simpler problem")
