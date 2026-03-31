using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")

# Constants
g = 9.81
ρ = 1025.
σ = 0.1

# Test with static bias
static_bias = 1  # Non-zero bias to test

vec3(v,ind) = SVector{3}(v[i] for i∈ind)
@functor with() zeromotion(x,t) = x[1]

# Reduced parameters for testing
nsteps = 5
Δtᵢₙᵥ = 0.1
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()

# Material properties (simplified)
x1_EA = 2.4681e09
x1_μ = 1000.
x1_w = 1000.
x1_mat = AxisymmetricBarCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=1., Clₜ=0., Cqₜ=0.5, Caₙ=1.4, Clₙ=0., Cqₙ=2.6)

# Segments
nel = [3, 3, 3]
segLength = [10., 20., 10.]
xSection = [x1_mat, x1_mat, x1_mat]
nseg = length(nel)

# Simple geometry
waterDepth = 50.
fairleadDepth = 5.
offsetHorizontal = 50.
offsetDownwards = -waterDepth + fairleadDepth
prestrechStaticAnalysis = 0.01

# Nodes
model = Model(:testline)
topNode = addnode!(model, [0., 0., -fairleadDepth])
nodeList, elementList, aNode = MeshLineGauge(model, topNode, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

# Constraints
@functor with() xMotionBottom(x,t) = x[1] - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))
@functor with() yMotionBottom(x,t) = x[1]
@functor with() zMotionBottom(x,t) = x[1] - (min(t,-5.)+10)/5 * offsetDownwards
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=yMotionBottom, mode=equal)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

# Top motion
x_disp = 0.01 * inverseLoadSteps ./ inverseLoadSteps[end]
x_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., x_disp))
@functor with() xMotionTop(x,t) = x[1] - x_disp_interp(t)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Forward solve
initialstate = initialize!(model)
staticLoadSteps = (-10:0.1:0)
staticStates = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)
stateForward = solve(SweepX{2}; initialstate=staticStates[end], time=inverseLoadSteps, verbose=false, maxiter=60)
req = @request εₐₓ
out = getresult(stateForward, req, [elementList[1]])
strain = [out[idxEl].εₐₓ for idxEl in 1:size(out,2)]
println("True strain: ", strain)
println("Measured strain (with bias): ", strain .+ static_bias)

# Inverse
model_inv = Model(:testline)
topNode_inv = addnode!(model_inv, [0., 0., -fairleadDepth])
nodeList_inv, elementList_inv, aNode_inv = MeshLineGauge(model_inv, topNode_inv, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

# Constraints
@functor with() xMotionBottom(x,t) = x[1] - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))
@functor with() yMotionBottom(x,t) = x[1]
@functor with() zMotionBottom(x,t) = x[1] - (min(t,-5.)+10)/5 * offsetDownwards
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=yMotionBottom, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

# Top motion
x_disp = 0.01 * inverseLoadSteps ./ inverseLoadSteps[end]
x_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., x_disp))
@functor with() xMotionTop(x,t) = x[1] - x_disp_interp(t)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Add costs
@functor with() costA(a) = 1e-3 * (a - 0.)^2
@functor with() costAother(a) = 1e3 * (a - 0.)^2
eAγ₀ = addelement!(model_inv, SingleAcost, [aNode_inv]; field=:γ₀, cost=costA)
eAγ₁ = addelement!(model_inv, SingleAcost, [aNode_inv]; field=:γ₁, cost=costAother)

measured_strain_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain .+ static_bias))

element1 =  elementList_inv[1]
@functor with(measured_strain_interp,element1, model_inv, σ) function straincost(X,U,A,t)
    elestraingauge = model_inv.eleobj[element1]
    elebar = elestraingauge.eleobj
    P,ND = constants(X),length(X)
    x_ = motion{P}(X)
    uᵧ₁,uᵧ₂ = vec3(x_,1:3), vec3(x_,4:6)
    tg = elebar.tgₘ + uᵧ₂ - uᵧ₁
    L = √(tg[1]^2+tg[2]^2+tg[3]^2)
    ε_ = L/elebar.Lₛ - 1
    ε = motion⁻¹{P,ND}(ε_)
    ε = ∂0(ε)
    εₚ = elestraingauge.ηₙ .* ((1 .+ A[2]) .* ε .+ A[1])
    εₘ = measured_strain_interp(t)
    Δε = εₚ .- εₘ
    cost_val = (Δε⋅Δε) / (2σ^2)
    return cost_val
end

edofcost = addelement!(model_inv, DofCost, [nodeList_inv[1][1], nodeList_inv[1][2], aNode_inv];
    xinod=(1,1,1,2,2,2), xfield=(:t1,:t2,:t3,:t1,:t2,:t3),
    ainod=(3,3), afield=(:γ₀,:γ₁),
    cost=straincost)

# Constraints (same as above)

initialstate_inv = initialize!(model_inv)
staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

InverseSolver = DirectXUA{2,0,1}
stateXUA = solve(InverseSolver;
    initialstate=[staticStates_inv[end]],
    time=[inverseLoadSteps],
    verbose=true,
    maxiter=20,
    maxΔx=1e-3,
    maxΔu=1e-3,
    maxΔa=1e-4,
    maxΔλ=Inf,
    saveiter=false
)

laststep = findlastassigned(stateXUA)
if laststep > 0
    state = stateXUA[laststep][1]
    staticdev = getdof(state; class=:A, field=:γ₀, nodID=[aNode_inv])[1]
    dynamicdev = getdof(state; class=:A, field=:γ₁, nodID=[aNode_inv])[1]
    println("Converged: Static dev = $staticdev, Dynamic dev = $dynamicdev")
else
    println("Did not converge")
end