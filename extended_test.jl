using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")

# Constants
g = 9.81
ρ = 1025.
σ = 1.e0

static_bias = 0.1  # Smaller bias

vec3(v,ind) = SVector{3}(v[i] for i∈ind)
@functor with() zeromotion(x,t) = x[1]

# Parameters
nsteps = 5
Δtᵢₙᵥ = 0.1
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()

# Materials
x1_EA = 2.4681e09
x1_μ = 1000.
x1_w = 1000.
x1_mat = AxisymmetricBarCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=1., Clₜ=0., Cqₜ=0.5, Caₙ=1.4, Clₙ=0., Cqₙ=2.6)

x2_EA = 3.44e08
x2_ρₛ = 1222.32
x2_Area = (0.25)^2 * π/4
x2_μ = x2_Area * x2_ρₛ
x2_w = x2_μ * g - π * 0.25^2/4 * ρ * g
x2_mat = AxisymmetricBarCrossSection(EA=x2_EA, μ=x2_μ, w=x2_w, Caₜ=0., Clₜ=0., Cqₜ=0., Caₙ=1.0, Clₙ=0., Cqₙ=1.6)

# Segments
nel = [5, 23, 12, 7]
segLength = [150., 414., 250., 150.]
xSection = [x1_mat, x2_mat, x2_mat, x1_mat]
nseg = length(nel)

# Geometry
waterDepth = 200.
fairleadDepth = 10.
offsetHorizontal = 1000. - sum(segLength) - 58.75
offsetDownwards = -waterDepth + fairleadDepth
prestrechStaticAnalysis = sum(segLength) * 0.01

# Nodes
model = Model(:testline)
topNode = addnode!(model, [0., 0., -fairleadDepth])
nodeList, elementList, anodeList = MeshLineGauge(model, topNode, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

# Constraints
@functor with(offsetHorizontal, prestrechStaticAnalysis) xMotionBottom(x,t) = x[1] - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))
@functor with(offsetDownwards) zMotionBottom(x,t) = x[1] - ((min(t,-5.)+10)/5 * offsetDownwards)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

# Top motion
x_disp = 0.01 * inverseLoadSteps ./ inverseLoadSteps[end]
x_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., x_disp))
@functor with() xMotionTop(x,t) = x[1] - x_disp_interp(t)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Buoys
nodnum_buoygrav_topchain = 2
nodnum_buoygrav_midpolyester = nseg - 1
nodnum_buoygrav_bottomchain = nseg
@functor with() buoygravForce_topchain(t) = ((min(t,-5.)+10)/5) * (-3 + 0.) * 1e3 * g
@functor with() buoygravForce_midpolyester(t) = ((min(t,-5.)+10)/5) * (-3 + 10.) * 1e3 * g
@functor with() buoygravForce_bottomchain(t) = ((min(t,-5.)+10)/5) * (-3 + 15.) * 1e3 * g
addelement!(model, DofLoad, [nodeList[nodnum_buoygrav_topchain][1]], field=:t3, value=buoygravForce_topchain)
addelement!(model, DofLoad, [nodeList[nodnum_buoygrav_midpolyester][1]], field=:t3, value=buoygravForce_midpolyester)
addelement!(model, DofLoad, [nodeList[nodnum_buoygrav_bottomchain][1]], field=:t3, value=buoygravForce_bottomchain)

# Soil contact
Kv = 14000.
segnumsoil = nseg
[addelement!(model, SoilContact, [nodeList[segnumsoil][idxNod]], z₀=-waterDepth, Kh=0.0, Kv=Kv, Ch=0., Cv=0.0) for idxNod in 1:length(nodeList[segnumsoil])]

# Forward solve
initialstate = initialize!(model)
staticLoadSteps = (-10:0.1:0)
staticStates = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)
stateForward = solve(SweepX{2}; initialstate=staticStates[end], time=inverseLoadSteps, verbose=false, maxiter=60)
req = @request εₐₓ
out = getresult(stateForward, req, [elementList[1]])
strain = [out[idxEl].εₐₓ for idxEl in 1:size(out,2)]

# Inverse
model_inv = Model(:testline)
topNode_inv = addnode!(model_inv, [0., 0., -fairleadDepth])
nodeList_inv, elementList_inv, anodeList_inv = MeshLineGauge(model_inv, topNode_inv, 0., Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

# Constraints (same as forward)
@functor with(offsetHorizontal, prestrechStaticAnalysis) xMotionBottom(x,t) = x[1] - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))
@functor with(offsetDownwards) zMotionBottom(x,t) = x[1] - ((min(t,-5.)+10)/5 * offsetDownwards)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [nodeList_inv[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal)

@functor with() xMotionTop(x,t) = x[1] - x_disp_interp(t)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Buoys
addelement!(model_inv, DofLoad, [nodeList_inv[nodnum_buoygrav_topchain][1]], field=:t3, value=buoygravForce_topchain)
addelement!(model_inv, DofLoad, [nodeList_inv[nodnum_buoygrav_midpolyester][1]], field=:t3, value=buoygravForce_midpolyester)
addelement!(model_inv, DofLoad, [nodeList_inv[nodnum_buoygrav_bottomchain][1]], field=:t3, value=buoygravForce_bottomchain)

# Soil
[addelement!(model_inv, SoilContact, [nodeList_inv[segnumsoil][idxNod]], z₀=-waterDepth, Kh=0.0, Kv=Kv, Ch=0., Cv=0.0) for idxNod in 1:length(nodeList_inv[segnumsoil])]

# Penalties
@functor with() costA(a) = 1e-2 * (a - 0.)^2  # Smaller penalty for γ₀
@functor with() costAother(a) = 1e2 * (a - 0.)^2  # Larger penalty for γ₁
eAγ₀ = addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₀, cost=costA)
eAγ₁ = addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₁, cost=costAother)

# Strain cost
measured_strain_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain .+ static_bias))
element1 = elementList_inv[1]

@functor with(measured_strain_interp, element1, model_inv, σ) function straincost(X,U,A,t)
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

edofcost = addelement!(model_inv, DofCost, [nodeList_inv[1][1], nodeList_inv[1][2], anodeList_inv];
    xinod=(1,1,1,2,2,2), xfield=(:t1,:t2,:t3,:t1,:t2,:t3),
    ainod=(3,3), afield=(:γ₀,:γ₁),
    cost=straincost)

# Solve
initialstate_inv = initialize!(model_inv)
staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

InverseSolver = DirectXUA{2,0,1}
stateXUA = solve(InverseSolver;
    initialstate=[staticStates_inv[end]],
    time=[inverseLoadSteps],
    verbose=true,
    maxiter=50,
    maxΔx=1e-3,
    maxΔu=1e-3,
    maxΔa=1e-4,
    maxΔλ=Inf,
    saveiter=false
)

laststep = findlastassigned(stateXUA)
if laststep > 0
    state = stateXUA[laststep][1]
    staticdev = getdof(state; class=:A, field=:γ₀, nodID=[anodeList_inv])[1]
    dynamicdev = getdof(state; class=:A, field=:γ₁, nodID=[anodeList_inv])[1]
    println("Converged: Static dev = $staticdev, True = $static_bias, Error = $(abs(staticdev - static_bias)/static_bias * 100)%")
    println("Dynamic dev = $dynamicdev")
else
    println("Did not converge")
end