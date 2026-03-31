using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")
currentDir = @__DIR__
cd(currentDir)
# Constants
g = 9.81
ρ = 1025.

# Gradient scaling: σ is measurement noise std dev
# Use physics-meaningful scale: 1% strain error as reference
σ = 1e-2  # 1% strain error threshold (was 1.0 - caused tiny gradients!)
Ca0 = 1e-3
Ca1 = 1e3
δ = 1e-2 # For Huber loss

quadra(x) = x⋅x # quadratic loss
expo(x) = 1 - exp(- x⋅x) # exponetial loss
cauchy(x) = log(1 + x⋅x) # Cauchy loss
huber(x) = VALUE(∂0(x⋅x)) < δ ? 0.5 * x⋅x : δ * (sqrt(x⋅x) - 0.5*δ) # Huber loss
pseudo_huber(x) = δ^2 * (sqrt(1 + x⋅x/(δ^2)) - 1) # Pseudo huber loss
scaled_quadra(x) = sqrt(1 + x⋅x) # Custom
ch(x) = cosh(x) # Cosh
logch(x) = log(ch(x)) # Logcosh

loss_function = expo

max_disp = 1
static_bias = 0.015 

vec3(v,ind) = SVector{3}(v[i] for i∈ind)
@functor with() zeromotion(x,t) = x[1]

# Parameters
nsteps = 10  # REDUCED from 50 for faster testing (increase once convergence works)
Δtᵢₙᵥ = 0.01
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()

# Materials
# Define parameters for cross-section 1 (170mm R4 chain)
x1_D        = 0.306                              # Outer diameter [m]
x1_Dh       =   0.306                             # Hydrodynamic outer diameter [m]
x1_Area = (x1_D)^2*π/4                          # Inner area [m^2]
x1_ρₛ       =   7859.45                          # Steel denstiy [kg/m^3]
x1_EA       =   2.4681e09                          # Axial stiffness [N]
x1_μ        =   x1_Area*x1_ρₛ                   # Mass per unit length [m]
x1_w        =   x1_μ*g - π*x1_D^2/4*ρ*g          # Weight per unit length [N/m]
x1_Caₜ      =   1.0 *         ρ * π*x1_Dh^2/4    # Transverse added mass coefficients [N/m/(m/s^2)]
x1_Caₙ      =   1.4 *         ρ * π*x1_Dh^2/4   # Normal added mass coefficients [N/m/(m/s^2)]
x1_Cqₜ      =   0.5 *   0.5 * ρ * x1_Dh          # Transverse drag coefficients [N/m/(m/s)^2]
x1_Cqₙ      =   2.6 *   0.5 * ρ * x1_Dh         # Normal drag coefficients [N/m/(m/s)^2]
x1_Clₙ      =   0.0 *   0.5 * ρ * x1_Dh
x1_Clₜ      =   0.0 *   0.5 * ρ * x1_Dh
x1_mat         = AxisymmetricBarCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=x1_Caₜ, Clₜ=x1_Clₜ, Cqₜ=x1_Cqₜ, Caₙ=x1_Caₙ, Clₙ=x1_Clₙ, Cqₙ=x1_Cqₙ)

# Define parameters for cross-section 2 (250mm polyester)
x2_D        = 0.25                               # Outer diameter [m]
x2_Dh       =   0.25                             # Hydrodynamic outer diameter [m]
x2_Area     = (x2_D)^2*π/4                      # Inner area [m^2]
x2_ρₛ       =   1222.32                          # Steel denstiy [kg/m^3]
x2_EA       =   3.44e08                          # Axial stiffness [N]
x2_μ        =   x2_Area*x2_ρₛ                   # Mass per unit length [m]
x2_w        =   x2_μ*g - π*x2_D^2/4*ρ*g          # Weight per unit length [N/m]
x2_Caₜ      =   0.0 *         ρ * π*x2_Dh^2/4    # Transverse added mass coefficients [N/m/(m/s^2)]
x2_Caₙ      =   1.0 *         ρ * π*x2_Dh^2/4
x2_Cqₜ      =   0.0 *   0.5 * ρ * x2_Dh          # Transverse drag coefficients [N/m/(m/s)^2]
x2_Cqₙ      =   1.6 *   0.5 * ρ * x2_Dh
x2_Clₙ      =   0.0 *   0.5 * ρ * x2_Dh
x2_Clₜ      =   0.0 *   0.5 * ρ * x2_Dh
x2_mat         = AxisymmetricBarCrossSection(EA=x2_EA, μ=x2_μ, w=x2_w, Caₜ=x2_Caₜ, Clₜ=x2_Clₜ, Cqₜ=x2_Cqₜ, Caₙ=x2_Caₙ, Clₙ=x2_Clₙ, Cqₙ=x2_Cqₙ)

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

##########################################
## Forward
##########################################

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
x_disp = max_disp * inverseLoadSteps ./ inverseLoadSteps[end]
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

# Produce an animation
fig2   = Figure(size = (2000,1000))
ax2 = Axis3(fig2[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3))
xlims!(ax2,-1000,1000); ylims!(ax2,-1000,1000); zlims!(ax2,-waterDepth - 20,10)
graphic = draw!(ax2,stateForward[1])
ax2.azimuth[]=-π/2+π/180*10;
ax2.elevation[]=0+π/180*10;
framerate = 20
# loadStepsIterator = 1:3:nDynamicLoadSteps
loadStepsIterator = 1:3:length(stateForward)
record(fig2, "figs/animationForward.mp4", loadStepsIterator;
        framerate = framerate) do stateIdx
        draw!(graphic,stateForward[stateIdx])
end

##########################################
## Inverse
##########################################

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

# Penalties - adjusted for larger strains
# @functor with() costA(a) = 1e-3 * exp((a - 0.)^2)  # Increased penalty for γ₀
# @functor with() costAother(a) = 1e3 * exp((a - 0.)^2)  # Increased penalty for γ₁
@functor with() costA(a) = loss_function(sqrt(Ca0) * (a - 0.))  # Increased penalty for γ₀
@functor with() costAother(a) = loss_function(sqrt(Ca1) * (a - 0.))  # Increased penalty for γ₁
# @functor with() costA(a) = Ca0 * (a - 0.)^2  # Increased penalty for γ₀
# @functor with() costAother(a) = Ca1 * (a - 0.)^2  # Increased penalty for γ₁
eAγ₀ = addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₀, cost=costA)
eAγ₁ = addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₁, cost=costAother)


# Strain cost
measured_strain_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain .+ static_bias))
element1 = elementList_inv[1]

@functor with(measured_strain_interp, element1, model_inv, σ) function straincost(X,U,A,t)
    elestraingauge = model_inv.eleobj[element1]
    elebar = elestraingauge.eleobj
    
    # X is a Tuple with one element: a 6-element SVector
    # DOFs 1-3 are from node 1, DOFs 4-6 are from node 2
    Xvec = X[1]
    uᵧ₁ = vec3(Xvec, 1:3)
    uᵧ₂ = vec3(Xvec, 4:6)
    
    # Compute tangent vector from displacements
    tg = elebar.tgₘ + uᵧ₂ - uᵧ₁
    L = √(tg[1]^2+tg[2]^2+tg[3]^2)
    ε_val = L/elebar.Lₛ - 1
    
    # Compute strain with full AD w.r.t. both A and X
    # A[2] is multiplier, A[1] is bias - both are variated by solver
    εₚ = elestraingauge.ηₙ[1] * ((1 + A[2]) * ε_val + A[1]) # we use  "elestraingauge.ηₙ[1]" since we have only one strain gauge
    εₘ_val = measured_strain_interp(t)
    Δε = εₚ - εₘ_val
    # Scale error by σ (1% strain) for better conditioned gradients
    # This makes cost magnitude independent of absolute strain scale
    cost_val = loss_function(Δε / σ)
    
    # Debug prints (only at t == -5.0 for brevity)
    if VALUE(t) == -0.0
        println("ε_val: ", VALUE(ε_val))
        println("A: ", VALUE.(A))
        println("εₚ: ", VALUE(εₚ))
        println("εₘ_val: ", εₘ_val)
        println("Δε: ", VALUE(Δε))
        println("Δε/σ (scaled): ", VALUE(Δε)/σ)
        println("cost_val: ", VALUE(cost_val))
        println("[Gradient scaling enabled: σ=0.01 increases gradients ~100x]")
        
        # Extract and display gradients w.r.t. A and X
        Ptot = precedence(cost_val)
        Ntot = npartial(cost_val)
        if Ptot > 0 && Ntot > 0
            ∂cost_all = ∂{Ptot, Ntot}(cost_val)
            # Show some of the gradient components
            println("∂cost/∂Xvec[1]: ", VALUE(∂cost_all[1]))
            println("∂cost/∂A[1]: ", VALUE(∂cost_all[Ntot-1]))
            println("∂cost/∂A[2]: ", VALUE(∂cost_all[Ntot]))
        end
    end
    
    return cost_val
end


edofcost = addelement!(model_inv, DofCost, [nodeList_inv[1][1], nodeList_inv[1][2], anodeList_inv];
    xinod=(1,1,1,2,2,2), xfield=(:t1,:t2,:t3,:t1,:t2,:t3),
    ainod=(3,3), afield=(:γ₀,:γ₁),
    cost=straincost)

# Solve
initialstate_inv = initialize!(model_inv)
staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)


# TEST CHECK
#------------------------------------------
println("\n" * "="^70)
println("CONVERGENCE DIAGNOSTICS - Before running DirectXUA solver")
println("="^70)

# Test 1: Evaluate cost at current state
println("\n▶ TEST 1: Cost function evaluation at initial state")
state_check = staticStates_inv[end]
t_check = -5.0

# Get X for the first element dofs
X_current = (SVector{6}(state_check.X[1][j] for j in 1:6),)  
A_current = SVector(0.0, 0.0)  # Initial A

cost_initial = straincost(X_current, (), A_current, t_check)
println("  Cost at A=[0,0]: $(VALUE(cost_initial))")

# Test 2: Cost at slightly perturbed A
println("\n▶ TEST 2: Cost sensitivity to A parameters")
A_test2 = SVector(0.01, 0.0)
cost_test2 = straincost(X_current, (), A_test2, t_check)
δA1_effect = (VALUE(cost_test2) - VALUE(cost_initial)) / 0.01
println("  ∂cost/∂A[1] ≈ $δA1_effect (finite diff)")

A_test3 = SVector(0.0, 0.01)
cost_test3 = straincost(X_current, (), A_test3, t_check)
δA2_effect = (VALUE(cost_test3) - VALUE(cost_initial)) / 0.01
println("  ∂cost/∂A[2] ≈ $δA2_effect (finite diff)")

# Test 3: Check AD gradients
println("\n▶ TEST 3: AD gradient computation")
P = constants(X_current, A_current)
∂A_ad = variate{P, length(A_current)}(A_current)
cost_with_ad = straincost(X_current, (), ∂A_ad, t_check)
Ptot = precedence(cost_with_ad)
Ntot = npartial(cost_with_ad)
if Ptot > 0 && Ntot > 0
    ∂_vals = ∂{Ptot, Ntot}(cost_with_ad)
    println("  ∂cost/∂A[1] (AD) = $(VALUE(∂_vals[Ntot-1]))")
    println("  ∂cost/∂A[2] (AD) = $(VALUE(∂_vals[Ntot]))")
else
    println("  WARNING: No gradient computed!")
end

# Test 4: Check if cost is reasonable magnitude
println("\n▶ TEST 4: Cost value reasonableness")
println("  Absolute value: $(abs(VALUE(cost_initial)))")
if abs(VALUE(cost_initial)) < 1e-10
    println("  ⚠️  WARNING: Cost is extremely small - might cause numerical issues!")
elseif abs(VALUE(cost_initial)) > 100
    println("  ⚠️  WARNING: Cost is very large - check loss function scaling!")
else
    println("  ✓ Cost magnitude seems reasonable")
end

println("\nRECOMMENDATIONS:")
println("1. If ∂cost/∂A[1] and ∂cost/∂A[2] are ZERO → problem is under-determined")
println("2. If they have OPPOSITE signs → solver might oscillate")
println("3. If they are VERY LARGE → gradient scaling issue")
println("4. If cost magnitude << 1e-10 or >> 100 → numerical conditioning issue")
println("\n" * "="^70)
#------------------------------------------
# TEST CHECK


InverseSolver = DirectXUA{2,0,1}
stateXUA = solve(InverseSolver;
    initialstate=[staticStates_inv[end]],
    time=[inverseLoadSteps],
    verbose=true,
    maxiter=50,
    maxΔx=1e-3,   # More relaxed convergence (was 1e-3, keep for now)
    maxΔu=Inf,
    maxΔa=1e-4,   # REDUCED from 1e-3 to focus on A convergence
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