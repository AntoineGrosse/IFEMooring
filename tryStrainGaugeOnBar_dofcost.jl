using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")

##########################################
## Pre-Script
##########################################

# Some physical constants
#------------------------------------------ 
const g=9.81
const ρ=1025.

# Parameters inverse analysis
#------------------------------------------
nsteps = 10
Δtᵢₙᵥ             = 0.1               # Time step for the inverse analysis [s]
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps-1)*Δtᵢₙᵥ)*1.
nLoadSteps_inv = length(inverseLoadSteps)
InverseSolver        = DirectXUA{2,0,1}   # Dynamic solver for the inverse analysis
maxiterinv = 12
maxΔx = 1e-5
maxΔu = 1e-5
maxΔa = 1e-6
maxΔλ = Inf

##########################################
## Initialisation
##########################################
# Define the prescribed displacements on the x direction on node 2
x_disp = (50/100) .* inverseLoadSteps ./ inverseLoadSteps[end] # displacement causes a strain that is 50% of the bar length at the end timestep
x_disp_interp = linear_interpolation(inverseLoadSteps, x_disp)

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


# Nodes
#------------------------------------------   
model = Model(:test)
nod1 = addnode!(model, [0.,0.,0.])
nod2 = addnode!(model, [1.,0.,0.])
Anod = addnode!(model, [0.,0.])

# Costs A
#------------------------------------------
@functor with() costA(a)             = 1.e-3 *    (a - 0.)^2
@functor with() costAother(a)        = 1.e6 *    (a - 0.)^2
eAγ₀             = [addelement!(model,SingleAcost,[Anod]; field=:γ₀,cost=costA)];
eAγ₁             = [addelement!(model,SingleAcost,[Anod]; field=:γ₁,cost=costAother)];

# Element
#------------------------------------------
# Constructing the element
element1 = addelement!(model,StrainGaugeOnBar3D,[nod1,nod2,Anod];
    P=SMatrix{3,1}(0.,.5,0.),
    D=SMatrix{3,1}(1.,0.,0.),
    elementkwargs = (mat=x1_mat,)
);

# Costs ε
#------------------------------------------
strain = [
    2.220446049250313e-16
    0.0555555555555558
    0.11111111111111138
    0.16666666666666696
    0.22222222222222254
    0.2777777777777779
    0.3333333333333335
    0.38888888888888906
    0.44444444444444464
    0.5000000000000004
]
    
static_bias = 0.3
measured_strain_interp = linear_interpolation(inverseLoadSteps, strain .+ static_bias)
vec3(v,ind) = SVector{3}(v[i] for i∈ind);
@functor with(measured_strain_interp, element1, model) function straincost(X,U,A, t)
    σ = 1.

    ############## Computation of the strain as it is done in the element bar ##############
    elestraingauge = model.eleobj[element1]                                                 
    elebar = elestraingauge.eleobj
    # Obtain motions (i.e. including velocity and accelerations) from X
    P,ND    = constants(X),length(X)
    x_      = motion{P}(X)
    # Motions of the nodes, center of the element
    uᵧ₁,uᵧ₂   = vec3(x_,1:3), vec3(x_,4:6)
    # Element direction and length
    tg      = elebar.tgₘ + uᵧ₂ - uᵧ₁
    L       = √(tg[1]^2+tg[2]^2+tg[3]^2)
    # Strains
    ε_       = L/elebar.Lₛ - 1
    # Compute how strains vary with nodal displacements
    ε= motion⁻¹{P,ND}(ε_)
    ############## Computation of the strain as it is done in the element bar ##############
    
    # Predicted strain
    εₚ = elestraingauge.ηₙ .* ((1 .+ A[2]) .* ε .+ A[1])
    
    # Measured strain
    εₘ = measured_strain_interp(t)
    
    # Error
    Δε = εₚ .- εₘ
    
    # Compute the cost value
    cost_val = (Δε⋅Δε) / (2σ^2)

    return cost_val
end

# Adding the strain cost as a dof cost
edofcost             = addelement!(model,DofCost,[nod1,nod2,Anod]; 
    xinod=(1,1,1,2,2,2),xfield = (:t1,:t2,:t3,:t1,:t2,:t3),
    ainod=(3,3),afield = (:γ₀,:γ₁),
    cost = straincost)

# constraints
#------------------------------------------
[addelement!(model,Hold,[nod1];field) for field ∈ [:t1, :t2, :t3]]  # Hold on node 1
[addelement!(model,Hold,[nod2];field) for field ∈ [:t2, :t3]]       # Only x displacements possible on node 2
@functor with(x_disp) x_prescription(x,t)= x[1] - x_disp_interp(t)
addelement!(model,DofConstraint,[nod2],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=x_prescription, mode=equal)


# Solving
#------------------------------------------
initialstate   = initialize!(model);

stateXUA = solve(InverseSolver;
    initialstate=[initialstate],
    time=[inverseLoadSteps],
    verbose=true,
    maxiter= maxiterinv ,
    maxΔx  = maxΔx   ,
    maxΔλ  = maxΔλ   ,
    maxΔu  = maxΔu   ,
    maxΔa= maxΔa        ,
    saveiter = true
)

laststep = findlastassigned(stateXUA)
println("Last step ", laststep)
states = [stateXUA[i][1] for i in 1:laststep]

state = states[end]

staticdev, dynamicdev = getdof(state[end];class=:A,field=:γ₀, nodID=[Anod])[1], getdof(state[end];class=:A,field=:γ₁, nodID=[Anod])[1]
println("Static deviation : ", staticdev)
println("True static deviation : ", static_bias)
println("Dynamic deviation : ", dynamicdev)