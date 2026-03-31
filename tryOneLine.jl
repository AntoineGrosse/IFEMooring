using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra
include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")
currentDir = @__DIR__
cd(currentDir)
##########################################
## Pre-Script
##########################################

# Some physical constants
#------------------------------------------ 
const g=9.81
const ρ=1025.

# Parameters inverse analysis
#------------------------------------------
nsteps = 66
Δtᵢₙᵥ             = 0.1               # Time step for the inverse analysis [s]
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()
nLoadSteps_inv = length(inverseLoadSteps)
InverseSolver        = DirectXUA{2,0,1}   # Dynamic solver for the inverse analysis
maxiterinv = 100
maxΔx = 1e-3
maxΔu = 1e-3
maxΔa = 1e-3
maxΔλ = Inf

static_bias = 0.0

Kv = 14000
yaw0 = 0. # yaw offset at initial condition

Cx = 1e9
Ca0 = 1e9
Ca1 = 1e9
σ = 1.e0

##########################################
## Initialisation
##########################################

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


# Segments description
#------------------------------------------
# ==============   TOP_CHAIN - TOP_ROPE - BOTTOM_ROPE - BOTTOM_CHAIN    ===================
# nel         = [10      ,    20     , 10] # Number of elements per segment
# segLength   = [10    ,    80.   , 10.] # Segment lengths
# xSection    = [x1_mat ,    x2_mat,  x1_mat]; # Cross-section type
nel         = [5      ,    23     , 12       ,        7     ] # Number of elements per segment
segLength   = [150    ,    414.   , 250.     ,        150.  ] # Segment lengths
xSection    = [x1_mat ,    x2_mat , x2_mat   ,        x1_mat]; # Cross-section type
nseg    = length(nel)
elLength = segLength./nel
lineLength = sum(segLength)

# Common line characteristics
#------------------------------------------
radialOffsetAnchor = 1000. # in case of a topnode being the platform center, we substract the fairlead radial offset to the static position of anchor 
waterDepth = 200
fairleadDepth = 10.
offsetHorizontal = radialOffsetAnchor - lineLength - 58.75  # adding also the fairlead offset in case of shared node
# radialOffsetAnchor = 1.037344398340249*lineLength # in case of a topnode being the platform center, we substract the fairlead radial offset to the static position of anchor 
# waterDepth = 0.2074688796680498*lineLength
# fairleadDepth = 10.
# offsetHorizontal = radialOffsetAnchor - lineLength - 0.060943983402489625*lineLength  # adding also the fairlead offset in case of shared node
offsetDownwards = -waterDepth + fairleadDepth  # adding also the fairlead offset in case of shared node
prestrechStaticAnalysis = lineLength*0.01;

initXPlatform = 0.
initYPlatform = 0.
initZPlatform = -fairleadDepth

statXPlatform = 0.
statYPlatform = 0.
statZPlatform = -fairleadDepth

refX1 = statXPlatform
refY1 = statYPlatform
refZ1 = statZPlatform
refX2 = statXPlatform
refY2 = statYPlatform
refZ2 = statZPlatform
refX3 = statXPlatform
refY3 = statYPlatform
refZ3 = statZPlatform

# Define the prescribed displacements on the x direction on node 2
nSamplesInverse = length(inverseLoadSteps);
ratioSample = 0.3
taperInv = floor(Int,ratioSample*nSamplesInverse)
x_disp = vcat(-(1/100 * lineLength) .* inverseLoadSteps[1:taperInv] ./ inverseLoadSteps[taperInv], -(1/100 * lineLength) * ones(nSamplesInverse-taperInv) ) # displacement causes a strain that is 60% of the bar inital length at the end timestep
# x_disp = -(30/100 * lineLength) .* inverseLoadSteps ./ inverseLoadSteps[end] # displacement causes a strain that is 60% of the bar inital length at the end timestep
x_disp_interp = linear_interpolation(vcat(-10.,inverseLoadSteps), vcat(0., x_disp))

# Line constructors
#------------------------------------------
azimuth = [
    (0. - yaw0) * π/180.,
]
topCoord = [
    [initXPlatform,initYPlatform,initZPlatform],
]
nlines = length(azimuth); 
prescribedTopMotion = [
        [x_disp_interp, nothing, nothing],
]
topStatCoord = [
    [refX1,refY1,refZ1],
]

# Buoy and soil contact
#------------------------------------------
nodnum_buoygrav_topchain = 2 # Buoy Number
nodnum_buoygrav_midpolyester = nseg-1 # Buoy Number
nodnum_buoygrav_bottomchain = nseg # Buoy Number
lastnodnum = nel[end]
segnumsoil = nseg # Soil contact number
@functor with() buoygravForce_topchain(t)           = ((min(t,-5.)+10)/5) *  (-3 + 0.)*1e3  * g ; #40. *ρ*g;
@functor with() buoygravForce_midpolyester(t)       = ((min(t,-5.)+10)/5) *  (-3 + 10.)*1e3 * g ; #40. *ρ*g;
@functor with() buoygravForce_bottomchain(t)        = ((min(t,-5.)+10)/5) *  (-3 + 15.)*1e3 * g ; #40. *ρ*g;

##########################################
## model forward
##########################################

model_ = Model(:testline)

topNode = Vector{Muscade.NodID}(undef,nlines)
nodeList  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines*nseg)
elementList = Vector{Vector{Muscade.EleID}}(undef,nlines)
aNode = Vector{Muscade.NodID}(undef,nlines)

# Nodes
#------------------------------------------   
idxLine = 1

local_azimuth = azimuth[idxLine]

topNode[idxLine] = addnode!(model_,topCoord[idxLine])

nodeList[idxLine],elementList[idxLine],aNode[idxLine] = MeshLineGauge(model_, topNode[idxLine], local_azimuth, Bar3D, StrainGaugeOnBar3D, xSection, segLength, nel)

# Constraints
#------------------------------------------
# Define the prescribed end displacements of the lower extremity
@functor with(offsetHorizontal,local_azimuth,prestrechStaticAnalysis)   xMotionBottom(x,t)= x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis ))
@functor with(offsetHorizontal,local_azimuth,prestrechStaticAnalysis)   yMotionBottom(x,t)= x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis ))
@functor with(offsetDownwards)                                          zMotionBottom(x,t)= x[1] -                      (                           (min(t,-5.)+10)/5*( offsetDownwards                             ))
addelement!(model_,DofConstraint,[nodeList[idxLine][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model_,DofConstraint,[nodeList[idxLine][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=yMotionBottom, mode=equal)
addelement!(model_,DofConstraint,[nodeList[idxLine][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal);

# add buoy
addelement!(model_,DofLoad,[nodeList[idxLine][nodnum_buoygrav_topchain][1]];field=:t3,value=buoygravForce_topchain);  
addelement!(model_,DofLoad,[nodeList[idxLine][nodnum_buoygrav_midpolyester][1]];field=:t3,value=buoygravForce_midpolyester);  
addelement!(model_,DofLoad,[nodeList[idxLine][nodnum_buoygrav_bottomchain][1]];field=:t3,value=buoygravForce_bottomchain);  

# add soil contact
[addelement!(model_,SoilContact,[nodeList[idxLine][segnumsoil][idxNod]],z₀=-waterDepth,Kh=0.0,Kv=Kv,Ch=0.,Cv=0.0)  for idxNod = 1:length(nodeList[idxLine][segnumsoil])]

# Define the prescribed end displacements of the top extremity
xMotion,yMotion,zMotion = prescribedTopMotion[idxLine]
refX,refY,refZ          = topStatCoord[idxLine]
@functor with(xMotion,refX) xMotionTop(x,t)= x[1] - ((min(t,-5.)+10)/5*(refX)    + xMotion(t))
@functor with() zeromotion(x,t)= x[1]
addelement!(model_,DofConstraint,[topNode[idxLine]],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model_,DofConstraint,[topNode[idxLine]],xinod=(1,),xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model_,DofConstraint,[topNode[idxLine]],xinod=(1,),xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Solving
#------------------------------------------
initialstate   = initialize!(model_);
staticLoadSteps = (-10:0.1:0)*1.
staticStates = solve(SweepX{0};initialstate,time=staticLoadSteps,verbose=false,maxΔx=1e-6,maxiter=60);

fig      = Figure(size = (1000,1000))
ax = Axis3(fig[1,1])
draw!(ax,initialstate)
display(fig)
# Plot the static analysis sequence
for stateIdx ∈ 1:length(staticLoadSteps)
    draw!(ax,staticStates[stateIdx])
end
display(fig)


stateForward = solve(SweepX{2};
    initialstate=staticStates[end],
    time=inverseLoadSteps,
    verbose=false,
    maxiter= maxiterinv
)

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

req = @request εₐₓ
out = getresult(stateForward,req,[elementList[idxLine][1]])
@show strain = [ out[idxEl].εₐₓ for idxEl ∈ 1:size(out,2)];


##########################################
## model inverse
##########################################

model = Model(:testline)

topNode = Vector{Muscade.NodID}(undef,nlines)
nodeList  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines*nseg)
elementList = Vector{Vector{Muscade.EleID}}(undef,nlines)
aNode = Vector{Muscade.NodID}(undef,nlines)

# Nodes
#------------------------------------------   
idxLine = 1

local_azimuth = azimuth[idxLine]

topNode[idxLine] = addnode!(model,topCoord[idxLine])

nodeList[idxLine],elementList[idxLine],aNode[idxLine] = MeshLineGauge(model, topNode[idxLine], local_azimuth, Bar3D, StrainGaugeOnBar3D, xSection,segLength,nel)

# Costs A
#------------------------------------------
@functor with() costA(a)             = Ca0 *    (a - 0.)^2
@functor with() costAother(a)        = Ca1 *    (a - 0.)^2
eAγ₀             = addelement!(model,SingleAcost,[aNode[idxLine]]; field=:γ₀,cost=costA)
eAγ₁             = addelement!(model,SingleAcost,[aNode[idxLine]]; field=:γ₁,cost=costAother)

# Costs X
#------------------------------------------
# for seg in 1:nseg 
#     for idxNode in 1:nel[seg]
#         ref = getdof(staticStates[end];class=:X,field=:t1, nodID=[nodeList[idxLine][seg][idxNode]])[1]
#         @functor with(ref) costX(x,t) = Cx *    (x .- ref)^2
#         addelement!(model,SingleDofCost,[nodeList[idxLine][seg][idxNode]]; class = :X, field=:t1 ,cost=costX)
#         ref = getdof(staticStates[end];class=:X,field=:t2, nodID=[nodeList[idxLine][seg][idxNode]])[1]
#         @functor with(ref) costX(x,t) = Cx *    (x .- ref)^2
#         addelement!(model,SingleDofCost,[nodeList[idxLine][seg][idxNode]]; class = :X, field=:t2 ,cost=costX)
#         ref = getdof(staticStates[end];class=:X,field=:t3, nodID=[nodeList[idxLine][seg][idxNode]])[1]
#         @functor with(ref) costX(x,t) = Cx *    (x .- ref)^2
#         addelement!(model,SingleDofCost,[nodeList[idxLine][seg][idxNode]]; class = :X, field=:t3 ,cost=costX)
#     end
# end

# Costs ε
#------------------------------------------
measured_strain_interp = linear_interpolation(vcat(-10.,inverseLoadSteps), vcat(0., strain .+ static_bias))
vec3(v,ind) = SVector{3}(v[i] for i∈ind);
element1 = elementList[idxLine][1]
@functor with(measured_strain_interp, element1, model, σ) function straincost(X,U,A, t)
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
    ε = motion⁻¹{P,ND}(ε_)
    ε = ∂0(ε)
    ############## Computation of the strain as it is done in the element bar ##############
    
    # Predicted strain
    εₚ = elestraingauge.ηₙ .* ((1 .+ A[2]) .* ε .+ A[1])
    # εₚ = elestraingauge.ηₙ .* ((1) .* ε .+ A[1])
    # εₚ = elestraingauge.ηₙ .* ((1) .* ε)
    println("Predicted strain : ", VALUE(εₚ))
    
    # Measured strain
    εₘ = measured_strain_interp(t)
    println("Measured strain : ", VALUE(εₘ))
    
    # Error
    Δε = εₚ .- εₘ
    
    # Compute the cost value
    cost_val = (Δε⋅Δε) / (2σ^2)

    return cost_val
end

# Adding the strain cost as a dof cost
edofcost             = addelement!(model,DofCost,[nodeList[idxLine][1][1],nodeList[idxLine][1][2],aNode[idxLine]]; 
    xinod=(1,1,1,2,2,2),xfield = (:t1,:t2,:t3,:t1,:t2,:t3),
    ainod=(3,3),afield = (:γ₀,:γ₁),
    cost = straincost)


# Constraints
#------------------------------------------
# Define the prescribed end displacements of the lower extremity
@functor with(offsetHorizontal,local_azimuth,prestrechStaticAnalysis)   xMotionBottom(x,t)= x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis ))
@functor with(offsetHorizontal,local_azimuth,prestrechStaticAnalysis)   yMotionBottom(x,t)= x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis ))
@functor with(offsetDownwards)                                          zMotionBottom(x,t)= x[1] -                      (                           (min(t,-5.)+10)/5*( offsetDownwards                             ))
addelement!(model,DofConstraint,[nodeList[idxLine][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
addelement!(model,DofConstraint,[nodeList[idxLine][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=yMotionBottom, mode=equal)
addelement!(model,DofConstraint,[nodeList[idxLine][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal);

# add buoy
addelement!(model,DofLoad,[nodeList[idxLine][nodnum_buoygrav_topchain][1]];field=:t3,value=buoygravForce_topchain);  
addelement!(model,DofLoad,[nodeList[idxLine][nodnum_buoygrav_midpolyester][1]];field=:t3,value=buoygravForce_midpolyester);  
addelement!(model,DofLoad,[nodeList[idxLine][nodnum_buoygrav_bottomchain][1]];field=:t3,value=buoygravForce_bottomchain);  

# add soil contact
[addelement!(model,SoilContact,[nodeList[idxLine][segnumsoil][idxNod]],z₀=-waterDepth,Kh=0.0,Kv=Kv,Ch=0.,Cv=0.0)  for idxNod = 1:length(nodeList[idxLine][segnumsoil])]

# Define the prescribed end displacements of the top extremity
xMotion,yMotion,zMotion = prescribedTopMotion[idxLine]
refX,refY,refZ          = topStatCoord[idxLine]
@functor with(xMotion,refX) xMotionTop(x,t)= x[1] - ((min(t,-5.)+10)/5*(refX)    + xMotion(t))
@functor with() zeromotion(x,t)= x[1]
addelement!(model,DofConstraint,[topNode[idxLine]],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
addelement!(model,DofConstraint,[topNode[idxLine]],xinod=(1,),xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=zeromotion, mode=equal)
addelement!(model,DofConstraint,[topNode[idxLine]],xinod=(1,),xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zeromotion, mode=equal)

# Solving
#------------------------------------------
initialstate   = initialize!(model);
staticLoadSteps = (-10:0.1:0)
staticStates = solve(SweepX{0};initialstate,time=staticLoadSteps,verbose=false,maxΔx=1e-6,maxiter=60);

fig_statinv      = Figure(size = (1000,1000))
ax_statinv = Axis3(fig_statinv[1,1])
draw!(ax_statinv,initialstate)
display(fig_statinv)
# Plot the static analysis sequence
for stateIdx ∈ 1:length(staticLoadSteps)
    draw!(ax_statinv,staticStates[stateIdx])
save("figs/inverse_static.png",fig_statinv)
end
display(fig_statinv)




stateXUA = solve(InverseSolver;
    initialstate=[staticStates[end]],
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

idxT = 50
idxS = 2

statechosen = [states[idxS][i] for i in eachindex(states[1])]
# Produce an animation
fig3   = Figure(size = (2000,1000))
ax3 = Axis3(fig3[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3))
xlims!(ax3,-1000,1000); ylims!(ax3,-1000,1000); zlims!(ax3,-waterDepth - 20,10)
graphic = draw!(ax3,staticStates[end])
ax3.azimuth[]=-π/2+π/180*10;
ax3.elevation[]=0+π/180*10;
framerate = 20
loadStepsIterator = 1:length(statechosen)
record(fig3, "figs/animationTimeInverse.mp4", loadStepsIterator;
framerate = framerate) do stateIdx
    draw!(graphic,statechosen[stateIdx])
end

statechosen = [states[i][idxT] for i in eachindex(states)]
# Produce an animation
fig3   = Figure(size = (2000,1000))
ax3 = Axis3(fig3[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3))
xlims!(ax3,-1000,1000); ylims!(ax3,-1000,1000); zlims!(ax3,-waterDepth - 20,10)
graphic = draw!(ax3,staticStates[end])
ax3.azimuth[]=-π/2+π/180*10;
ax3.elevation[]=0+π/180*10;
framerate = 20
loadStepsIterator = 1:length(statechosen)
record(fig3, "figs/animationStateInverse.mp4", loadStepsIterator;
framerate = framerate) do stateIdx
    draw!(graphic,statechosen[stateIdx])
end


state = states[end]

staticdev, dynamicdev = getdof(state[end];class=:A,field=:γ₀, nodID=[aNode[idxLine]])[1], getdof(state[end];class=:A,field=:γ₁, nodID=[aNode[idxLine]])[1]
println("Static deviation : ", staticdev)
println("True static deviation : ", static_bias)
println("Error static deviation : ", round(100 * abs(static_bias-staticdev)/staticdev ; digits=2) ," %")
println("\nDynamic deviation : ", dynamicdev)
println("True dynamic deviation : ", 0.)