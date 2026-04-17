using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra, CSV, DataFrames, Statistics
using Muscade: equal
include("BiasedStrainGauge.jl")
include("MeshLineGauge.jl")
include("ChainLink.jl")
include("PostProcessHelpers.jl")
currentDir = @__DIR__
cd(currentDir)

# Constants
g = 9.81
ρ = 1025.
@show Kv = 14000.
@show Cv = 0.

# Cost factors
# Strain
@show σ = 1e-2
# A cost
@show Ca0 = 1e-3
@show Ca1 = 1e3
# U cost
@show Cu0 = 1e-3
@show Cu1 = 1e-2
# X cost
@show Cx1 = 1e10
@show Cc1 = 1e0
@show Cs1 = 1e8 # Soil cost
@show dz = 0.5

# Parameters
@show maxΔx = 1e-5
@show maxΔa = 1e-6

# attenuationFactors = [0.5, 1.]
static_bias = 0.00003 # Static bias to retrieve
Δtᵢₙᵥ = 1
nsteps = 6e2
staticLoadSteps = (-10:0.1:0)
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()
InverseSolver = DirectXUA{2,0,1}

# Scaling    
@show scale = (
    X=(
        t1=1, 
        t2=1, 
        t3=1, 

        λat1=1e7,
        λat2=1e6,
        λat3=1e7,

        λpt1=1e7,
        λpt2=1e6,
        λpt3=1e7
    ),
    A=(
        γ₀=1e-3,
        γ₁=1e-6
    ),
    U=(
        ut1 = 1e-2, 
        ut2 = 1e-2, 
        ut3 = 1e-2, 

        upt1 = 1e6, 
        upt2 = 1e6, 
        upt3 = 1e6,

        uat1 = 1e6,
        uat2 = 1e6, 
        uat3 = 1e6
    ),
)
@show Λscale = 1e0 # The bigger the more Importance given to Residual over Costs, so the more importance given to Physical Model on Data Model 

# Script booleans
@show boolIntegrateBuoys = true
@show boolIntegrateSoilForward = true
@show boolIntegrateSoilInverse = true

@show boolCostSoil = false

@show boolUdofOnTop = false
@show boolUdofOnAnchor = false
@show boolUdofOnLine = false

@show boolXcostOnTop = false
@show boolXcostOnAnchor = false

@show boolXconstrOnTop = true
@show boolXconstrOnAnchor = true

@show boolStrainCost = true

# Plot booleans
boolPlotStatic = false
boolAnimateForward = true
boolAnimateInverse = true
boolPlotUdofs = false
boolPlotAdofs = false
boolPlotComparisonSIMA = true

# Loss functions
δ = 1e-2 # For Huber loss
quadra(x) = x⋅x # quadratic loss
expo(x) = 1 - exp(- x⋅x/1e10) # exponential loss
cauchy(x) = log(1 + x⋅x/1e10) # Cauchy loss
huber(x) = VALUE(∂0(x⋅x)) < δ ? 0.5 * x⋅x : δ * (sqrt(x⋅x) - 0.5*δ) # Huber loss
pseudo_huber(x) = δ^2 * (sqrt(1 + x⋅x/(δ^2)) - 1) # Pseudo huber loss
scaled_quadra(x) = sqrt(1 + x⋅x) # Custom
ch(x) = cosh(x) # Cosh
logch(x) = log(ch(x)) # Logcosh
mse(x) = (x⋅x/length(x))
loss_function = quadra

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
x1_mat         = AxisymmetricChainLinkCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, Caₜ=x1_Caₜ, Clₜ=x1_Clₜ, Cqₜ=x1_Cqₜ, Caₙ=x1_Caₙ, Clₙ=x1_Clₙ, Cqₙ=x1_Cqₙ)

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
x2_mat         = AxisymmetricChainLinkCrossSection(EA=x2_EA, μ=x2_μ, w=x2_w, Caₜ=x2_Caₜ, Clₜ=x2_Clₜ, Cqₜ=x2_Cqₜ, Caₙ=x2_Caₙ, Clₙ=x2_Clₙ, Cqₙ=x2_Cqₙ)

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

initX1 = 58.75
initY1 = 0.
initZ1 = -fairleadDepth
initX2 = -29.375
initY2 = 50.879
initZ2 = -fairleadDepth
initX3 = -29.375
initY3 = -50.879
initZ3 = -fairleadDepth

topInitCoord = [
    [initX1,initY1,initZ1],
    [initX2,initY2,initZ2],
    [initX3,initY3,initZ3],
]
nlines = length(topInitCoord)
yaw0 = 0.
azimuths = [
    (0. - yaw0) * π/180.,
    (120. - yaw0) * π/180.,
    (-120. - yaw0) * π/180.,
]

# Loading inputs
file_path = "input\\test_wo.csv"
file_path_w = "input\\test_w.csv"
df = CSV.read(file_path, DataFrame; delim=',')
df_w = CSV.read(file_path_w, DataFrame; delim=',')
firstX1 = df[:,"Xfairlead1 [m]"][1]
firstY1 = df[:,"Yfairlead1 [m]"][1]
firstZ1 = df[:,"Zfairlead1 [m]"][1]
firstX2 = df[:,"Xfairlead2 [m]"][1]
firstY2 = df[:,"Yfairlead2 [m]"][1]
firstZ2 = df[:,"Zfairlead2 [m]"][1]
firstX3 = df[:,"Xfairlead3 [m]"][1]
firstY3 = df[:,"Yfairlead3 [m]"][1]
firstZ3 = df[:,"Zfairlead3 [m]"][1]
nSamples = length(df[:,"time"]);
ratioSampleTaper = 0.02
taper = floor(Int,ratioSampleTaper*nSamples)
ramp = vcat(LinRange(0.,1.,taper),ones(nSamples-2*taper),LinRange(1.,0.,taper))

# Iterative continuation loop
#------------------------------------------
iterContinuation, attenuationFactor = (1,1.)

# Prescribed displacements
#------------------------------------------

xMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead1 [m]"].- firstX1).*ramp .* attenuationFactor,0.))
yMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead1 [m]"].- firstY1).*ramp .* attenuationFactor,0.))
zMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead1 [m]"].- firstZ1).*ramp .* attenuationFactor,0.))
xMotion2 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead2 [m]"].- firstX2).*ramp .* attenuationFactor,0.))
yMotion2 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead2 [m]"].- firstY2).*ramp .* attenuationFactor,0.))
zMotion2 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead2 [m]"].- firstZ2).*ramp .* attenuationFactor,0.))
xMotion3 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead3 [m]"].- firstX3).*ramp .* attenuationFactor,0.))
yMotion3 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead3 [m]"].- firstY3).*ramp .* attenuationFactor,0.))
zMotion3 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead3 [m]"].- firstZ3).*ramp .* attenuationFactor,0.))

prescribed_disp_interp = [
    [
        xMotion1,
        yMotion1,
        zMotion1,
    ],
    [
        xMotion2,
        yMotion2,
        zMotion2,
    ],
    [
        xMotion3,
        yMotion3,
        zMotion3,
    ],
]


##########################################
## Forward
##########################################

topNodes = Vector{Muscade.NodID}(undef,nlines)
nodeLists  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines*nseg)
elementLists = Vector{Vector{Muscade.EleID}}(undef,nlines)
anodeLists = Vector{Muscade.NodID}(undef,nlines)

model = Model(:testline)
for iline in 1:nlines
    local_azimuth = azimuths[iline]
    # Line
    coordTopNode = topInitCoord[iline]
    topNodes[iline] = addnode!(model,coordTopNode)
    nodeLists[iline],elementLists[iline],anodeLists[iline] = MeshLineGauge(model, topNodes[iline], local_azimuth, ChainLink, BiasedStrainGauge,xSection,segLength,nel)

    # X Constraints : Anchor
    @functor with(offsetHorizontal,prestrechStaticAnalysis)   xMotionBottom(x,t,local_azimuth)  = SVector{1}([Cc1 * (x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis )))])
    @functor with(offsetHorizontal,prestrechStaticAnalysis)   yMotionBottom(x,t,local_azimuth)  = SVector{1}([Cc1 * (x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis )))])
    @functor with(offsetDownwards)                            zMotionBottom(x,t)                = SVector{1}([Cc1 * (x[1] -                      (                           (min(t,-5.)+10)/5*( offsetDownwards                             )))])
    addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λat1,), gap=xMotionBottom, gargs=(local_azimuth,), mode=equal)
    addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λat2,), gap=yMotionBottom, gargs=(local_azimuth,), mode=equal)
    addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λat3,), gap=zMotionBottom,                         mode=equal);
    
    # Buoys and Clampweights
    if boolIntegrateBuoys
        nodnum_buoygrav_topchain = 2
        nodnum_buoygrav_midpolyester = nseg - 1
        nodnum_buoygrav_bottomchain = nseg
        @functor with() buoygravForce_topchain(t)       = ((min(t,-5.)+10)/5) * (-3 + 0. ) * 1e3 * g
        @functor with() buoygravForce_midpolyester(t)   = ((min(t,-5.)+10)/5) * (-3 + 10.) * 1e3 * g
        @functor with() buoygravForce_bottomchain(t)    = ((min(t,-5.)+10)/5) * (-3 + 15.) * 1e3 * g
        addelement!(model,DofLoad,[nodeLists[iline][nodnum_buoygrav_topchain][1]];field=:t3,value=buoygravForce_topchain);  
        addelement!(model,DofLoad,[nodeLists[iline][nodnum_buoygrav_midpolyester][1]];field=:t3,value=buoygravForce_midpolyester);  
        addelement!(model,DofLoad,[nodeLists[iline][nodnum_buoygrav_bottomchain][1]];field=:t3,value=buoygravForce_bottomchain);  
    end

    # Soil contact
    if boolIntegrateSoilForward
        segnumsoil = nseg
        [addelement!(model,SoilContact,[nodeLists[iline][segnumsoil][idxNod]],z₀=offsetDownwards,Kh=0.0,Kv=Kv,Ch=0.,Cv=Cv)  for idxNod = 1:length(nodeLists[iline][segnumsoil])]
    end
    
    # X Constraints : Top
    xMotion,yMotion,zMotion = prescribed_disp_interp[iline]
    @functor with() MotionTop(x,t,Motion)= SVector{1}([Cc1 * (x[1] - Motion(t))])
    addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λpt1,), gap=MotionTop, gargs = (xMotion,), mode=equal)
    addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λpt2,), gap=MotionTop, gargs = (yMotion,), mode=equal)
    addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λpt3,), gap=MotionTop, gargs = (zMotion,), mode=equal);
end

# Forward solve
initialstate = initialize!(model)
staticStates = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

if boolPlotStatic
    plotStaticStates("",length(staticLoadSteps), initialstate, staticStates, "figs/static.png")
end

stateForward = solve(SweepX{2};
    initialstate=staticStates[end], 
    time=inverseLoadSteps, 
    maxΔx=1e-6, 
    verbose=false, 
    maxiter=100,
    β=1/3.,γ=0.605, # Unconditionally stable for : γ >= .5 &&  2 β >= γ)
)

# Making artificial measurement data from forward solving
measured_strain_list = Vector{}(undef,nlines)
for iline in 1:nlines
    req = @request εₐₓ
    out = getresult(stateForward, req, [elementLists[iline][1]])
    strain = [out[idxEl].εₐₓ for idxEl in axes(out,2)]
    println("For line $(iline), sensor bias is $(round(static_bias/std(strain)*100)) % of the strain std.")
    measured_strain_list[iline] = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain .+ static_bias))
end

if boolAnimateForward
    animateStates("Animation Forward analysis", 1:3:length(inverseLoadSteps), 10., 10., waterDepth, stateForward, "figs/animationForward.mp4")
end

if boolPlotComparisonSIMA
    Fgps = extractAxialForceFromStrainGauge(stateForward, [elementLists[iline][1]  for iline in 1:nlines], length(inverseLoadSteps))
    Disps = extractDisplacements(stateForward, [(iline,1,1) for iline in 1:nlines], nodeLists, length(inverseLoadSteps))
    xs = [Disps[1][:,iline] for iline in 1:nlines]
    ys = [Disps[2][:,iline] for iline in 1:nlines]
    zs = [Disps[3][:,iline] for iline in 1:nlines]
    plotComparisonWithSIMA(prescribed_disp_interp, xs,ys,zs, inverseLoadSteps, taper, Fgps, df, df_w, "test", 120, 180)
end

##########################################
## Inverse
##########################################
# Penalties - adjusted for larger strains
@functor with() costA(a) = loss_function(sqrt(Ca0) * (a - 0.))
@functor with() costAother(a) = loss_function(sqrt(Ca1) * (a - 0.))
@functor with() costUother(u,t) = loss_function(sqrt(Cu1) * (u - 0.))

vec3(v,ind) = SVector{3}(v[i] for i∈ind)

topNodes_inv = Vector{Muscade.NodID}(undef,nlines)
nodeLists_inv  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines*nseg)
elementLists_inv = Vector{Vector{Muscade.EleID}}(undef,nlines)
anodeLists_inv = Vector{Muscade.NodID}(undef,nlines)

model_inv = Model(:testline)
for iline in 1:nlines
    local_azimuth = azimuths[iline]
    # Line
    coordTopNode = topInitCoord[iline]
    topNodes_inv[iline] = addnode!(model_inv,coordTopNode)
    nodeLists_inv[iline],elementLists_inv[iline],anodeLists_inv[iline] = MeshLineGauge(model_inv, topNodes_inv[iline], local_azimuth, ChainLink, BiasedStrainGauge,xSection,segLength,nel)

    # X Constraints : Anchor
    anchorConstr = boolXconstrOnAnchor ? :equal : :off
    @functor with() modeconstraintAnchor(t) = t < eps() ? :equal : anchorConstr
    @functor with(offsetHorizontal,prestrechStaticAnalysis)   xMotionBottom(x,t,local_azimuth)  = SVector{1}([Cc1 * (x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis )))])
    @functor with(offsetHorizontal,prestrechStaticAnalysis)   yMotionBottom(x,t,local_azimuth)  = SVector{1}([Cc1 * (x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis )))])
    @functor with(offsetDownwards)                            zMotionBottom(x,t)                = SVector{1}([Cc1 * (x[1] -                      (                           (min(t,-5.)+10)/5*( offsetDownwards                             )))])
    addelement!(model_inv,DofConstraint,[nodeLists_inv[iline][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λat1,), gap=xMotionBottom, gargs=(local_azimuth,), mode=modeconstraintAnchor)
    addelement!(model_inv,DofConstraint,[nodeLists_inv[iline][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λat2,), gap=yMotionBottom, gargs=(local_azimuth,), mode=modeconstraintAnchor)
    addelement!(model_inv,DofConstraint,[nodeLists_inv[iline][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λat3,), gap=zMotionBottom,                         mode=modeconstraintAnchor);
    
    # Buoys and Clampweights
    if boolIntegrateBuoys
        nodnum_buoygrav_topchain = 2
        nodnum_buoygrav_midpolyester = nseg - 1
        nodnum_buoygrav_bottomchain = nseg
        @functor with() buoygravForce_topchain(t)       = ((min(t,-5.)+10)/5) * (-3 + 0. ) * 1e3 * g
        @functor with() buoygravForce_midpolyester(t)   = ((min(t,-5.)+10)/5) * (-3 + 10.) * 1e3 * g
        @functor with() buoygravForce_bottomchain(t)    = ((min(t,-5.)+10)/5) * (-3 + 15.) * 1e3 * g
        addelement!(model_inv,DofLoad,[nodeLists_inv[iline][nodnum_buoygrav_topchain][1]];field=:t3,value=buoygravForce_topchain);  
        addelement!(model_inv,DofLoad,[nodeLists_inv[iline][nodnum_buoygrav_midpolyester][1]];field=:t3,value=buoygravForce_midpolyester);  
        addelement!(model_inv,DofLoad,[nodeLists_inv[iline][nodnum_buoygrav_bottomchain][1]];field=:t3,value=buoygravForce_bottomchain);  
    end

    # Soil contact
    if boolIntegrateSoilInverse
        segnumsoil = nseg
        [addelement!(model_inv,SoilContact,[nodeLists_inv[iline][segnumsoil][idxNod]],z₀=offsetDownwards,Kh=0.0,Kv=Kv,Ch=0.,Cv=Cv)  for idxNod = 1:length(nodeLists_inv[iline][segnumsoil])]
    end
    
    # X Constraints : Top
    xMotion,yMotion,zMotion = prescribed_disp_interp[iline]
    topConstr = boolXconstrOnTop ? :equal : :off
    @functor with() modeconstraintTop(t) = t < eps() ? :equal : topConstr
    @functor with() MotionTop(x,t,Motion)= SVector{1}([Cc1 * (x[1] - Motion(t))])
    addelement!(model_inv,DofConstraint,[topNodes_inv[iline]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λpt1,), gap=MotionTop, gargs=(xMotion,), mode=modeconstraintTop)
    addelement!(model_inv,DofConstraint,[topNodes_inv[iline]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λpt2,), gap=MotionTop, gargs=(yMotion,), mode=modeconstraintTop)
    addelement!(model_inv,DofConstraint,[topNodes_inv[iline]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λpt3,), gap=MotionTop, gargs=(zMotion,), mode=modeconstraintTop);

    # Strain cost
    element1 = elementLists_inv[iline][1]
    measured_strain_interp = measured_strain_list[iline]
    @functor with(measured_strain_interp, element1, model_inv, σ) function straincost(X,U,A,t)
        elestraingauge = model_inv.eleobj[element1]
        elebar = elestraingauge.eleobj
        
        # X is a Tuple with one element: a 6-element SVector
        # DOFs 1-3 are from node 1, DOFs 4-6 are from node 2
        Xvec = X[1]
        uᵧ₁ = vec3(Xvec, 1:3)
        uᵧ₂ = vec3(Xvec, 4:6)
        
        # Compute tangent vector from displacements in global coordinates
        tg = elebar.tgₘ + uᵧ₂ - uᵧ₁
        L = √(tg[1]^2+tg[2]^2+tg[3]^2)
        ε_val = max(eps(),L/elebar.Lₛ - 1)
        
        # Compute strain with full AD w.r.t. both A and X
        # A[2] is multiplier, A[1] is bias, both are variated by solver
        εₚ = elestraingauge.ηₙ[1] * ((1 + A[2]) * ε_val + A[1]) # we use  "elestraingauge.ηₙ[1]" since we have only one strain gauge
        εₘ_val = measured_strain_interp(t)
        Δε = εₚ - εₘ_val
        cost_val = loss_function(Δε / σ)

        return cost_val
    end

    # A costs
    eAγ₀ = addelement!(model_inv, SingleAcost, [anodeLists_inv[iline]]; field=:γ₀, cost=costA)
    eAγ₁ = addelement!(model_inv, SingleAcost, [anodeLists_inv[iline]]; field=:γ₁, cost=costAother)
    # U costs
    if boolUdofOnTop
        @functor with() costUpx(u,t) = loss_function(sqrt(Cu0) * (u))
        @functor with() costUpy(u,t) = loss_function(sqrt(Cu0) * (u))
        @functor with() costUpz(u,t) = loss_function(sqrt(Cu0) * (u))
        eUtop1 = addelement!(model_inv, SingleUdof, [topNodes_inv[iline]]; Xfield=:t1,Ufield=:upt1,cost=costUpx)
        eUtop2 = addelement!(model_inv, SingleUdof, [topNodes_inv[iline]]; Xfield=:t2,Ufield=:upt2,cost=costUpy)
        eUtop3 = addelement!(model_inv, SingleUdof, [topNodes_inv[iline]]; Xfield=:t3,Ufield=:upt3,cost=costUpz)
    end
    if boolUdofOnAnchor
        @functor with()    costUax(u,t,local_azimuth) = loss_function(sqrt(Cu0) * (u - cos(local_azimuth) * 1.5e6))
        @functor with()    costUay(u,t,local_azimuth) = loss_function(sqrt(Cu0) * (u - sin(local_azimuth) * 1.5e6))
        @functor with()    costUaz(u,t)               = loss_function(sqrt(Cu0) * (u - 3e5))
        eUanch1 = addelement!(model_inv, SingleUdof, [nodeLists_inv[iline][end][end]]; Xfield=:t1,Ufield=:uat1,cost=costUax, costargs=(local_azimuth,))
        eUanch2 = addelement!(model_inv, SingleUdof, [nodeLists_inv[iline][end][end]]; Xfield=:t2,Ufield=:uat2,cost=costUay, costargs=(local_azimuth,))
        eUanch3 = addelement!(model_inv, SingleUdof, [nodeLists_inv[iline][end][end]]; Xfield=:t3,Ufield=:uat3,cost=costUaz)
    end
    if boolUdofOnLine
        # eUt1 = [addelement!(model_inv, SingleUdof, [nodeLists_inv[iline][iseg][inod]]; Xfield=:t1,Ufield=:ut1,cost=costUother) for iseg in nseg:nseg for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
        # eUt2 = [addelement!(model_inv, SingleUdof, [nodeLists_inv[iline][iseg][inod]]; Xfield=:t2,Ufield=:ut2,cost=costUother) for iseg in nseg:nseg for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
        eUt3 = [addelement!(model_inv, SingleUdof, [nodeLists_inv[iline][iseg][inod]]; Xfield=:t3,Ufield=:ut3,cost=costUother) for iseg in nseg:nseg for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
    end
    # Top motion (perfect) measurement X cost
    if boolXcostOnTop
        xMotion,yMotion,zMotion = prescribed_disp_interp[iline]
        @functor with() CostTop(x,t,Motion) = loss_function(sqrt(Cx1) * (x - Motion(t)))
        eXtopt1 = addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][1][1]]; class=:X,field=:t1,cost=CostTop,costargs=(xMotion,))
        eXtopt2 = addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][1][1]]; class=:X,field=:t2,cost=CostTop,costargs=(yMotion,))
        eXtopt3 = addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][1][1]]; class=:X,field=:t3,cost=CostTop,costargs=(zMotion,))
    end
    # Anchor motion (perfect) measurement X cost
    if boolXcostOnAnchor
        @functor with(offsetHorizontal, prestrechStaticAnalysis)     xCostAnch(x,t,local_azimuth)   = loss_function(sqrt(Cx1) * (x - cos(local_azimuth) *   (prestrechStaticAnalysis +  (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))))
        @functor with(offsetHorizontal, prestrechStaticAnalysis)     yCostAnch(x,t,local_azimuth)   = loss_function(sqrt(Cx1) * (x - sin(local_azimuth) *   (prestrechStaticAnalysis +  (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))))
        @functor with(offsetDownwards)                               zCostAnch(x,t)                 = loss_function(sqrt(Cx1) * (x -                        (                           (min(t,-5.)+10)/5 * offsetDownwards                             )))
        eXancht1 = addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][end][end]]; class=:X,field=:t1,cost=xCostAnch,costargs=(local_azimuth,))
        eXancht2 = addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][end][end]]; class=:X,field=:t2,cost=yCostAnch,costargs=(local_azimuth,))
        eXancht3 = addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][end][end]]; class=:X,field=:t3,cost=zCostAnch)
    end
    # Strain measurement cost
    if boolStrainCost
        edofcost = addelement!(model_inv, DofCost, [nodeLists_inv[iline][1][1], nodeLists_inv[iline][1][2], anodeLists_inv[iline]];
            xinod=(1,1,1,2,2,2), xfield=(:t1,:t2,:t3,:t1,:t2,:t3),
            ainod=(3,3), afield=(:γ₀,:γ₁),
            cost=straincost)
    end
    # Soil cost
    if boolCostSoil
        @functor with() CostSoil(x,t,z₀) = z₀ + dz < x ? loss_function(sqrt(Cs1) * 2*dz) : loss_function(sqrt(Cs1) * (x - (z₀-dz)))
        esoilcost = [addelement!(model_inv, SingleDofCost, [nodeLists_inv[iline][iseg][inod]]; class=:X,field=:t3,cost=CostSoil,costargs=(offsetDownwards,)) for iseg in nseg:nseg for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
    end
end

# Scaling
# Use the same order-of-magnitude separation suggested by study_scale,
# but keep the model two-parameter (γ₀ and γ₁).
setscale!(model_inv; scale = scale, Λscale = Λscale)

initialstate_inv = initialize!(model_inv)

staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

traj = Vector{Muscade.State}(undef,length(inverseLoadSteps))
intermediateState = nothing
let primer = staticStates_inv[end]
    for (it,ti) in enumerate(inverseLoadSteps)
        println("=== Inverse static for timestep $(it)/$(length(inverseLoadSteps)) ===")
        # Solve X
        itermax = 200
        local stateX = solve(DirectXUA{0,0,1};
        initialstate=[primer],
        time=[ti-eps():eps():ti],
        verbose=false,
        maxiter=itermax,
        maxΔx=5e-3,   # More relaxed convergence (was 1e-3, keep for now)
        maxΔu=Inf,
        maxΔλ=Inf,
        saveiter=true,
        )
        
        local laststepX = findlastassigned(stateX)
        println("Stopped after iteration $(laststepX)/$(itermax)")
        traj[it] = stateX[laststepX][1][end]
        global intermediateState = stateX[laststepX][1][end]
        primer = intermediateState
    end
end
initialtrajectory = [traj]


# Solve XUA
stateXUA = solve(InverseSolver;
    initialstate=[intermediateState],
    initialtrajectory = initialtrajectory,
    time=[inverseLoadSteps],
    verbose=true,
    maxiter=20,
    maxΔx=maxΔx,   # More relaxed convergence (was 1e-3, keep for now)
    maxΔu=Inf,
    maxΔa=maxΔa,   # REDUCED from 1e-3 to focus on A convergence
    maxΔλ=Inf,
    saveiter=true
)

laststep = findlastassigned(stateXUA)
if laststep > 0
    state = stateXUA[laststep][1]
    staticdev = VALUE(state[end].A[1])
    dynamicdev = VALUE(state[end].A[2])
    println("Converged: Static dev Line 1 = $staticdev, True = $static_bias, Error = $(abs(staticdev - static_bias)/static_bias * 100)%")
    println("Dynamic dev Line 1 = $dynamicdev")
    staticdev = VALUE(state[end].A[3])
    dynamicdev = VALUE(state[end].A[4])
    println("Converged: Static dev Line 2 = $staticdev, True = $static_bias, Error = $(abs(staticdev - static_bias)/static_bias * 100)%")
    println("Dynamic dev Line 2 = $dynamicdev")
    staticdev = VALUE(state[end].A[5])
    dynamicdev = VALUE(state[end].A[6])
    println("Converged: Static dev Line 3 = $staticdev, True = $static_bias, Error = $(abs(staticdev - static_bias)/static_bias * 100)%")
    println("Dynamic dev Line 3 = $dynamicdev")
else
    println("Did not converge")
end

if boolAnimateInverse
    animateStates("Animation inverse reconstruction", 1:3:length(inverseLoadSteps), 10., 10., waterDepth, state, "figs/animationInverse.mp4")
    states_reconstruction = [stateXUA[i][1][end] for i in 1:laststep]
    animateStates("Animation inverse iterations", 1:1:laststep, 10., 10., waterDepth, states_reconstruction, "figs/reconstructionIterations.mp4")
end

Fgps = extractAxialForceFromStrainGauge(state, [elementLists_inv[iline][1]  for iline in 1:nlines], length(inverseLoadSteps))
Disps = extractDisplacements(state, [(iline,1,1) for iline in 1:nlines], nodeLists_inv, length(inverseLoadSteps))
xs = [Disps[1][:,iline] for iline in 1:nlines]
ys = [Disps[2][:,iline] for iline in 1:nlines]
zs = [Disps[3][:,iline] for iline in 1:nlines]

plotComparisonWithSIMA(prescribed_disp_interp, xs,ys,zs, inverseLoadSteps, taper, Fgps, df, df_w, "testinverse", 120, 180)

if boolPlotUdofs
    UdofsAnchor = extractUdofs(state, [(iline,nseg,nel[nseg]) for iline in 1:nlines], [:uat1,:uat2,:uat3], nodeLists_inv, length(inverseLoadSteps))
    UdofsTop = extractUdofs(state, [(iline,1,1) for iline in 1:nlines], [:upt1,:upt2,:upt3], nodeLists_inv, length(inverseLoadSteps))

    #plotting for anchor
    for iline in 1:nlines
        figU      = Figure(size = (1000,1000))
        axUt1 = Axis(figU[1, 1],ylabel="UXforce at anchor [kN]",xlabel = "Time (s)")
        lines!(axUt1, inverseLoadSteps, UdofsAnchor[1][:,iline]/1e3, color = :blue,   linestyle = :solid ,   label="Ut1")
        axislegend()
        axUt2 = Axis(figU[2, 1],ylabel="UYforce at anchor [kN]",xlabel = "Time (s)")
        lines!(axUt2, inverseLoadSteps, UdofsAnchor[2][:,iline]/1e3, color = :blue,   linestyle = :solid ,   label="Ut2")
        axislegend()
        axUt3 = Axis(figU[3, 1],ylabel="UZforce at anchor [kN]",xlabel = "Time (s)")
        lines!(axUt3, inverseLoadSteps, UdofsAnchor[3][:,iline]/1e3, color = :blue,   linestyle = :solid ,   label="Ut3")
        axislegend()
        save("figs/UdofsAnchor$(iline).png", figU)
    end

    #plotting for fairlead
    for iline in 1:nlines
        figU      = Figure(size = (1000,1000))
        axUt1 = Axis(figU[1, 1],ylabel="UXforce at fairlead [kN]",xlabel = "Time (s)")
        lines!(axUt1, inverseLoadSteps, UdofsTop[1][:,iline]/1e3, color = :blue,   linestyle = :solid ,   label="Ut1")
        axislegend()
        axUt2 = Axis(figU[2, 1],ylabel="UYforce at fairlead [kN]",xlabel = "Time (s)")
        lines!(axUt2, inverseLoadSteps, UdofsTop[2][:,iline]/1e3, color = :blue,   linestyle = :solid ,   label="Ut2")
        axislegend()
        axUt3 = Axis(figU[3, 1],ylabel="UZforce at fairlead [kN]",xlabel = "Time (s)")
        lines!(axUt3, inverseLoadSteps, UdofsTop[3][:,iline]/1e3, color = :blue,   linestyle = :solid ,   label="Ut3")
        axislegend()
        save("figs/UdofsTop$(iline).png", figU)
    end

    DispsAnchor = extractDisplacements(state, [(iline,4,nel[4]-1) for iline in 1:nlines], nodeLists_inv, length(inverseLoadSteps))
    #plotting for fairlead
    for iline in 1:nlines
        local az = azimuths[iline]
        iX,iY,iZ = offsetHorizontal * cos(az), offsetHorizontal * sin(az), offsetDownwards
        figX      = Figure(size = (1000,1000))
        axXt1 = Axis(figX[1, 1],ylabel="x at anchor [m]",xlabel = "Time (s)")
        lines!(axXt1, inverseLoadSteps, DispsAnchor[1][:,iline] .- iX, color = :green,   linestyle = :solid ,   label="t1")
        axislegend()
        axXt2 = Axis(figX[2, 1],ylabel="y at anchor [m]",xlabel = "Time (s)")
        lines!(axXt2, inverseLoadSteps, DispsAnchor[2][:,iline] .- iY, color = :green,   linestyle = :solid ,   label="t2")
        axislegend()
        axXt3 = Axis(figX[3, 1],ylabel="z at anchor [m]",xlabel = "Time (s)")
        lines!(axXt3, inverseLoadSteps, DispsAnchor[3][:,iline] .- iZ, color = :green,   linestyle = :solid ,   label="t3")
        axislegend()
        save("figs/XdofsAnchor$(iline).png", figX)
    end

    for iline in 1:nlines
        strain = measured_strain_list[iline](inverseLoadSteps)
        println("For line $(iline), sensor bias is $(round(static_bias/std(strain)*100)) % of the strain std.")
    end
end