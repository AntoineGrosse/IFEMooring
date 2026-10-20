using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra, CSV, DataFrames, Statistics
using Muscade: equal
include("BiasedStrainGauge.jl")
include("MeshLine.jl")
include("MeshLineGauge.jl")
include("ChainLink.jl")
include("PostProcessHelpers.jl")
include("Material.jl")
currentDir = @__DIR__
cd(currentDir)

# Cost factors
#==NONE==#

# Strain std
#==NONE==#

# Parameters
@show maxΔx = 1e-5
@show maxΔa = 1e-5
@show maxiterinv = 12

# Load steps
staticLoadSteps = (-10:0.1:0)
time = (0 + eps():0.01:5)

# scaling
#==NONE==#

# Script booleans
#==NONE==#

# Plot booleans
#==NONE==#

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
#==DEFINEDBYSCRIPT==#

# Segments
nel = [5]
segLength = [150.]
xSection = [x1_mat]
nseg = length(nel)

# Geometry
waterDepth = 50.
fairleadDepth = 10.
offsetHorizontal = -10
offsetDownwards = -waterDepth + fairleadDepth
prestrechStaticAnalysis = sum(segLength) * 0.01

initX1 = 5
initY1 = 0.
initZ1 = -fairleadDepth

topInitCoord = [
    [initX1,initY1,initZ1],
]
nlines = length(topInitCoord)
yaw0 = 0.
azimuths = [
    (0. - yaw0) * π/180.,
]

# Loading inputs
#==NONE==#

# Prescribed displacements
#------------------------------------------
xMotion1(t) = 0.
yMotion1(t) = 0.
zMotion1(t) = 0.
# xMotion1(t) = sin(t)
# yMotion1(t) = sin(t)
# zMotion1(t) = sin(t)


prescribed_disp_interp = [
    [
        xMotion1,
        yMotion1,
        zMotion1,
    ],
]

##########################################
## Forward
##########################################
println("======= Forward solve =======")
@time begin
    #TODO Faire des structures , empacter tout ca
    topNodes = Vector{Muscade.NodID}(undef,nlines)
    nodeLists  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines*nseg)
    elementLists = Vector{Vector{Muscade.EleID}}(undef,nlines)
    anodeLists = Vector{Muscade.NodID}(undef,nlines)

    println("TOUT VA BIEN DEF LISTES")
    println(nlines)

    model = Model(:testline)
    for iline in 1:nlines
        local_azimuth = azimuths[iline]
        # Line
        coordTopNode = topInitCoord[iline]
        topNodes[iline] = addnode!(model,coordTopNode)
        # nodeLists[iline],elementLists[iline],anodeLists[iline] = MeshLineGauge(model, topNodes[iline], local_azimuth, ChainLink, BiasedStrainGauge,xSection,segLength,nel)
        nodeLists[iline],elementLists[iline] = MeshLine(model, topNodes[iline], local_azimuth, ChainLink, xSection,segLength,nel)

        # X Constraints : Anchor
        @functor with(offsetHorizontal,prestrechStaticAnalysis)   xMotionBottom(x,t,local_azimuth)  = SVector{1}([1.0 * (x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis )))])
        @functor with(offsetHorizontal,prestrechStaticAnalysis)   yMotionBottom(x,t,local_azimuth)  = SVector{1}([1.0 * (x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis )))])
        @functor with(offsetDownwards)                            zMotionBottom(x,t)                = SVector{1}([1.0 * (x[1] -                      (                           (min(t,-5.)+10)/5*( offsetDownwards                             )))])
        addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λat1,), gap=xMotionBottom, gargs=(local_azimuth,), mode=equal)
        addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λat2,), gap=yMotionBottom, gargs=(local_azimuth,), mode=equal)
        addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λat3,), gap=zMotionBottom,                         mode=equal);
        
        # Buoys and Clampweights
        #==NONE==#

        # Soil contact
        segnumsoil = nseg
        [addelement!(model,SoilContact,[nodeLists[iline][segnumsoil][idxNod]],z₀=offsetDownwards,Kh=0.0,Kv=Kv,Ch=0.,Cv=0.)  for idxNod = 1:length(nodeLists[iline][segnumsoil])]
        
        # X Constraints : Top
        xMotion,yMotion,zMotion = prescribed_disp_interp[iline]
        @functor with() MotionTop(x,t,Motion)= SVector{1}([1.0 * (x[1] - Motion(t))])
        addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λpt1,), gap=MotionTop, gargs = (xMotion,), mode=equal)
        addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λpt2,), gap=MotionTop, gargs = (yMotion,), mode=equal)
        addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λpt3,), gap=MotionTop, gargs = (zMotion,), mode=equal);
    end

    println("TOUT VA BIEN : DEF MODEL AND ALL")

    # Forward solve
    initialstate = initialize!(model)
    plotStaticStates("",1, initialstate, [initialstate], "figs/initial.png")
    staticStates = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)


    plotStaticStates("",length(staticLoadSteps), initialstate, staticStates, "figs/static.png")

end
