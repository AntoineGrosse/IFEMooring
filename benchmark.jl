using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra, CSV, DataFrames, Statistics
using Muscade: equal
include("BiasedStrainGauge.jl")
include("MeshLine.jl")
include("MeshLineGauge.jl")
# include("ChainLink.jl")
include("ChainLink_new.jl")
include("PostProcessHelpers.jl")
currentDir = @__DIR__
cd(currentDir)


# Parameters
include("Input.jl")
@show nlines = 3

# Prescribed displacements
file_path = "input\\test_wo.csv"
# file_path_w = "input\\test_w.csv"
# file_path = "input\\test_wo_20years.csv"
# file_path_w = "input\\test_w_20years.csv"
prescribed_disp_interp = read_input(file_path)

# # xMotion1(t) = 0.
# # yMotion1(t) = 0.
# # zMotion1(t) = 0.
# xMotion1(t) = exp(min((t-0.5)/0.001,0)) * 3 * sin(0.5 * t)
# yMotion1(t) = exp(min((t-0.5)/0.001,0)) * 3 * sin(0.3 * t)
# zMotion1(t) = exp(min((t-0.5)/0.001,0)) * 1 * sin(0.1 * t)

# prescribed_disp_interp = [
#     [
#         xMotion1,
#         yMotion1,
#         zMotion1,
#     ],
# ]

##########################################
## Forward
##########################################
println("======= Forward solve =======")
@time begin
    #TODO Faire des structures , empacter tout ca
    topNodes = Vector{Muscade.NodID}(undef,nlines)
    nodeLists  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines)
    elementLists = Vector{Vector{Muscade.EleID}}(undef,nlines)
    anodeLists = Vector{Muscade.NodID}(undef,nlines)

    model = Model(:testline)
    for iline in 1:nlines
        local_azimuth = azimuths[iline]
        # Line
        coordTopNode = topInitCoord[iline]
        topNodes[iline] = addnode!(model,coordTopNode)
        nodeLists[iline],elementLists[iline],anodeLists[iline] = MeshLineGauge(model, topNodes[iline], local_azimuth, ChainLink, BiasedStrainGauge,xSection,segLength,nel)
        # nodeLists[iline],elementLists[iline] = MeshLine(model, topNodes[iline], local_azimuth, ChainLink, xSection,segLength,nel)

        # X Constraints : Anchor
        @functor with(offsetHorizontal,prestrechStaticAnalysis,staticFunction)   xMotionBottom(x,t,local_azimuth)  = SVector{1}([1.0 * (x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  staticFunction(t)*( offsetHorizontal  - prestrechStaticAnalysis )))])
        @functor with(offsetHorizontal,prestrechStaticAnalysis,staticFunction)   yMotionBottom(x,t,local_azimuth)  = SVector{1}([1.0 * (x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  staticFunction(t)*( offsetHorizontal  - prestrechStaticAnalysis )))])
        @functor with(offsetDownwards,staticFunction)                            zMotionBottom(x,t)                = SVector{1}([1.0 * (x[1] -                      (                           staticFunction(t)*( offsetDownwards                             )))])
        addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λat1,), gap=xMotionBottom, gargs=(local_azimuth,), mode=equal)
        addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λat2,), gap=yMotionBottom, gargs=(local_azimuth,), mode=equal)
        addelement!(model,DofConstraint,[nodeLists[iline][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λat3,), gap=zMotionBottom,                         mode=equal);
        
        # Buoys and Clampweights
        nodnum_buoygrav_topchain = 2
        nodnum_buoygrav_midpolyester = nseg - 1
        nodnum_buoygrav_bottomchain = nseg
        @functor with(g, staticFunction) buoygravForce_topchain(t)       = staticFunction(t) * (-3 + 0. ) * 1e3 * abs(g)
        @functor with(g, staticFunction) buoygravForce_midpolyester(t)   = staticFunction(t) * (-3 + 10.) * 1e3 * abs(g)
        @functor with(g, staticFunction) buoygravForce_bottomchain(t)    = staticFunction(t) * (-3 + 15.) * 1e3 * abs(g)
        addelement!(model,DofLoad,[nodeLists[iline][nodnum_buoygrav_topchain][1]];field=:t3,value=buoygravForce_topchain);  
        addelement!(model,DofLoad,[nodeLists[iline][nodnum_buoygrav_midpolyester][1]];field=:t3,value=buoygravForce_midpolyester);  
        addelement!(model,DofLoad,[nodeLists[iline][nodnum_buoygrav_bottomchain][1]];field=:t3,value=buoygravForce_bottomchain);  

        # Soil contact
        segnumsoil = nseg
        Kv = k_seabed_SIMA * 0.5 * segLength[end]/nel[end]
        [addelement!(model,SoilContact,[nodeLists[iline][segnumsoil][idxNod]],z₀=offsetDownwards,Kh=0.0,Kv=Kv,Ch=0.,Cv=0.)  for idxNod = 1:length(nodeLists[iline][segnumsoil])]
        
        # X Constraints : Top
        xMotion,yMotion,zMotion = prescribed_disp_interp[iline]
        @functor with() MotionTop(x,t,Motion)= SVector{1}([1.0 * (x[1] - Motion(t))])
        addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t1,), λinod=(1,), λclass=:X, λfield=(:λpt1,), gap=MotionTop, gargs = (xMotion,), mode=equal)
        addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t2,), λinod=(1,), λclass=:X, λfield=(:λpt2,), gap=MotionTop, gargs = (yMotion,), mode=equal)
        addelement!(model,DofConstraint,[topNodes[iline]],xinod=(1,),xfield=(:t3,), λinod=(1,), λclass=:X, λfield=(:λpt3,), gap=MotionTop, gargs = (zMotion,), mode=equal);
    end
    
    # Forward Static solve
    initialstate = initialize!(model)
    statesFWD_static = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=true, maxΔx=1e-6, maxiter=200)
    # Forward Dynamic solve
    statesFWD_dynamic = solve(SweepX{2};
    initialstate=statesFWD_static[end], 
    time=forwardTimesteps, 
    maxΔx=1e-6, 
    verbose=true, 
    maxiter=100,
    β=1/3.,γ=0.605, # Unconditionally stable for : γ >= .5 &&  2 β >= γ)
    )
    
    tension_range = (0,4e6)
    animateStates("Animation Forward analysis", 1:2:length(forwardTimesteps), 10., 1., waterDepth, statesFWD_dynamic, "figs/animationFWD_benchmark.mp4", tension_range)

end
