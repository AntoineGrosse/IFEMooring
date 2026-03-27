using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, CSV, DataFrames, Statistics
currentDir = @__DIR__
cd(currentDir)

include("BiasedStrainGaugeOnBarElement.jl")
# include("MeshLine.jl")
include("MeshLineGauge.jl")


##########################################
## Pre-Script
##########################################

# Script booleans
#------------------------------------------
bSameNode = true

bForwardAnalysis = true
bPlotStaticAnalysis = false
bPlotForwardAnalysis = false
bForwardAnimation = false

bInverseAnalysis = true
bNoise = false
bPlotInverseAnalysis = true
bInverseAnimation = false

bFromSavedDatas = false
bSaveForward = false
bSaveInverse = false


# Save and Load
#------------------------------------------
file_path = "input\\test_wo.csv"
file_path_w = "input\\test_w.csv"
runName = "test"

# Some physical constants
#------------------------------------------ 
const g=9.81
const ρ=1025.
const waterDepth = 200.
Kv = 14000
yaw0 = 0. # yaw offset at initial condition


# Parameters static analysis
#------------------------------------------
staticLoadSteps = -10:.1:0
nStaticLoadSteps = length(staticLoadSteps)

# Parameters dynamic analysis
#------------------------------------------
Δt = 0.3
nsteps = 6e2
dynamicLoadSteps = Δt:Δt:nsteps*Δt
nDynamicLoadSteps = length(dynamicLoadSteps)
ForwardSolver = SweepX{2}
maxiterdyn = 50
checkResultsUntilStep = 500


# Parameters inverse analysis
#------------------------------------------
Δtᵢₙᵥ             = 0.3               # Time step for the inverse analysis [s]
inverseLoadSteps = Δt:Δtᵢₙᵥ:nsteps*Δt
# inverseLoadSteps = -10.:Δtᵢₙᵥ:nsteps*Δt
nLoadSteps_inv = length(inverseLoadSteps)
InverseSolver        = DirectXUA{2,0,1}   # Dynamic solver for the inverse analysis
maxiterinv = 10
maxΔx = 1e-5
maxΔu = 1e-5
maxΔa = 1e-6
maxΔλ = Inf

# Saving
#------------------------------------------



##########################################
## Initialisation
##########################################

# Read the TSV file into a DataFrame
df = CSV.read(file_path, DataFrame; delim=',')
df_w = CSV.read(file_path_w, DataFrame; delim=',')

# Mean motions
μX1 = mean(df[:,"Xfairlead1 [m]"])
μY1 = mean(df[:,"Yfairlead1 [m]"])
μZ1 = mean(df[:,"Zfairlead1 [m]"])
μX2 = mean(df[:,"Xfairlead2 [m]"])
μY2 = mean(df[:,"Yfairlead2 [m]"])
μZ2 = mean(df[:,"Zfairlead2 [m]"])
μX3 = mean(df[:,"Xfairlead3 [m]"])
μY3 = mean(df[:,"Yfairlead3 [m]"])
μZ3 = mean(df[:,"Zfairlead3 [m]"])
μXPlatform = mean(df[:,"Surge [m]"])
μYPlatform = mean(df[:,"Sway [m]"])
μZPlatform = mean(df[:,"Heave [m]"])

initX1 = 58.75
initY1 = 0
initZ1 = -10
initX2 = -29.375
initY2 = 50.879
initZ2 = -10
initX3 = -29.375
initY3 = -50.879
initZ3 = -10

statX1 = df[:,"Xfairlead1 [m]"][1]
statY1 = df[:,"Yfairlead1 [m]"][1]
statZ1 = df[:,"Zfairlead1 [m]"][1]
statX2 = df[:,"Xfairlead2 [m]"][1]
statY2 = df[:,"Yfairlead2 [m]"][1]
statZ2 = df[:,"Zfairlead2 [m]"][1]
statX3 = df[:,"Xfairlead3 [m]"][1]
statY3 = df[:,"Yfairlead3 [m]"][1]
statZ3 = df[:,"Zfairlead3 [m]"][1]
statXPlatform = df[:,"Surge [m]"][1]
statYPlatform = df[:,"Sway [m]"][1]
statZPlatform = df[:,"Heave [m]"][1]

if bSameNode
    refX1 = statXPlatform
    refY1 = statYPlatform
    refZ1 = statZPlatform
    refX2 = statXPlatform
    refY2 = statYPlatform
    refZ2 = statZPlatform
    refX3 = statXPlatform
    refY3 = statYPlatform
    refZ3 = statZPlatform
else
    refX1 = statX1
    refY1 = statY1
    refZ1 = statZ1
    refX2 = statX2
    refY2 = statY2
    refZ2 = statZ2
    refX3 = statX3
    refY3 = statY3
    refZ3 = statZ3
end


# Dynamic motions (will be applied in the dynamic analysis)
nSamples = length(df[:,"time"]);
ratioSampleTaper = 0.02
taper = floor(Int,ratioSampleTaper*nSamples)
ramp = vcat(LinRange(0.,1.,taper),ones(nSamples-2*taper),LinRange(1.,0.,taper))

xMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead1 [m]"].- refX1).*ramp,0.))
yMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead1 [m]"].- refY1).*ramp,0.))
zMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead1 [m]"].- refZ1).*ramp,0.))
xMotion2 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead2 [m]"].- refX2).*ramp,0.))
yMotion2 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead2 [m]"].- refY2).*ramp,0.))
zMotion2 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead2 [m]"].- refZ2).*ramp,0.))
xMotion3 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead3 [m]"].- refX3).*ramp,0.))
yMotion3 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead3 [m]"].- refY3).*ramp,0.))
zMotion3 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead3 [m]"].- refZ3).*ramp,0.))
xMotionPlatform = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Surge [m]"].- statXPlatform).*ramp,0.))
yMotionPlatform = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Sway [m]"].- statYPlatform).*ramp,0.))
zMotionPlatform = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Heave [m]"].- statZPlatform).*ramp,0.))

# Line cross-section description
#------------------------------------------
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
nel         = [5      ,    23     , 12       ,        7     ] # Number of elements per segment
segLength   = [150    ,    414.   , 250.     ,        150.  ] # Segment lengths
xSection    = [x1_mat ,    x2_mat , x2_mat   ,        x1_mat]; # Cross-section type
nseg    = length(nel)
elLength = segLength./nel


# Line constructors
#------------------------------------------
azimuth = [
    (0. - yaw0) * π/180.,
    (120. - yaw0) * π/180.,
    (-120. - yaw0) * π/180.,
]
topCoord = [
    [initX1,initY1,initZ1],
    [initX2,initY2,initZ2],
    [initX3,initY3,initZ3],
]
nlines = length(azimuth); 
if bSameNode
    prescribedTopMotion = [
        [xMotionPlatform, yMotionPlatform, zMotionPlatform],
        [xMotionPlatform, yMotionPlatform, zMotionPlatform],
        [xMotionPlatform, yMotionPlatform, zMotionPlatform],
    ]
else
    prescribedTopMotion = [
        [xMotion1, yMotion1, zMotion1],
        [xMotion2, yMotion2, zMotion2],
        [xMotion3, yMotion3, zMotion3],
    ]
end
# TODO what if incomplete information on the stat position of the fairlead => Adof
topStatCoord = [
    [refX1,refY1,refZ1],
    [refX2,refY2,refZ2],
    [refX3,refY3,refZ3],
]
    
# Create node positions along the line for easy access
cum_len = cumsum([0; segLength[1:end-1]])  # Cumulative length up to start of each segment
X_segments = [cum_len[seg] .+ (0:elLength[seg]:segLength[seg]) for seg in 1:nseg]
X_flat = vcat(X_segments...)  # Flattened x-positions for all nodes

# Function to get global node index from segment and local node index (1-based)
function get_global_node_idx(seg, local_idx)
    if seg < 1 || seg > nseg || local_idx < 1 || local_idx > nel[seg] + 1
        error("Invalid segment ($seg) or local node index ($local_idx)")
    end
    start_idx = sum(nel[i] for i in 1:seg-1; init=0)
    return start_idx + local_idx
end

# Common line characteristics
#------------------------------------------
radialOffsetAnchor = 1000. # in case of a topnode being the platform center, we substract the fairlead radial offset to the static position of anchor 
lineLength = sum(segLength)
offsetHorizontal = bSameNode ? radialOffsetAnchor - lineLength - initX1 : radialOffsetAnchor - lineLength # adding also the fairlead offset in case of shared node
offsetDownwards = bSameNode ? -waterDepth - initZ1 : -waterDepth  # adding also the fairlead offset in case of shared node
prestrechStaticAnalysis = lineLength*0.01;


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
## Model construction
##########################################

topNode = Vector{Muscade.NodID}(undef,nlines)
nodeList  =   Vector{Vector{Vector{Muscade.NodID}}}(undef,nlines*nseg)
elementList = Vector{Vector{Muscade.EleID}}(undef,nlines)
aNode = Vector{Muscade.NodID}(undef,nlines)

# Create Muscade model
model       = Model(:optiflex22MW);

# Line construction
#------------------------------------------
for ind in 1:3
    local_azimuth = azimuth[ind]
    # Line
    if bSameNode
        if ind == 1 
            topNode[ind] = addnode!(model, [0., 0., 0.])
        else
            topNode[ind] = topNode[1]
        end
    else
        coordTopNode = topCoord[ind]
        topNode[ind] = addnode!(model,coordTopNode)
    end
    nodeList[ind],elementList[ind],aNode[ind] = MeshLineGauge(model, topNode[ind], local_azimuth, Bar3D, StrainGaugeOnBar3D, xSection,segLength,nel)

    # Define the prescribed end displacements of the lower extremity
    @functor with(offsetHorizontal,local_azimuth,prestrechStaticAnalysis)   xMotionBottom(x,t)= x[1] - cos(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis ))
    @functor with(offsetHorizontal,local_azimuth,prestrechStaticAnalysis)   yMotionBottom(x,t)= x[1] - sin(local_azimuth)*  (prestrechStaticAnalysis +  (min(t,-5.)+10)/5*( offsetHorizontal  - prestrechStaticAnalysis ))
    @functor with(offsetDownwards)                                          zMotionBottom(x,t)= x[1] -                      (                           (min(t,-5.)+10)/5*( offsetDownwards                             ))
    addelement!(model,DofConstraint,[nodeList[ind][nseg][end]],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionBottom, mode=equal)
    addelement!(model,DofConstraint,[nodeList[ind][nseg][end]],xinod=(1,),xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=yMotionBottom, mode=equal)
    addelement!(model,DofConstraint,[nodeList[ind][nseg][end]],xinod=(1,),xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionBottom, mode=equal);
    # add buoy
    addelement!(model,DofLoad,[nodeList[ind][nodnum_buoygrav_topchain][1]];field=:t3,value=buoygravForce_topchain);  
    addelement!(model,DofLoad,[nodeList[ind][nodnum_buoygrav_midpolyester][1]];field=:t3,value=buoygravForce_midpolyester);  
    addelement!(model,DofLoad,[nodeList[ind][nodnum_buoygrav_bottomchain][1]];field=:t3,value=buoygravForce_bottomchain);  
    # add soil contact
    [addelement!(model,SoilContact,[nodeList[ind][segnumsoil][idxNod]],z₀=-waterDepth,Kh=0.0,Kv=Kv,Ch=0.,Cv=0.0)  for idxNod = 1:length(nodeList[ind][segnumsoil])]
    # Define the prescribed end displacements of the top extremity
    xMotion,yMotion,zMotion = prescribedTopMotion[ind]
    refX,refY,refZ          = topStatCoord[ind]
    @functor with(xMotion,refX) xMotionTop(x,t)= x[1] - (exp(minimum([t,0]))*(refX)    + xMotion(t))
    @functor with(yMotion,refY) yMotionTop(x,t)= x[1] - (exp(minimum([t,0]))*(refY)    + yMotion(t))
    @functor with(zMotion,refZ) zMotionTop(x,t)= x[1] - (exp(minimum([t,0]))*(refZ)    + zMotion(t))
    addelement!(model,DofConstraint,[topNode[ind]],xinod=(1,),xfield=(:t1,), λinod=1, λclass=:X, λfield=:λt1, gap=xMotionTop, mode=equal)
    addelement!(model,DofConstraint,[topNode[ind]],xinod=(1,),xfield=(:t2,), λinod=1, λclass=:X, λfield=:λt2, gap=yMotionTop, mode=equal)
    addelement!(model,DofConstraint,[topNode[ind]],xinod=(1,),xfield=(:t3,), λinod=1, λclass=:X, λfield=:λt3, gap=zMotionTop, mode=equal);

    # Costs
    #------------------------------------------
    @functor with() costA(a)             = 1.e1 *    (a-0.)^2
    @functor with() costAother(a)        = 1.e1 *    (a-0.)^2
    
    # Define the prescribed end displacements of the top extremity
    if  !bSameNode || (ind == 1) 
        eAγ₀             = [addelement!(model,SingleAcost,[aNode[ind]]; field=:γ₀,cost=costA)];
        eAγ₁             = [addelement!(model,SingleAcost,[aNode[ind]]; field=:γ₁,cost=costAother)];
    end
end

# Static analysis
#------------------------------------------
initialstate = initialize!(model);    
staticStates = solve(SweepX{0};initialstate,time=staticLoadSteps,verbose=false,maxΔx=1e-6,maxiter=60);

if bPlotStaticAnalysis
    fig      = Figure(size = (1000,1000))
    ax = Axis3(fig[1,1])
    draw!(ax,initialstate)
    display(fig)
    # Plot the static analysis sequence
    for stateIdx ∈ 1:nStaticLoadSteps
        draw!(ax,staticStates[stateIdx])
    end
    save("figs/"*runName*"_static.png",fig)
    display(fig)
end

##########################################
## Forward Analysis
##########################################

if bForwardAnalysis
    # Dynamic analysis
    #------------------------------------------ 
    # Run the dynamic analysis
    dynamicStates          = solve(ForwardSolver;
            initialstate=staticStates[nStaticLoadSteps],
            time=dynamicLoadSteps,
            verbose=true,
            β=1/3.,γ=0.605, # Unconditionally stable for : γ >= .5 &&  2 β >= γ
            # β=1/2.5,γ=0.705, # Unconditionally stable for : γ >= .5 &&  2 β >= γ
            # β=1/3.9,γ=0.505,
            maxiter=maxiterdyn,
            maxΔx = 1e-5);
            
    # Post-process
    #------------------------------------------ 
    # Retrieve axial force at top location
    # TODO Extract the tension for the first element in the StrainGauge element, and not the one in the second element (Bar3D)due to not easy use of request
    req = @request gp(resultants(fᵢ))
    out = getresult(dynamicStates,req,[elementList[1][2]])
    Fgp1_ = [ out[idxEl].gp[1][:resultants][:fᵢ] for idxEl ∈ 1:size(out,2)];
    out = getresult(dynamicStates,req,[elementList[2][2]])
    Fgp2_ = [ out[idxEl].gp[1][:resultants][:fᵢ] for idxEl ∈ 1:size(out,2)];
    out = getresult(dynamicStates,req,[elementList[3][2]])
    Fgp3_ = [ out[idxEl].gp[1][:resultants][:fᵢ] for idxEl ∈ 1:size(out,2)];


    if bForwardAnimation
        # Produce an animation
        fig2   = Figure(size = (2000,1000))
        ax2 = Axis3(fig2[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3),title=file_path)
        xlims!(ax2,-1000,1000); ylims!(ax2,-1000,1000); zlims!(ax2,-waterDepth - 20,10)
        graphic = draw!(ax2,dynamicStates[1])
        ax2.azimuth[]=-π/2+π/180*10;
        ax2.elevation[]=0+π/180*10;
        framerate = 20
        # loadStepsIterator = 1:3:nDynamicLoadSteps
        loadStepsIterator = 1:3:checkResultsUntilStep
        record(fig2, "figs/"*runName*".mp4", loadStepsIterator;
                framerate = framerate) do stateIdx
                draw!(graphic,dynamicStates[stateIdx])
        end
    end


    if bPlotForwardAnalysis
        # Plot comparison between Muscade and RIFLEX results. 
        fig3      = Figure(size = (1000,1000))
        
        ax1 = Axis(fig3[1, 1],ylabel="Top x. disp. [m]", title = "3 = Port side, 2 = Starboard side, 1 = Stern")
        lines!(ax1,dynamicLoadSteps,xMotion1(dynamicLoadSteps),         color = :red,      linestyle = :solid,   label = "Prescribed 1")
        lines!(ax1,dynamicLoadSteps,xMotion2(dynamicLoadSteps),         color = :blue,     linestyle = :solid,   label = "Prescribed 2")
        lines!(ax1,dynamicLoadSteps,xMotion3(dynamicLoadSteps),         color = :green,    linestyle = :solid,   label = "Prescribed 3")
        vlines!(df[:,"time"][taper]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
        axislegend()
        
        ax2 = Axis(fig3[2, 1],ylabel="Top y. disp. [m]")
        lines!(ax2,dynamicLoadSteps,yMotion1(dynamicLoadSteps),         color = :red,      linestyle = :solid,   label = "Prescribed 1")
        lines!(ax2,dynamicLoadSteps,yMotion2(dynamicLoadSteps),         color = :blue,     linestyle = :solid,   label = "Prescribed 2")
        lines!(ax2,dynamicLoadSteps,yMotion3(dynamicLoadSteps),         color = :green,    linestyle = :solid,   label = "Prescribed 3")
        vlines!(df[:,"time"][taper]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
        axislegend()
        
        ax3 = Axis(fig3[3, 1],ylabel="Top vert. disp. [m]")
        lines!(ax3,dynamicLoadSteps,zMotion1(dynamicLoadSteps),         color = :red,      linestyle = :solid,   label = "Prescribed 1")
        lines!(ax3,dynamicLoadSteps,zMotion2(dynamicLoadSteps),         color = :blue,     linestyle = :solid,   label = "Prescribed 2")
        lines!(ax3,dynamicLoadSteps,zMotion3(dynamicLoadSteps),         color = :green,    linestyle = :solid,   label = "Prescribed 3")
        vlines!(df[:,"time"][taper]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
        axislegend()
        
        ax4 = Axis(fig3[4, 1],ylabel="Axial force [kN]")
        lines!(ax4, dynamicLoadSteps, Fgp1_/1e3,                     color = :red,   linestyle = :solid ,   label="Muscade1")
        lines!(ax4, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3, color = :red,   linestyle = :dot   , label="SIMA1")
        lines!(ax4, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3, color = :orange,   linestyle = :dot   , label="SIMA1_wavesHydroForces")
        axislegend()
        
        ax5 = Axis(fig3[5:6, 1],ylabel="Axial force [kN]", xlabel="Time [s]")
        lines!(ax5, dynamicLoadSteps, Fgp2_/1e3,                                color = :blue,             linestyle = :solid ,    label="Muscade2")
        lines!(ax5, df[:,"time"], df[:,"TensionTopChainL2 [N]"]/1e3,            color = :blue,             linestyle = :dot   ,  label="SIMA2")
        lines!(ax5, df_w[:,"time"], df_w[:,"TensionTopChainL2 [N]"]/1e3,        color = :purple,         linestyle = :dot   ,  label="SIMA2_wavesHydroForces")
        lines!(ax5, dynamicLoadSteps, Fgp3_/1e3,                                color = :green,            linestyle = :solid ,     label="Muscade3")
        lines!(ax5, df[:,"time"], df[:,"TensionTopChainL3 [N]"]/1e3,            color = :green,            linestyle = :dot   ,   label="SIMA3")
        lines!(ax5, df_w[:,"time"], df_w[:,"TensionTopChainL3 [N]"]/1e3,        color = :grey,        linestyle = :dot   ,   label="SIMA3_wavesHydroForces")
        axislegend()
        
        [xlims!(idxAx,0,dynamicLoadSteps[end]) for idxAx∈[ax1,ax2,ax3,ax4,ax5]]
        save("figs/"*runName*"_dynamic.png",fig3)
        display(fig3)
        
        # Plot comparison between Muscade and RIFLEX results. 
        fig4      = Figure(size = (1000,500))
        
        ax2 = Axis(fig4[1,1], ylabel="Axial force [kN]", xlabel="Time [s]", ylabelsize=20, xlabelsize=20)
        lines!(ax2, dynamicLoadSteps, Fgp2_/1e3,                                color = :blue,             linestyle = :solid ,    label="Muscade2")
        lines!(ax2, df[:,"time"], df[:,"TensionTopChainL2 [N]"]/1e3,            color = :blue,             linestyle = :dot   ,  label="SIMA2")
        lines!(ax2, df_w[:,"time"], df_w[:,"TensionTopChainL2 [N]"]/1e3,        color = :purple,         linestyle = :dot   ,  label="SIMA2_wavesHydroForces")
        lines!(ax2, dynamicLoadSteps, Fgp3_/1e3,                                color = :green,            linestyle = :solid ,     label="Muscade3")
        lines!(ax2, df[:,"time"], df[:,"TensionTopChainL3 [N]"]/1e3,            color = :green,            linestyle = :dot   ,   label="SIMA3")
        lines!(ax2, df_w[:,"time"], df_w[:,"TensionTopChainL3 [N]"]/1e3,        color = :grey,        linestyle = :dot   ,   label="SIMA3_wavesHydroForces")
        axislegend(position=:rb)
        
        Fgp3_interp = linear_interpolation(
        df[:,"time"],
        df[:,"TensionTopChainL3 [N]"],
        )
        [xlims!(idxAx,500,600) for idxAx∈[ax2]]
        [ylims!(idxAx,minimum(Fgp3_interp(500:1:600)/1e3)-100,maximum(Fgp3_interp(500:1:600)/1e3)+200) for idxAx∈[ax2]]
        save("figs/"*runName*"_dynamic_zoom.png",fig4)
        display(fig4)
    end

    # Getting results from forward analysis
    #------------------------------------------
    
    x_dir = Vector{Matrix{Float64}}(undef, nlines)
    y_dir = Vector{Matrix{Float64}}(undef, nlines)
    z_dir = Vector{Matrix{Float64}}(undef, nlines)

    for il in 1:nlines
        # Collect all node IDs in order for line 1
        all_nodes = vcat([nodeList[il][seg][:] for seg in 1:nseg]...)
        
        # Get displacements for all nodes at each load step
        xs = [getdof(dynamicStates[idxLoad]; field=:t1, nodID=all_nodes) for idxLoad ∈ 1:nDynamicLoadSteps]
        ys = [getdof(dynamicStates[idxLoad]; field=:t2, nodID=all_nodes) for idxLoad ∈ 1:nDynamicLoadSteps]
        zs = [getdof(dynamicStates[idxLoad]; field=:t3, nodID=all_nodes) for idxLoad ∈ 1:nDynamicLoadSteps]
        
        # Convert to matrices for easier manipulation (rows: time steps, columns: nodes)
        x_dir[il] = permutedims(hcat(xs...))  # Shape: (nDynamicLoadSteps, nnodes)
        y_dir[il] = permutedims(hcat(ys...))
        z_dir[il] = permutedims(hcat(zs...))
    end

    println("============= SUCCESS EXTRACTING RESULTS =============")
    # println("x_matrix_L1 size: ", size(x_matrix_L1))
    # println("Total nodes: ", length(X_flat))
    # println("X positions: ", X_flat[1:10])  # First 10 node positions
    # # Example: access history of node 3 in segment 2
    # global_idx = get_global_node_idx(2, 3)
    # println("Global index for segment 2, node 3: ", global_idx)
    # println("X history for that node (first 5 steps): ", x_matrix_L1[1:5, global_idx])

    # Saving results
    #------------------------------------------
    if bSaveForward
        # TODO Implement the saving

    end
end

##########################################
## Inverse Analysis
##########################################

if bInverseAnalysis

    inverseStates = solve(InverseSolver;
        initialstate=[staticStates[nStaticLoadSteps]],
        time=[inverseLoadSteps],
        verbose=true,
        maxiter= maxiterinv ,
        maxΔx  = maxΔx   ,
        maxΔλ  = maxΔλ   ,
        maxΔu  = maxΔu   ,
        maxΔa= maxΔa
    );

    println(" NO MORE ")

end