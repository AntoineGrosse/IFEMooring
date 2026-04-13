using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra, CSV, DataFrames
include("BiasedStrainGauge.jl")
include("MeshLineGauge.jl")
include("ChainLink.jl")
currentDir = @__DIR__
cd(currentDir)

# Constants
g = 9.81
ρ = 1025.

# Cost factors
# Strain
@show σ = 1e-2
# A cost
@show Ca0 = 1e-3
@show Ca1 = 1e3
# U cost
@show Cu0 = 1e-3
@show Cu1 = 1e3
# X cost
@show Cx1 = 1e8
@show Cc1 = 1e6

# Parameters
attenuationFactors = [0.4, 0.7, 1.]
static_bias = 0.0015 # Static bias to retrieve
Δtᵢₙᵥ = 1.
nsteps = 2e2
staticLoadSteps = (-10:0.1:0)
inverseLoadSteps = (0:Δtᵢₙᵥ:(nsteps)*Δtᵢₙᵥ) .+ eps()
InverseSolver = DirectXUA{2,0,1}

# Scaling    @show 
scale = (
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
        γ₀=1, 
        γ₁=1
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
@show boolFromRealDisp = true

@show boolIntegrateBuoys = true
@show boolIntegrateSoil = true

@show boolUdofOnTop = true
@show boolUdofOnAnchor = true
@show boolUdofOnLine = false

@show boolXcostOnTop = true
@show boolXcostOnAnchor = true

@show boolXconstrOnTop = false
@show boolXconstrOnAnchor = false

@show boolStrainCost = true


# Loss functions
δ = 1e-2 # For Huber loss
quadra(x) = x⋅x # quadratic loss
expo(x) = 1 - exp(- x⋅x) # exponetial loss
cauchy(x) = log(1 + x⋅x) # Cauchy loss
huber(x) = VALUE(∂0(x⋅x)) < δ ? 0.5 * x⋅x : δ * (sqrt(x⋅x) - 0.5*δ) # Huber loss
pseudo_huber(x) = δ^2 * (sqrt(1 + x⋅x/(δ^2)) - 1) # Pseudo huber loss
scaled_quadra(x) = sqrt(1 + x⋅x) # Custom
ch(x) = cosh(x) # Cosh
logch(x) = log(ch(x)) # Logcosh
loss_function = quadra


# Iterative continuation loop
#------------------------------------------
for (iterContinuation, attenuationFactor) in enumerate(attenuationFactors)

    println("\n======= Continuation iteration number $(iterContinuation)/$(length(attenuationFactors)) with factor $(attenuationFactor) =======")

    # Prescribed displacements
    #------------------------------------------
    if boolFromRealDisp
        file_path = "input\\test_wo.csv"
        df = CSV.read(file_path, DataFrame; delim=',')
        statX1 = df[:,"Xfairlead1 [m]"][1]
        statY1 = df[:,"Yfairlead1 [m]"][1]
        statZ1 = df[:,"Zfairlead1 [m]"][1]
        nSamples = length(df[:,"time"]);
        ratioSampleTaper = 0.02
        taper = floor(Int,ratioSampleTaper*nSamples)
        ramp = vcat(LinRange(0.,1.,taper),ones(nSamples-2*taper),LinRange(1.,0.,taper))
        xMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead1 [m]"].- statX1).*ramp .* attenuationFactor,0.))
        yMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead1 [m]"].- statY1).*ramp .* attenuationFactor,0.))
        zMotion1 = linear_interpolation(vcat(df[1,"time"]-10.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead1 [m]"].- statZ1).*ramp .* attenuationFactor,0.))
        x_disp_interp = xMotion1 
        y_disp_interp = yMotion1 
        z_disp_interp = zMotion1 
    else
        max_disp = -10.
        # x_disp = max_disp * sin.(4π * inverseLoadSteps ./ inverseLoadSteps[end])
        x_disp = max_disp * inverseLoadSteps ./ inverseLoadSteps[end]
        x_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., x_disp .* attenuationFactor))
        # y_disp = max_disp * sin.(4π * inverseLoadSteps ./ inverseLoadSteps[end])
        y_disp = 0 .* inverseLoadSteps
        y_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., y_disp .* attenuationFactor))
        # z_disp = max_disp * sin.(4π * inverseLoadSteps ./ inverseLoadSteps[end])
        z_disp = 0 .* inverseLoadSteps
        z_disp_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., z_disp .* attenuationFactor))
    end

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

    ##########################################
    ## Forward
    ##########################################

    model = Model(:testline)
    topNode = addnode!(model, [0., 0., -fairleadDepth])
    nodeList, elementList, anodeList = MeshLineGauge(model, topNode, 0., ChainLink, BiasedStrainGauge, xSection, segLength, nel)

    # X Constraints : Anchor
    @functor with(offsetHorizontal, prestrechStaticAnalysis)    xMotionBottom(x,t) = Cc1 * (x[1] - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis)))
    @functor with()                                             yMotionBottom(x,t) = Cc1 * (x[1])
    @functor with(offsetDownwards)                              zMotionBottom(x,t) = Cc1 * (x[1] - ((min(t,-5.)+10)/5 * offsetDownwards))
    addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λat1, gap=xMotionBottom, mode=equal)
    addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λat2, gap=yMotionBottom, mode=equal)
    addelement!(model, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λat3, gap=zMotionBottom, mode=equal)

    # X Constraints : Top
    @functor with() xMotionTop(x,t) = Cc1 * (x[1] - x_disp_interp(t))
    @functor with() yMotionTop(x,t) = Cc1 * (x[1] - y_disp_interp(t))
    @functor with() zMotionTop(x,t) = Cc1 * (x[1] - z_disp_interp(t))
    addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λpt1, gap=xMotionTop, mode=equal)
    addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λpt2, gap=yMotionTop, mode=equal)
    addelement!(model, DofConstraint, [topNode], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λpt3, gap=zMotionTop, mode=equal)

    # Buoys and Clampweights
    if boolIntegrateBuoys
        nodnum_buoygrav_topchain = 2
        nodnum_buoygrav_midpolyester = nseg - 1
        nodnum_buoygrav_bottomchain = nseg
        @functor with() buoygravForce_topchain(t)       = ((min(t,-5.)+10)/5) * (-3 + 0. ) * 1e3 * g
        @functor with() buoygravForce_midpolyester(t)   = ((min(t,-5.)+10)/5) * (-3 + 10.) * 1e3 * g
        @functor with() buoygravForce_bottomchain(t)    = ((min(t,-5.)+10)/5) * (-3 + 15.) * 1e3 * g
        addelement!(model, DofLoad, [nodeList[nodnum_buoygrav_topchain][1]], field=:t3, value=buoygravForce_topchain)
        addelement!(model, DofLoad, [nodeList[nodnum_buoygrav_midpolyester][1]], field=:t3, value=buoygravForce_midpolyester)
        addelement!(model, DofLoad, [nodeList[nodnum_buoygrav_bottomchain][1]], field=:t3, value=buoygravForce_bottomchain)
    end

    # Soil contact
    if boolIntegrateSoil
        local KvExponent = log(14/14000)/log(attenuationFactors[1])
        local Kv = attenuationFactor^KvExponent * 14000.
        # local Kv = 14000.
        segnumsoil = nseg
        [addelement!(model, SoilContact, [nodeList[segnumsoil][idxNod]], z₀=offsetDownwards, Kh=0.0, Kv=Kv, Ch=0., Cv=0.0) for idxNod in 1:length(nodeList[segnumsoil])]
    end

    # Forward solve
    initialstate = initialize!(model)
    staticStates = solve(SweepX{0}; initialstate, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)
    stateForward = solve(SweepX{2}; initialstate=staticStates[end], time=inverseLoadSteps, maxΔx=1e-5, verbose=false, maxiter=100)

    # Making artificial measurement data from forward solving
    req = @request εₐₓ
    out = getresult(stateForward, req, [elementList[1]])
    strain = [out[idxEl].εₐₓ for idxEl in axes(out,2)]
    measured_strain_interp = linear_interpolation(vcat(-10., inverseLoadSteps), vcat(0., strain .+ static_bias))

    # Forward animation
    fig_anim   = Figure(size = (2000,1000))
    ax_for = Axis3(fig_anim[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3),title="Animation Forward analysis")
    xlims!(ax_for,-1000,1000); ylims!(ax_for,-1000,1000); zlims!(ax_for,-waterDepth - 20,10)
    graphic = draw!(ax_for,stateForward[1])
    ax_for.azimuth[]=-π/2+π/180*10;
    ax_for.elevation[]=0+π/180*10;
    framerate = 20
    loadStepsIterator = 1:3:length(inverseLoadSteps)
    record(fig_anim, "figs/animationForward.mp4", loadStepsIterator;
    framerate = framerate) do stateIdx
        draw!(graphic,stateForward[stateIdx])
    end

    ##########################################
    ## Inverse
    ##########################################
    # Penalties - adjusted for larger strains
    @functor with() costA(a) = loss_function(sqrt(Ca0) * (a - 0.))
    @functor with() costAother(a) = loss_function(sqrt(Ca1) * (a - 0.))
    @functor with() costUx(u,t) = loss_function(sqrt(Cu0) * (u - (-1e6)))
    @functor with() costUy(u,t) = loss_function(sqrt(Cu0) * (u - 0.))
    @functor with() costUz(u,t) = loss_function(sqrt(Cu0) * (u - 1e6))
    @functor with() costUother(u,t) = loss_function(sqrt(Cu1) * (u - 0.))

    vec3(v,ind) = SVector{3}(v[i] for i∈ind)

    model_inv = Model(:testline)
    topNode_inv = addnode!(model_inv, [0., 0., -fairleadDepth])
    nodeList_inv, elementList_inv, anodeList_inv = MeshLineGauge(model_inv, topNode_inv, 0., ChainLink, BiasedStrainGauge, xSection, segLength, nel)

    # Constraints (same as forward)
    # X Constraints : Anchor
    anchorConstr = boolXconstrOnAnchor ? :equal : :off
    @functor with() modeconstraintAnchor(t) = t < eps() ? :equal : anchorConstr
    addelement!(model_inv, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λat1, gap=xMotionBottom, mode=modeconstraintAnchor)
    addelement!(model_inv, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λat2, gap=yMotionBottom, mode=modeconstraintAnchor)
    addelement!(model_inv, DofConstraint, [nodeList[nseg][end]], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λat3, gap=zMotionBottom, mode=modeconstraintAnchor)
    # X Constraints : Top
    topConstr = boolXconstrOnTop ? :equal : :off
    @functor with() modeconstraintTop(t) = t < eps() ? :equal : topConstr
    addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t1,), λinod=1, λclass=:X, λfield=:λpt1, gap=xMotionTop, mode=modeconstraintTop)
    addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t2,), λinod=1, λclass=:X, λfield=:λpt2, gap=yMotionTop, mode=modeconstraintTop)
    addelement!(model_inv, DofConstraint, [topNode_inv], xinod=(1,), xfield=(:t3,), λinod=1, λclass=:X, λfield=:λpt3, gap=zMotionTop, mode=modeconstraintTop)

    # Buoys and Clampweights
    if boolIntegrateBuoys
        addelement!(model_inv, DofLoad, [nodeList_inv[nodnum_buoygrav_topchain][1]], field=:t3, value=buoygravForce_topchain)
        addelement!(model_inv, DofLoad, [nodeList_inv[nodnum_buoygrav_midpolyester][1]], field=:t3, value=buoygravForce_midpolyester)
        addelement!(model_inv, DofLoad, [nodeList_inv[nodnum_buoygrav_bottomchain][1]], field=:t3, value=buoygravForce_bottomchain)
    end

    # Soil
    if boolIntegrateSoil
        [addelement!(model_inv, SoilContact, [nodeList_inv[segnumsoil][idxNod]], z₀=offsetDownwards, Kh=0.0, Kv=Kv, Ch=0., Cv=0.0) for idxNod in 1:length(nodeList_inv[segnumsoil])]
    end

    # Strain cost
    element1 = elementList_inv[1]
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

    function total_cost(state)
        total = 0.0
        for s in state[1]
            X = (SVector{6}(s.X[1][j] for j in 1:6),)
            total += VALUE(straincost(X, (), s.A, s.time))
            total += VALUE(costA(s.A[1])) + VALUE(costAother(s.A[2]))
        end
        return total
    end

    # A costs
    eAγ₀ = addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₀, cost=costA)
    eAγ₁ = addelement!(model_inv, SingleAcost, [anodeList_inv]; field=:γ₁, cost=costAother)
    # U costs
    if boolUdofOnTop
        eUtop1 = addelement!(model_inv, SingleUdof, [topNode_inv]; Xfield=:t1,Ufield=:upt1,cost=costUx)
        eUtop2 = addelement!(model_inv, SingleUdof, [topNode_inv]; Xfield=:t2,Ufield=:upt2,cost=costUy)
        eUtop3 = addelement!(model_inv, SingleUdof, [topNode_inv]; Xfield=:t3,Ufield=:upt3,cost=costUz)
    end
    if boolUdofOnAnchor
        eUanch1 = addelement!(model_inv, SingleUdof, [nodeList_inv[end][end]]; Xfield=:t1,Ufield=:uat1,cost=costUx)
        eUanch2 = addelement!(model_inv, SingleUdof, [nodeList_inv[end][end]]; Xfield=:t2,Ufield=:uat2,cost=costUy)
        eUanch3 = addelement!(model_inv, SingleUdof, [nodeList_inv[end][end]]; Xfield=:t3,Ufield=:uat3,cost=costUz)
    end
    if boolUdofOnLine
        eUt1 = [addelement!(model_inv, SingleUdof, [nodeList_inv[iseg][inod]]; Xfield=:t1,Ufield=:ut1,cost=costUother) for iseg in 1:nseg-1 for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
        eUt2 = [addelement!(model_inv, SingleUdof, [nodeList_inv[iseg][inod]]; Xfield=:t2,Ufield=:ut2,cost=costUother) for iseg in 1:nseg-1 for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
        eUt3 = [addelement!(model_inv, SingleUdof, [nodeList_inv[iseg][inod]]; Xfield=:t3,Ufield=:ut3,cost=costUother) for iseg in 1:nseg-1 for inod in 1:nel[iseg] if (iseg,inod) ∉ [(1,1),(4,nel[4])]]
    end
    # Top motion (perfect) measurement X cost
    if boolXcostOnTop
        @functor with(x_disp_interp) xCostTop(x,t) = quadra(sqrt(Cx1) * (x - x_disp_interp(t)))
        @functor with(y_disp_interp) yCostTop(x,t) = quadra(sqrt(Cx1) * (x - y_disp_interp(t)))
        @functor with(z_disp_interp) zCostTop(x,t) = quadra(sqrt(Cx1) * (x - z_disp_interp(t)))
        eXtopt1 = addelement!(model_inv, SingleDofCost, [nodeList_inv[1][1]]; class=:X,field=:t1,cost=xCostTop)
        eXtopt2 = addelement!(model_inv, SingleDofCost, [nodeList_inv[1][1]]; class=:X,field=:t2,cost=yCostTop)
        eXtopt3 = addelement!(model_inv, SingleDofCost, [nodeList_inv[1][1]]; class=:X,field=:t3,cost=zCostTop)
    end
    # Anchor motion (perfect) measurement X cost
    if boolXcostOnAnchor
        @functor with(prestrechStaticAnalysis,offsetHorizontal) xCostAnch(x,t) = quadra(sqrt(Cx1) * (x - (prestrechStaticAnalysis + (min(t,-5.)+10)/5 * (offsetHorizontal - prestrechStaticAnalysis))))
        @functor with()                                         yCostAnch(x,t) = quadra(sqrt(Cx1) * (x - 0.))
        @functor with(offsetDownwards)                          zCostAnch(x,t) = quadra(sqrt(Cx1) * (x - ((min(t,-5.)+10)/5 * offsetDownwards)))
        eXancht1 = addelement!(model_inv, SingleDofCost, [nodeList_inv[end][end]]; class=:X,field=:t1,cost=xCostAnch)
        eXancht2 = addelement!(model_inv, SingleDofCost, [nodeList_inv[end][end]]; class=:X,field=:t2,cost=yCostAnch)
        eXancht3 = addelement!(model_inv, SingleDofCost, [nodeList_inv[end][end]]; class=:X,field=:t3,cost=zCostAnch)
    end
    # Strain measurement cost
    if boolStrainCost
        edofcost = addelement!(model_inv, DofCost, [nodeList_inv[1][1], nodeList_inv[1][2], anodeList_inv];
            xinod=(1,1,1,2,2,2), xfield=(:t1,:t2,:t3,:t1,:t2,:t3),
            ainod=(3,3), afield=(:γ₀,:γ₁),
            cost=straincost)
    end

    # Scaling
    # Use the same order-of-magnitude separation suggested by study_scale,
    # but keep the model two-parameter (γ₀ and γ₁).
    setscale!(model_inv; scale = scale, Λscale = Λscale)


    initialstate_inv = initialize!(model_inv)
    if iterContinuation == 1
        # Solve static
        local staticStates_inv = solve(SweepX{0}; initialstate=initialstate_inv, time=staticLoadSteps, verbose=false, maxΔx=1e-6, maxiter=60)

        # Solve X
        local stateX = solve(DirectXUA{0,0,0};
            initialstate=[staticStates_inv[end]],
            time=[eps():eps():eps()*2],
            verbose=true,
            maxiter=100,
            maxΔx=1e-5,   # More relaxed convergence (was 1e-3, keep for now)
            maxΔu=Inf,
            maxΔa=1e-5,   # REDUCED from 1e-3 to focus on A convergence
            maxΔλ=Inf,
            saveiter=true,
        )

        local laststepX = findlastassigned(stateX)
        local intermediateState = stateX[laststepX][1][end]
        local initialtrajectory = nothing

        # Solve XU
        local stateXU = solve(DirectXUA{2,0,0};
            initialstate=[intermediateState],
            initialtrajectory = initialtrajectory,
            time=[inverseLoadSteps],
            verbose=true,
            maxiter=20,
            maxΔx=1e-3,   # More relaxed convergence (was 1e-3, keep for now)
            maxΔu=Inf,
            maxΔa=1e-5,   # REDUCED from 1e-3 to focus on A convergence
            maxΔλ=Inf,
            saveiter=true,
        )
        local laststepXU = findlastassigned(stateXU)
        local intermediateState = stateXU[laststepXU][1][1]
        local initialtrajectory = [stateXU[laststepXU][1]]
        
    else
        local initialtrajectory = [stateXUA[laststep][1]]
        local intermediateState = stateXUA[laststep][1][1]
    end


    # Solve XUA
    global stateXUA = solve(InverseSolver;
        initialstate=[intermediateState],
        initialtrajectory = initialtrajectory,
        time=[inverseLoadSteps],
        verbose=true,
        maxiter=20,
        maxΔx=1e-3,   # More relaxed convergence (was 1e-3, keep for now)
        maxΔu=Inf,
        maxΔa=1e-5,   # REDUCED from 1e-3 to focus on A convergence
        maxΔλ=Inf,
        saveiter=true,
    )

    global laststep = findlastassigned(stateXUA)
    if laststep > 0
        global state = stateXUA[laststep][1]
        staticdev = VALUE(state[end].A[1])
        dynamicdev = VALUE(state[end].A[2])
        println("Converged: Static dev = $staticdev, True = $static_bias, Error = $(abs(staticdev - static_bias)/static_bias * 100)%")
        println("Dynamic dev = $dynamicdev")
    else
        println("Did not converge")
    end
    # Produce an animation
    fig_anim_inv   = Figure(size = (2000,1000))
    ax_inv = Axis3(fig_anim_inv[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3),title="Animation inverse reconstruction")
    xlims!(ax_inv,-1000,1000); ylims!(ax_inv,-1000,1000); zlims!(ax_inv,-waterDepth - 20,10)
    graphic = draw!(ax_inv,state[1])
    ax_inv.azimuth[]=-π/2+π/180*10;
    ax_inv.elevation[]=0+π/180*10;
    framerate = 20
    loadStepsIterator = 1:3:length(inverseLoadSteps)
    record(fig_anim_inv, "figs/animationInverse.mp4", loadStepsIterator;
            framerate = framerate) do stateIdx
            draw!(graphic,state[stateIdx])
    end
    
    # Produce an animation
    fig_rec_inv   = Figure(size = (2000,1000))
    ax_rec = Axis3(fig_rec_inv[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3),title="Animation inverse iteration")
    xlims!(ax_rec,-1000,1000); ylims!(ax_rec,-1000,1000); zlims!(ax_rec,-waterDepth - 20,10)
    graphic = draw!(ax_rec,stateXUA[1][1][end])
    ax_rec.azimuth[]=-π/2+π/180*10;
    ax_rec.elevation[]=0+π/180*10;
    framerate = 20
    loadStepsIterator = 1:1:laststep
    record(fig_rec_inv, "figs/reconstructionIterations.mp4", loadStepsIterator;
            framerate = framerate) do stateIdx
            draw!(graphic,stateXUA[stateIdx][1][end])
    end
end

# Muscade.study_scale(stateXUA[1][1][1]; SP = stateXUA[1][1][1].SP, verbose = true)