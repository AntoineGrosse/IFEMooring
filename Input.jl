# Cost factors
#==NONE==#

# Strain std
#==NONE==#

# Parameters
@show maxΔx = 1e-5
@show maxΔa = 1e-5
@show maxiterinv = 12

# Load steps
staticLoadSteps = (-20:0.01:0)
forwardTimesteps = (0:2:1000) .+ eps()
function ramp(t1,t2,slope,t)
    return max(0,t-t1)*slope - max(0,t-t2)*slope
end
staticFunction(t) = ramp(-10., 0., 1/10, t)
gravityFunction(t) = ramp(-20., -10., 1/10, t)

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
include("Material.jl")
@functor with() g̃_(t) = SVector(0.,0.,1. * gravityFunction(t));
x1_mat         = AxisymmetricChainLinkCrossSection(EA=x1_EA, μ=x1_μ, w=x1_w, g̃=g̃_, Caₜ=x1_Caₜ, Clₜ=x1_Clₜ, Cqₜ=x1_Cqₜ, Caₙ=x1_Caₙ, Clₙ=x1_Clₙ, Cqₙ=x1_Cqₙ)
x2_mat         = AxisymmetricChainLinkCrossSection(EA=x2_EA, μ=x2_μ, w=x2_w, g̃=g̃_, Caₜ=x2_Caₜ, Clₜ=x2_Clₜ, Cqₜ=x2_Cqₜ, Caₙ=x2_Caₙ, Clₙ=x2_Clₙ, Cqₙ=x2_Cqₙ)


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
prestrechStaticAnalysis = sum(segLength) * 0.001

# Initial position
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

# Azimuths
yaw0 = 0.
azimuths = [
    (0. - yaw0) * π/180.,
    (120. - yaw0) * π/180.,
    (240. - yaw0) * π/180.,
]



function read_input(file_path)

    df = CSV.read(file_path, DataFrame; delim=',')
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

    # Prescribed displacements
    #------------------------------------------
    xMotion1 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead1 [m]"].- firstX1).*ramp,0.))
    yMotion1 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead1 [m]"].- firstY1).*ramp,0.))
    zMotion1 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead1 [m]"].- firstZ1).*ramp,0.))
    xMotion2 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead2 [m]"].- firstX2).*ramp,0.))
    yMotion2 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead2 [m]"].- firstY2).*ramp,0.))
    zMotion2 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead2 [m]"].- firstZ2).*ramp,0.))
    xMotion3 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Xfairlead3 [m]"].- firstX3).*ramp,0.))
    yMotion3 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Yfairlead3 [m]"].- firstY3).*ramp,0.))
    zMotion3 = linear_interpolation(vcat(df[1,"time"]-20.,df[:,"time"],df[end,"time"]+10.),vcat(0.,(df[:,"Zfairlead3 [m]"].- firstZ3).*ramp,0.))

    res = [
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
    return res
end