"""
Visualization tool for inverse analysis iterates on cost surfaces.

Takes inverse solve results and plots the optimization path on interesting
parameter combinations (e.g., gamma0 vs X-DOFs, gamma0 vs gamma1, etc.)
"""

using Muscade, StaticArrays, GLMakie, Muscade.Toolbox, Interpolations, LinearAlgebra

include("BiasedStrainGaugeOnBarElement.jl")
include("MeshLineGauge.jl")

# ============================================================================
# Configuration
# ============================================================================

currentDir = @__DIR__
cd(currentDir)

# Constants
g = 9.81
ρ = 1025.
σ = 1e-2
Ca0 = 1e-3
Ca1 = 1e4
δ = 1e-1

# Loss functions
quadra(x)         = x⋅x
expo(x)           = 1 - exp(- x⋅x)
cauchy(x)         = log(1 + x⋅x)
huber(x)          = VALUE(∂0(x⋅x)) < δ ? 0.5 * x⋅x : δ * (sqrt(x⋅x) - 0.5*δ)
pseudo_huber(x)   = δ^2 * (sqrt(1 + x⋅x/(δ^2)) - 1)
scaled_quadra(x)  = sqrt(1 + x⋅x)
ch(x)             = cosh(x)
logch(x)          = log(ch(x))

loss_name = "quadra"
loss_function = quadra

# ============================================================================
# Utility Functions  
# ============================================================================

"""
    create_cost_surface(model, time, param_ranges::Dict, base_params::NamedTuple; 
                        param_indices::Dict, verbose=false)

Create a cost surface for two parameters.

# Arguments
- `model`: Muscade model with forward solve capability
- `time`: time step to evaluate
- `param_ranges`: Dict with parameter names → (start, stop, n_points)
- `base_params`: base values of all parameters
- `param_indices`: Dict mapping parameter names to their indices
- `verbose`: print progress
"""
function create_cost_surface(model, time, param_ranges::Dict, base_params::NamedTuple; 
                             param_indices::Dict=Dict(), verbose=false)
    
    param_names = collect(keys(param_ranges))
    length(param_names) == 2 || error("Must provide exactly 2 parameters")
    
    p1_name, p2_name = param_names
    p1_start, p1_stop, p1_n = param_ranges[p1_name]
    p2_start, p2_stop, p2_n = param_ranges[p2_name]
    
    p1_idx = get(param_indices, p1_name, 1)
    p2_idx = get(param_indices, p2_name, 2)
    
    # Create parameter grids
    p1_vals = range(p1_start, p1_stop, length=p1_n)
    p2_vals = range(p2_start, p2_stop, length=p2_n)
    
    # Cost surface matrix
    cost_surface = zeros(p1_n, p2_n)
    
    verbose && println("Computing cost surface for ($p1_name, $p2_name)...")
    
    for (i, p1_val) in enumerate(p1_vals)
        for (j, p2_val) in enumerate(p2_vals)
            # Update parameters
            params = collect(base_params)
            params[p1_idx] = p1_val
            params[p2_idx] = p2_val
            
            # Compute cost (simplified - adapt to your actual cost function)
            try
                # This is a placeholder - adapt based on your actual inverse problem
                cost_val = 0.0  # TODO: call straincost with updated params
                cost_surface[i,j] = cost_val
            catch
                cost_surface[i,j] = NaN
            end
        end
        verbose && mod(i, max(1, div(p1_n, 10))) == 0 && println("  Progress: $i/$p1_n")
    end
    
    return p1_vals, p2_vals, cost_surface, (p1_name, p2_name)
end

# ============================================================================
# Plotting Functions
# ============================================================================

"""
    plot_iterates_on_surface!(fig, surface_data, iterates; 
                              title="", colormap=:viridis, log_scale=false)

Plot optimization iterates on a cost surface.

# Arguments
- `fig`: GLMakie figure to plot into
- `surface_data`: (x_vals, y_vals, z_values, (name_x, name_y))
- `iterates`: Matrix of shape (n_params, n_iterates) with iterate values
- `title`: plot title
- `colormap`: color scheme for surface
- `log_scale`: use log scale for cost surface
"""
function plot_iterates_on_surface!(fig, surface_data, iterates; 
                                   title="Optimization Path on Cost Surface",
                                   colormap=:viridis,
                                   log_scale=false,
                                   param_indices=(1, 2),
                                   show_cost_values=false)
    
    x_vals, y_vals, z_vals, param_names = surface_data
    p1_idx, p2_idx = param_indices
    
    ax = Axis3(fig[1, 1], title=title, xlabel=param_names[1], ylabel=param_names[2], zlabel="Cost")
    
    # Plot surface
    z_plot = log_scale ? log10.(z_vals .+ 1e-10) : z_vals
    surface!(ax, x_vals, y_vals, z_plot', colormap=colormap, alpha=0.6, transparency=true)
    
    # Extract iterate positions on the surface
    iterate_x = iterates[p1_idx, :]
    iterate_y = iterates[p2_idx, :]
    
    # Interpolate cost values at iterate positions
    interp = linear_interpolation((collect(x_vals), collect(y_vals)), z_plot)
    iterate_z = [interp(iterate_x[i], iterate_y[i]) for i in 1:length(iterate_x)]
    
    # Plot iterate path
    lines!(ax, iterate_x, iterate_y, iterate_z, color=1:length(iterate_x), 
           colormap=:hot, linewidth=3, label="Optimization path")
    
    # Plot iterate points
    scatter!(ax, iterate_x, iterate_y, iterate_z, 
             color=1:length(iterate_x), colormap=:hot, markersize=8,
             label="Iterates")
    
    # Optionally show cost values at each iterate
    if show_cost_values
        for i in 1:length(iterate_x)
            text!(ax, iterate_x[i], iterate_y[i], iterate_z[i],
                  text="$(round(iterate_z[i], digits=3))", fontsize=10)
        end
    end
    
    axislegend(ax, position=:lt)
    return ax
end

"""
    plot_multiple_surfaces(surface_list::Vector; figsize=(1200, 1000))

Create a grid of plots showing iterates on multiple surfaces.

# Arguments
- `surface_list`: List of (surface_data, iterates, title, param_indices)
- `figsize`: (width, height) of output figure
"""
function plot_multiple_surfaces(surface_list::Vector; figsize=(1200, 1000))
    
    n_surfaces = length(surface_list)
    n_cols = ceil(Int, sqrt(n_surfaces))
    n_rows = ceil(Int, n_surfaces / n_cols)
    
    fig = Figure(size=figsize)
    
    for (idx, (surface_data, iterates, title, param_indices)) in enumerate(surface_list)
        row = div(idx - 1, n_cols) + 1
        col = mod(idx - 1, n_cols) + 1
        
        ax = Axis3(fig[row, col], title=title, 
                   xlabel=surface_data[4][1], 
                   ylabel=surface_data[4][2],
                   zlabel="Cost")
        
        x_vals, y_vals, z_vals, param_names = surface_data
        
        # Plot surface
        surface!(ax, x_vals, y_vals, z_vals', colormap=:viridis, alpha=0.5)
        
        # Extract and plot iterates
        p1_idx, p2_idx = param_indices
        iterate_x = iterates[p1_idx, :]
        iterate_y = iterates[p2_idx, :]
        
        interp = linear_interpolation((collect(x_vals), collect(y_vals)), z_vals)
        iterate_z = [interp(iterate_x[i], iterate_y[i]) for i in 1:length(iterate_x)]
        
        lines!(ax, iterate_x, iterate_y, iterate_z, 
               color=1:length(iterate_x), colormap=:hot, linewidth=2)
        scatter!(ax, iterate_x, iterate_y, iterate_z, 
                 color=1:length(iterate_x), colormap=:hot, markersize=8)
    end
    
    return fig
end

"""
    plot_iterate_history(iterates, param_names; figsize=(1200, 600))

Plot time history of each parameter during optimization.

# Arguments
- `iterates`: Matrix (n_params, n_iterates)
- `param_names`: Vector of parameter names
"""
function plot_iterate_history(iterates, param_names; figsize=(1200, 600))
    
    fig = Figure(size=figsize)
    n_params = size(iterates, 1)
    n_cols = ceil(Int, sqrt(n_params))
    n_rows = ceil(Int, n_params / n_cols)
    
    for (idx, param_name) in enumerate(param_names)
        row = div(idx - 1, n_cols) + 1
        col = mod(idx - 1, n_cols) + 1
        
        ax = Axis(fig[row, col], title=param_name, xlabel="Iteration", ylabel="Value")
        lines!(ax, 1:size(iterates, 2), vec(iterates[idx, :]), linewidth=2, label=param_name)
        scatter!(ax, 1:size(iterates, 2), vec(iterates[idx, :]), markersize=6)
        axislegend(ax)
    end
    
    return fig
end

# ============================================================================
# Example Usage Template
# ============================================================================

"""
This is a template for using this visualization module. Adapt it to your needs:

```julia
# 1. Store iterates during inverse solve (modify your inverse solve to capture these)
iterates = [A_history...] # shape: (n_params, n_iterates)

# 2. Define parameter ranges for cost surfaces
surface_configs = [
    # (param_ranges, param_indices, title)
    (Dict(:gamma0=> (-0.1, 0.1, 20), :gamma1 => (-0.05, 0.05, 20)), (1, 2), "γ₀ vs γ₁"),
    (Dict(:gamma0=> (-0.1, 0.1, 20), :x_dof1 => (-1, 1, 20)), (1, 3), "γ₀ vs X-DOF"),
    (Dict(:gamma0=> (-0.1, 0.1, 20), :tx => (-0.1, 0.1, 20)), (1, 4), "γ₀ vs Tₓ"),
]

# 3. Create surfaces and plot
surfaces = []
for (param_ranges, param_indices, title) in surface_configs
    surf_data = create_cost_surface(model, t_test, param_ranges, base_params, 
                                    param_indices=param_indices)
    push!(surfaces, (surf_data, iterates, title, param_indices))
end

# 4. Visualize
fig = plot_multiple_surfaces(surfaces)
display(fig)

# 5. Also show parameter history
param_names = ["A₁", "A₂", ...]
fig_history = plot_iterate_history(iterates, param_names)
display(fig_history)
```
"""