"""
Multi-Surface Iterate Visualization Framework

This module provides utilities for creating and comparing optimization paths
across multiple parameter space cross-sections simultaneously.
"""

using GLMakie, Interpolations, LinearAlgebra, StaticArrays

# ============================================================================
# Core Structures
# ============================================================================

"""
    IterateVisualization

Container for a single visualization showing iterates on a cost surface.
"""
struct IterateVisualization
    name::String
    param_names::Tuple{String, String}
    param_ranges::Tuple{Vector, Vector}
    cost_surface::Matrix{Float64}
    iterates::Vector  # Generic to handle different dimensions
    param_indices::Tuple{Int, Int}  # Which parameters from iterates to use
end

"""
    IterateHistory

Container for complete optimization history.
"""
struct IterateHistory
    iterates::Vector{SVector}
    times::Vector{Float64}  # Optional: time/step for each iterate
    labels::Vector{String}   # Optional: iteration labels
    metadata::Dict  # Optional: additional info
end

# ============================================================================
# Surface Creation
# ============================================================================

"""
    create_surface_slice(eval_func, param1_range, param2_range, 
                        param_indices, base_params; n_points=30)

Create a 2D slice through parameter space.

Args:
- eval_func: Function(p1, p2) -> cost_value
- param1_range, param2_range: (min, max, n_points) tuples or ranges
- param_indices: (idx1, idx2) indices in iterate vectors
- base_params: Base parameter values (for other parameters)
"""
function create_surface_slice(eval_func, param_range_1, param_range_2; 
                             verbose=false, verbose_freq=5)
    
    # Handle both tuple format (min, max, n) and range format
    if isa(param_range_1, Tuple)
        p1_vals = range(param_range_1[1], param_range_1[2], length=param_range_1[3])
    else
        p1_vals = param_range_1
    end
    
    if isa(param_range_2, Tuple)
        p2_vals = range(param_range_2[1], param_range_2[2], length=param_range_2[3])
    else
        p2_vals = param_range_2
    end
    
    n1, n2 = length(p1_vals), length(p2_vals)
    cost_surf = zeros(n1, n2)
    
    verbose && println("Creating surface with $(n1)×$(n2) = $(n1*n2) points...")
    
    for (i, p1) in enumerate(p1_vals)
        verbose && mod(i, max(1, div(n1, verbose_freq))) == 0 && 
            println("  Progress: $i/$n1")
        
        for (j, p2) in enumerate(p2_vals)
            try
                cost_surf[i, j] = eval_func(p1, p2)
            catch e
                verbose && println("    Warning at ($p1, $p2): $e")
                cost_surf[i, j] = NaN
            end
        end
    end
    
    return collect(p1_vals), collect(p2_vals), cost_surf
end

# ============================================================================
# Visualization Utilities
# ============================================================================

"""
    interpolate_iterates_to_cost(iterates, cost_surface, param_ranges, param_indices)

Given iterates and a cost surface, interpolate the cost values at iterate positions.
"""
function interpolate_iterates_to_cost(iterates, cost_surface, param_ranges, param_indices)
    p1_vals, p2_vals = param_ranges
    idx1, idx2 = param_indices
    
    # Create interpolation function
    itp = linear_interpolation(
        (collect(p1_vals), collect(p2_vals)), 
        cost_surface,
        extrapolation_bc=Line()
    )
    
    # Extract iterate coordinates and evaluate
    iterate_costs = Float64[]
    for it in iterates
        try
            p1 = it[idx1]
            p2 = it[idx2]
            c = itp(p1, p2)
            push!(iterate_costs, c)
        catch
            push!(iterate_costs, NaN)
        end
    end
    
    return iterate_costs
end

# ============================================================================
# High-Level Plotting Functions
# ============================================================================

"""
    plot_iterates_3d(name, param_names, param_ranges, cost_surface, iterates,
                     param_indices=(1,2); show_surface=true, colormap=:viridis)

Create a single 3D plot of iterates on a cost surface.
"""
function plot_iterates_3d(name, param_names, param_ranges, cost_surface, iterates,
                         param_indices=(1,2); show_surface=true, colormap=:viridis)
    
    p1_vals, p2_vals = param_ranges
    idx1, idx2 = param_indices
    
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1,1], 
        title=name,
        xlabel=param_names[1],
        ylabel=param_names[2],
        zlabel="Cost")
    
    # Plot surface if requested
    if show_surface
        surface!(ax, p1_vals, p2_vals, cost_surface', 
                colormap=colormap, alpha=0.6, transparency=true)
    end
    
    # Interpolate iterate costs
    iterate_costs = interpolate_iterates_to_cost(
        iterates, cost_surface, param_ranges, param_indices
    )
    
    # Extract iterate coordinates
    iterate_p1 = [it[idx1] for it in iterates]
    iterate_p2 = [it[idx2] for it in iterates]
    
    # Plot path
    lines!(ax, iterate_p1, iterate_p2, iterate_costs,
        color=1:length(iterates), colormap=:hot, linewidth=3, label="Path")
    
    # Plot points
    scatter!(ax, iterate_p1, iterate_p2, iterate_costs,
        color=1:length(iterates), colormap=:hot, markersize=8, label="Iterates")
    
    axislegend(ax, position=:lt)
    
    return fig, ax
end

"""
    plot_iterate_comparison(vis_list::Vector{IterateVisualization}; 
                           figsize=(1600, 1000))

Create a grid of plots comparing iterates on multiple surfaces.
"""
function plot_iterate_comparison(vis_list::Vector{IterateVisualization}; 
                                 figsize=(1600, 1000), colormap=:viridis)
    
    n = length(vis_list)
    n_cols = ceil(Int, sqrt(n))
    n_rows = ceil(Int, n / n_cols)
    
    fig = Figure(size=figsize)
    
    for (idx, vis) in enumerate(vis_list)
        row = div(idx - 1, n_cols) + 1
        col = mod(idx - 1, n_cols) + 1
        
        ax = Axis3(fig[row, col],
            title=vis.name,
            xlabel=vis.param_names[1],
            ylabel=vis.param_names[2],
            zlabel="Cost")
        
        # Plot surface
        p1_vals, p2_vals = vis.param_ranges
        surface!(ax, p1_vals, p2_vals, vis.cost_surface',
                colormap=colormap, alpha=0.5)
        
        # Plot iterates
        iterate_costs = interpolate_iterates_to_cost(
            vis.iterates, vis.cost_surface, vis.param_ranges, vis.param_indices
        )
        
        iterate_p1 = [it[vis.param_indices[1]] for it in vis.iterates]
        iterate_p2 = [it[vis.param_indices[2]] for it in vis.iterates]
        
        lines!(ax, iterate_p1, iterate_p2, iterate_costs,
            color=1:length(vis.iterates), colormap=:hot, linewidth=2)
        scatter!(ax, iterate_p1, iterate_p2, iterate_costs,
            color=1:length(vis.iterates), colormap=:hot, markersize=6)
    end
    
    return fig
end

"""
    plot_parameter_evolution(iterates, param_names; figsize=(1000, 600))

Plot time history of all parameters.
"""
function plot_parameter_evolution(iterates, param_names; figsize=(1000, 600))
    
    n_params = length(iterates[1])
    n_cols = ceil(Int, sqrt(n_params))
    n_rows = ceil(Int, n_params / n_cols)
    
    fig = Figure(size=figsize)
    
    for p in 1:n_params
        row = div(p - 1, n_cols) + 1
        col = mod(p - 1, n_cols) + 1
        
        ax = Axis(fig[row, col],
            title=get(param_names, p, "Parameter $p"),
            xlabel="Iteration",
            ylabel="Value")
        
        values = [it[p] for it in iterates]
        lines!(ax, 1:length(iterates), values, linewidth=2, color=:steelblue)
        scatter!(ax, 1:length(iterates), values, markersize=6, color=:steelblue)
    end
    
    return fig
end

# ============================================================================
# Analysis Utilities
# ============================================================================

"""
    analyze_convergence(iterates; tolerance=1e-6)

Analyze convergence properties of iterates.
"""
function analyze_convergence(iterates; tolerance=1e-6)
    
    n = length(iterates)
    
    if n < 2
        return Dict(
            :n_iterates => n,
            :converged => false,
            :reason => "Too few iterates"
        )
    end
    
    # Compute step sizes
    step_sizes = [norm(iterates[i+1] - iterates[i]) for i in 1:n-1]
    
    # Check convergence
    final_steps = step_sizes[max(1, n-5):end]
    converged = all(s < tolerance for s in final_steps)
    
    return Dict(
        :n_iterates => n,
        :converged => converged,
        :final_step_size => step_sizes[end],
        :mean_step_size => mean(step_sizes),
        :max_step_size => maximum(step_sizes),
        :convergence_rate => step_sizes[end] / (step_sizes[1] + 1e-15),  # Final / Initial
        :oscillation => std(step_sizes) / mean(step_sizes)  # Consistency
    )
end

"""
    get_statistics(iterates)

Compute statistics on iterate history.
"""
function get_statistics(iterates)
    
    n = length(iterates)
    n_params = length(iterates[1])
    
    stats = Dict()
    
    for p in 1:n_params
        values = [it[p] for it in iterates]
        stats[p] = Dict(
            :initial => values[1],
            :final => values[end],
            :change => abs(values[end] - values[1]),
            :min => minimum(values),
            :max => maximum(values),
            :mean => mean(values),
            :std => std(values)
        )
    end
    
    return stats
end
