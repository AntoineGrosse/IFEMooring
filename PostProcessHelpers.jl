using Muscade, GLMakie

function animateStates(title, loadStepsIterator, azimuth_deg, elevation_deg, waterDepth, vec_states, savepath)
# Forward animation
    fig_anim   = Figure(size = (2000,1000))
    ax = Axis3(fig_anim[1,1],
        xgridvisible=false,ygridvisible=false,zgridvisible=false,
        xlabelsize=40,ylabelsize=40,zlabelsize=40,
        xlabel="x [m]",ylabel="y [m]",zlabel="z [m]",
        xticklabelsize=30,yticklabelsize=30,zticklabelsize=30,
        aspect = (1,1,.3),
        title=title)
    xlims!(ax,-1000,1000); ylims!(ax,-1000,1000); zlims!(ax,-waterDepth - 20,10)
    graphic = draw!(ax,vec_states[1])
    ax.azimuth[]=-π/2+π/180*azimuth_deg;
    ax.elevation[]=0+π/180*elevation_deg;
    framerate = 20
    record(fig_anim, savepath, loadStepsIterator;
    framerate = framerate) do stateIdx
        draw!(graphic,vec_states[stateIdx])
    end
end

function plotStaticStates(title, maxstep, initialstate, vec_states, savepath)
    fig      = Figure(size = (1000,1000))
    ax = Axis3(fig[1,1])
    draw!(ax,initialstate)
    # Plot the static analysis sequence
    for stateIdx ∈ 1:maxstep
        draw!(ax,vec_states[stateIdx])
    end
    save(savepath,fig)
end

function extractAxialForce(vec_states, elementsToExtractFrom, maxtimestep)
    req = @request gp(resultants(fᵢ))
    result = Matrix{Float64}(undef,length(elementsToExtractFrom), maxtimestep)
    for (i,element) in enumerate(elementsToExtractFrom)
        out = getresult(vec_states,req,[element])
        Fgp_ = [out[idxEl].gp[1][:resultants][:fᵢ] for idxEl ∈ axes(out,2)];
        result[i,:] = Fgp_[1:maxtimestep]
    end
    return result
end

function extractAxialForceFromStrainGauge(vec_states, elementsToExtractFrom, maxtimestep)
    req = @request eleres(gp(resultants(fᵢ)))
    result = Matrix{Float64}(undef,length(elementsToExtractFrom), maxtimestep)
    for (i,element) in enumerate(elementsToExtractFrom)
        out = getresult(vec_states,req,[element])
        Fgp_ = [out[idxEl].eleres.gp[1][:resultants][:fᵢ] for idxEl ∈ axes(out,2)];
        result[i,:] = Fgp_[1:maxtimestep]
    end
    return result
end

function extractUdofs(vec_states, tuplesLineSegmentNode, fields, nodeListPerLine, maxtimestep)
    all_nodes = [nodeListPerLine[iline][iseg][inode] for (iline,iseg,inode) in tuplesLineSegmentNode]
    
    # Get displacements for all nodes at each load step
    Us = Vector{}(undef,length(fields))
    for (i,field) in enumerate(fields)
        us = [getdof(vec_states[idxLoad]; class=:U, field=field, nodID=all_nodes) for idxLoad ∈ 1:maxtimestep]
        Us[i] = permutedims(hcat(us...)) # Convert to matrices for easier manipulation (rows: time steps, columns: nodes)
    end
    return Us
end

function extractDisplacements(vec_states, tuplesLineSegmentNode, nodeListPerLine, maxtimestep)
    all_nodes = [nodeListPerLine[iline][iseg][inode] for (iline,iseg,inode) in tuplesLineSegmentNode]
    fields = [:t1,:t2,:t3]
    # Get displacements for all nodes at each load step
    Xs = Vector{}(undef,length(fields))
    for (i,field) in enumerate(fields)
        xs = [getdof(vec_states[idxLoad]; field=field, nodID=all_nodes) for idxLoad ∈ 1:maxtimestep]
        Xs[i] = permutedims(hcat(xs...)) # Convert to matrices for easier manipulation (rows: time steps, columns: nodes)
    end
    return Xs
end

function plotComparisonWithSIMA(prescribed_disp_interp, xs,ys,zs, loadSteps, taper_ramp, Fgps, df, df_w, runName, t_min, t_max)
    
    Fgp1_=Fgps[1,:]
    Fgp2_=Fgps[2,:]
    Fgp3_=Fgps[3,:]

    
    # Plot comparison between Muscade and RIFLEX results. 
    fig3      = Figure(size = (1000,700))
    
    xMotion1,yMotion1,zMotion1 = prescribed_disp_interp[1]
    xMotion2,yMotion2,zMotion2 = prescribed_disp_interp[2]
    xMotion3,yMotion3,zMotion3 = prescribed_disp_interp[3]
    xs1 = xs[1]
    xs2 = xs[2]
    xs3 = xs[3]
    ax1 = Axis(fig3[1, 1],ylabel="Top x. disp. [m]")
    lines!(ax1,loadSteps,xMotion1(loadSteps),         color = :red,      linestyle = :dot,   label = "Prescribed [downwind]")
    lines!(ax1,loadSteps,xs1,         color = :red,      linestyle = :solid,   label = "Muscade [downwind]")
    lines!(ax1,loadSteps,xMotion2(loadSteps),         color = :blue,     linestyle = :dot,   label = "Prescribed [upwind right]")
    lines!(ax1,loadSteps,xs2,         color = :blue,     linestyle = :solid,   label = "Muscade [upwind right]")
    lines!(ax1,loadSteps,xMotion3(loadSteps),         color = :green,    linestyle = :dot,   label = "Prescribed [upwind left]")
    lines!(ax1,loadSteps,xs3,         color = :green,    linestyle = :solid,   label = "Muscade [upwind left]")
    vlines!(df[:,"time"][taper_ramp]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
    # axislegend()
    
    ys1 = ys[1]
    ys2 = ys[2]
    ys3 = ys[3]
    ax2 = Axis(fig3[2, 1],ylabel="Top y. disp. [m]")
    lines!(ax2,loadSteps,yMotion1(loadSteps),         color = :red,      linestyle = :dot,   label = "Prescribed [downwind]")
    lines!(ax2,loadSteps,ys1,         color = :red,      linestyle = :solid,   label = "Muscade [downwind]")
    lines!(ax2,loadSteps,yMotion2(loadSteps),         color = :blue,     linestyle = :dot,   label = "Prescribed [upwind right]")
    lines!(ax2,loadSteps,ys2,         color = :blue,     linestyle = :solid,   label = "Muscade [upwind right]")
    lines!(ax2,loadSteps,yMotion3(loadSteps),         color = :green,    linestyle = :dot,   label = "Prescribed [upwind left]")
    lines!(ax2,loadSteps,ys3,         color = :green,    linestyle = :solid,   label = "Muscade [upwind left]")
    vlines!(df[:,"time"][taper_ramp]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
    # axislegend()
    
    zs1 = zs[1]
    zs2 = zs[2]
    zs3 = zs[3]
    ax3 = Axis(fig3[3, 1],ylabel="Top vert. disp. [m]")
    lines!(ax3,loadSteps,zMotion1(loadSteps),         color = :red,      linestyle = :dot,   label = "Prescribed [downwind]")
    lines!(ax3,loadSteps,zs1,         color = :red,      linestyle = :solid,   label = "Muscade [downwind]")
    lines!(ax3,loadSteps,zMotion2(loadSteps),         color = :blue,     linestyle = :dot,   label = "Prescribed [upwind right]")
    lines!(ax3,loadSteps,zs2,         color = :blue,     linestyle = :solid,   label = "Muscade [upwind right]")
    lines!(ax3,loadSteps,zMotion3(loadSteps),         color = :green,    linestyle = :dot,   label = "Prescribed [upwind left]")
    lines!(ax3,loadSteps,zs3,         color = :green,    linestyle = :solid,   label = "Muscade [upwind left]")
    vlines!(df[:,"time"][taper_ramp]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
    # axislegend()

    Legend(fig3[4,1],ax2,orientation=:horizontal,nbanks=2)

    fig2      = Figure(size = (1000,700))
    
    ax4 = Axis(fig2[1, 1],ylabel="Line tension [kN]")
    lines!(ax4, loadSteps, Fgp1_/1e3,                     color = :red,   linestyle = :solid ,        label="Muscade [downwind]")
    lines!(ax4, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3, color = :red,   linestyle = :dot       ,  label="SIMA [downwind]")
    lines!(ax4, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3, color = :orange,   linestyle = :dot  ,  label="SIMA_full [downwind]")
    axislegend()
    
    ax5 = Axis(fig2[2, 1],ylabel="Line tension [kN]", xlabel="Time [s]")
    lines!(ax5, loadSteps, Fgp2_/1e3,                                color = :blue,             linestyle = :solid ,    label="Muscade [upwind right]")
    lines!(ax5, df[:,"time"], df[:,"TensionTopChainL2 [N]"]/1e3,            color = :blue,             linestyle = :dot   ,  label="SIMA [upwind right]")
    lines!(ax5, df_w[:,"time"], df_w[:,"TensionTopChainL2 [N]"]/1e3,        color = :purple,         linestyle = :dot   ,  label="SIMA_full [upwind right]")
    lines!(ax5, loadSteps, Fgp3_/1e3,                                color = :green,            linestyle = :solid ,     label="Muscade [upwind left]")
    lines!(ax5, df[:,"time"], df[:,"TensionTopChainL3 [N]"]/1e3,            color = :green,            linestyle = :dot   ,   label="SIMA [upwind left]")
    lines!(ax5, df_w[:,"time"], df_w[:,"TensionTopChainL3 [N]"]/1e3,        color = :grey,        linestyle = :dot   ,   label="SIMA_full [upwind left]")
    axislegend()
    
    [xlims!(idxAx,0,loadSteps[end]) for idxAx∈[ax1,ax2,ax3,ax4,ax5]]
    save("figs/"*runName*"_motions.png",fig3)
    save("figs/"*runName*"_axialtension.png",fig2)
    
    # Plot comparison between Muscade and RIFLEX results. 
    fig4      = Figure(size = (1000,700))
    
    ax1 = Axis(fig4[1,1], ylabel="Line tension [kN]", xlabel="Time [s]", ylabelsize=20, xlabelsize=20)
    lines!(ax1, loadSteps, Fgp1_/1e3,                                color = :red,             linestyle = :solid ,    label="Muscade [downwind]")
    lines!(ax1, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3,            color = :red,             linestyle = :dot   ,  label="SIMA [downwind]")
    lines!(ax1, df_w[:,"time"], df_w[:,"TensionTopChainL1 [N]"]/1e3,        color = :orange,         linestyle = :dot   ,  label="SIMA_full [downwind]")
    axislegend(position=:rb)
    ax2 = Axis(fig4[2,1], ylabel="Line tension [kN]", xlabel="Time [s]", ylabelsize=20, xlabelsize=20)
    lines!(ax2, loadSteps, Fgp2_/1e3,                                color = :blue,             linestyle = :solid ,    label="Muscade [upwind right]")
    lines!(ax2, df[:,"time"], df[:,"TensionTopChainL2 [N]"]/1e3,            color = :blue,             linestyle = :dot   ,  label="SIMA [upwind right]")
    lines!(ax2, df_w[:,"time"], df_w[:,"TensionTopChainL2 [N]"]/1e3,        color = :purple,         linestyle = :dot   ,  label="SIMA_full [upwind right]")
    lines!(ax2, loadSteps, Fgp3_/1e3,                                color = :green,            linestyle = :solid ,     label="Muscade [upwind left]")
    lines!(ax2, df[:,"time"], df[:,"TensionTopChainL3 [N]"]/1e3,            color = :green,            linestyle = :dot   ,   label="SIMA [upwind left]")
    lines!(ax2, df_w[:,"time"], df_w[:,"TensionTopChainL3 [N]"]/1e3,        color = :grey,        linestyle = :dot   ,   label="SIMA_full [upwind left]")
    axislegend(position=:rb)
    
    Fgp3_interp = linear_interpolation(
    df[:,"time"],
    df[:,"TensionTopChainL3 [N]"],
    )
    Fgp1_interp = linear_interpolation(
    df[:,"time"],
    df[:,"TensionTopChainL1 [N]"],
    )
    Δt =  df[2,"time"] - df[1,"time"]
    xlims!(ax1, t_min, t_max)
    ylims!(ax1, minimum(Fgp1_interp(t_min:Δt:t_max)/1e3)-100, maximum(Fgp1_interp(t_min:Δt:t_max)/1e3)+300)
    xlims!(ax2, t_min, t_max)
    ylims!(ax2, minimum(Fgp3_interp(t_min:Δt:t_max)/1e3)-100, maximum(Fgp3_interp(t_min:Δt:t_max)/1e3)+300)
    save("figs/"*runName*"_dynamic_zoom.png",fig4)
end

function plotComparisonWithForward(loadSteps, Fgps_inverse, Fgps_forward, Fgps_biased, Fgps_reconst, runName, t_min, t_max)
    
    Fgp1_inv=Fgps_inverse[1,:]
    Fgp2_inv=Fgps_inverse[2,:]
    Fgp3_inv=Fgps_inverse[3,:]
    Fgp1_for=Fgps_forward[1,:]
    Fgp2_for=Fgps_forward[2,:]
    Fgp3_for=Fgps_forward[3,:]
    Fgp1_bias=Fgps_biased[1,:]
    Fgp2_bias=Fgps_biased[2,:]
    Fgp3_bias=Fgps_biased[3,:]
    Fgp1_rec=Fgps_reconst[1,:]
    Fgp2_rec=Fgps_reconst[2,:]
    Fgp3_rec=Fgps_reconst[3,:]


    fig2      = Figure(size = (1000,700))
    
    ax4 = Axis(fig2[1, 1],ylabel="Line tension [kN]")
    lines!(ax4, loadSteps, Fgp1_for/1e3,                     color = :red,   linestyle = :dot       ,    label="Forward [downwind]")
    lines!(ax4, loadSteps, Fgp1_bias/1e3,                     color = :orange,   linestyle = :dash       ,    label="Measured [downwind]")
    lines!(ax4, loadSteps, Fgp1_inv/1e3,                     color = :red,   linestyle = :solid ,        label="Inverse [downwind]")
    axislegend()
    
    ax5 = Axis(fig2[2, 1],ylabel="Line tension [kN]", xlabel="Time [s]")
    lines!(ax5, loadSteps, Fgp2_for/1e3,                                color = :blue,             linestyle = :dot   ,  label="Forward [upwind right]")
    lines!(ax5, loadSteps, Fgp2_bias/1e3,                                color = :purple,             linestyle = :dash   ,  label="Measured [upwind right]")
    lines!(ax5, loadSteps, Fgp2_inv/1e3,                                color = :blue,             linestyle = :solid ,    label="Inverse [upwind right]")
    lines!(ax5, loadSteps, Fgp3_for/1e3,                                color = :green,            linestyle = :dot   ,   label="Forward [upwind left]")
    lines!(ax5, loadSteps, Fgp3_bias/1e3,                                color = :grey,            linestyle = :dash   ,   label="Measured [upwind left]")
    lines!(ax5, loadSteps, Fgp3_inv/1e3,                                color = :green,            linestyle = :solid ,     label="Inverse [upwind left]")
    axislegend()
    
    [xlims!(idxAx,0,loadSteps[end]) for idxAx∈[ax4,ax5]]
    save("figs/"*runName*"_axialtension.png",fig2)
    
    # Plot comparison between Muscade and RIFLEX results. 
    fig4      = Figure(size = (1000,700))
    
    ax1 = Axis(fig4[1,1], ylabel="Line tension [kN]", xlabel="Time [s]", ylabelsize=20, xlabelsize=20)
    lines!(ax1, loadSteps, Fgp1_for/1e3,                                color = :red,             linestyle = :dot   ,  label="Forward [downwind]")
    lines!(ax1, loadSteps, Fgp1_bias/1e3,                     color = :orange,   linestyle = :dash       ,    label="Measured input [downwind]")
    lines!(ax1, loadSteps, Fgp1_rec/1e3,                     color = :orange,   linestyle = :solid       ,    label="Measured reconstructed [downwind]")
    lines!(ax1, loadSteps, Fgp1_inv/1e3,                                color = :red,             linestyle = :solid ,    label="Inverse [downwind]")
    axislegend(position=:rt)
    ax2 = Axis(fig4[2,1], ylabel="Line tension [kN]", xlabel="Time [s]", ylabelsize=20, xlabelsize=20)
    lines!(ax2, loadSteps, Fgp2_for/1e3,                                color = :blue,             linestyle = :dot   ,  label="Forward [upwind]")
    lines!(ax2, loadSteps, Fgp2_bias/1e3,                                color = :purple,             linestyle = :dash   ,   label="Measured input [downwind]")
    lines!(ax2, loadSteps, Fgp2_rec/1e3,                                color = :purple,             linestyle = :solid   , label="Measured reconstructed [downwind]")
    lines!(ax2, loadSteps, Fgp2_inv/1e3,                                color = :blue,             linestyle = :solid ,    label="Inverse [upwind]")

    # lines!(ax2, loadSteps, Fgp2_for/1e3,                                color = :blue,             linestyle = :dot   ,  label="Forward [upwind right]")
    # lines!(ax2, loadSteps, Fgp2_bias/1e3,                                color = :purple,             linestyle = :dash   ,  label="Measured [upwind right]")
    # lines!(ax2, loadSteps, Fgp2_inv/1e3,                                color = :blue,             linestyle = :solid ,    label="Inverse [upwind right]")
    # lines!(ax2, loadSteps, Fgp3_for/1e3,                                color = :green,            linestyle = :dot   ,   label="Forward [upwind left]")
    # lines!(ax2, loadSteps, Fgp3_bias/1e3,                                color = :grey,            linestyle = :dash   ,   label="Measured [upwind left]")
    # lines!(ax2, loadSteps, Fgp3_inv/1e3,                                color = :green,            linestyle = :solid ,     label="Inverse [upwind left]")
    axislegend(position=:rb)
    
    minFgp3_interp = linear_interpolation(
    loadSteps,
    min.(Fgp2_bias,Fgp2_for,Fgp2_inv,Fgp3_bias,Fgp3_for,Fgp3_inv, Fgp2_rec, Fgp3_rec)

    )
    minFgp1_interp = linear_interpolation(
    loadSteps,
    min.(Fgp1_bias,Fgp1_for,Fgp1_inv,Fgp1_rec)
    )
    maxFgp3_interp = linear_interpolation(
    loadSteps,
    max.(Fgp2_bias,Fgp2_for,Fgp2_inv,Fgp3_bias,Fgp3_for,Fgp3_inv, Fgp2_rec, Fgp3_rec)

    )
    maxFgp1_interp = linear_interpolation(
    loadSteps,
    max.(Fgp1_bias,Fgp1_for,Fgp1_inv,Fgp1_rec)
    )
    Δt =  loadSteps[2] - loadSteps[1]
    xlims!(ax1, t_min, t_max)
    ylims!(ax1, minimum(minFgp1_interp(t_min:Δt:t_max))/1e3-100, maximum(maxFgp1_interp(t_min:Δt:t_max))/1e3+100)
    xlims!(ax2, t_min, t_max)
    ylims!(ax2, minimum(minFgp3_interp(t_min:Δt:t_max))/1e3-100, maximum(maxFgp3_interp(t_min:Δt:t_max))/1e3+100)
    save("figs/"*runName*"_dynamic_zoom.png",fig4)
end