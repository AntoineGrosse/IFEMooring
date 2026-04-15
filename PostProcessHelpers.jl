using Muscade, GLMakie

function animateStates(title, loadStepsIterator, azimuth_deg, elevation_deg, waterDepth, vec_states, savepath)
# Forward animation
    fig_anim   = Figure(size = (2000,1000))
    ax = Axis3(fig_anim[1,1],xgridvisible=false,ygridvisible=false,zgridvisible=false,aspect = (1,1,.3),title=title)
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
    all_nodes = [nodeListPerLine[iline][iseg][inode] for (iline,iseg,inode,_) in tuplesLineSegmentNode]
    
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

function plotComparisonWithSIMA(prescribed_disp_interp, loadSteps, taper_ramp, Fgps, df, df_w, runName)
    
    Fgp1_=Fgps[1,:]
    Fgp2_=Fgps[2,:]
    Fgp3_=Fgps[3,:]
    
    # Plot comparison between Muscade and RIFLEX results. 
    fig3      = Figure(size = (1000,1000))

    xMotion1,yMotion1,zMotion1 = prescribed_disp_interp[1]
    xMotion2,yMotion2,zMotion2 = prescribed_disp_interp[2]
    xMotion3,yMotion3,zMotion3 = prescribed_disp_interp[3]
    ax1 = Axis(fig3[1, 1],ylabel="Top x. disp. [m]", title = "3 = Port side, 2 = Starboard side, 1 = Stern")
    lines!(ax1,loadSteps,xMotion1(loadSteps),         color = :red,      linestyle = :solid,   label = "Prescribed 1")
    lines!(ax1,loadSteps,xMotion2(loadSteps),         color = :blue,     linestyle = :solid,   label = "Prescribed 2")
    lines!(ax1,loadSteps,xMotion3(loadSteps),         color = :green,    linestyle = :solid,   label = "Prescribed 3")
    vlines!(df[:,"time"][taper_ramp]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
    axislegend()
    
    ax2 = Axis(fig3[2, 1],ylabel="Top y. disp. [m]")
    lines!(ax2,loadSteps,yMotion1(loadSteps),         color = :red,      linestyle = :solid,   label = "Prescribed 1")
    lines!(ax2,loadSteps,yMotion2(loadSteps),         color = :blue,     linestyle = :solid,   label = "Prescribed 2")
    lines!(ax2,loadSteps,yMotion3(loadSteps),         color = :green,    linestyle = :solid,   label = "Prescribed 3")
    vlines!(df[:,"time"][taper_ramp]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
    axislegend()
    
    ax3 = Axis(fig3[3, 1],ylabel="Top vert. disp. [m]")
    lines!(ax3,loadSteps,zMotion1(loadSteps),         color = :red,      linestyle = :solid,   label = "Prescribed 1")
    lines!(ax3,loadSteps,zMotion2(loadSteps),         color = :blue,     linestyle = :solid,   label = "Prescribed 2")
    lines!(ax3,loadSteps,zMotion3(loadSteps),         color = :green,    linestyle = :solid,   label = "Prescribed 3")
    vlines!(df[:,"time"][taper_ramp]; ymin = 0.0, ymax = 1.0, label = "ramp slope end")
    axislegend()
    
    ax4 = Axis(fig3[4, 1],ylabel="Axial force [kN]")
    lines!(ax4, loadSteps, Fgp1_/1e3,                     color = :red,   linestyle = :solid ,   label="Muscade1")
    lines!(ax4, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3, color = :red,   linestyle = :dot   , label="SIMA1")
    lines!(ax4, df[:,"time"], df[:,"TensionTopChainL1 [N]"]/1e3, color = :orange,   linestyle = :dot   , label="SIMA1_wavesHydroForces")
    axislegend()
    
    ax5 = Axis(fig3[5:6, 1],ylabel="Axial force [kN]", xlabel="Time [s]")
    lines!(ax5, loadSteps, Fgp2_/1e3,                                color = :blue,             linestyle = :solid ,    label="Muscade2")
    lines!(ax5, df[:,"time"], df[:,"TensionTopChainL2 [N]"]/1e3,            color = :blue,             linestyle = :dot   ,  label="SIMA2")
    lines!(ax5, df_w[:,"time"], df_w[:,"TensionTopChainL2 [N]"]/1e3,        color = :purple,         linestyle = :dot   ,  label="SIMA2_wavesHydroForces")
    lines!(ax5, loadSteps, Fgp3_/1e3,                                color = :green,            linestyle = :solid ,     label="Muscade3")
    lines!(ax5, df[:,"time"], df[:,"TensionTopChainL3 [N]"]/1e3,            color = :green,            linestyle = :dot   ,   label="SIMA3")
    lines!(ax5, df_w[:,"time"], df_w[:,"TensionTopChainL3 [N]"]/1e3,        color = :grey,        linestyle = :dot   ,   label="SIMA3_wavesHydroForces")
    axislegend()
    
    [xlims!(idxAx,0,loadSteps[end]) for idxAx∈[ax1,ax2,ax3,ax4,ax5]]
    save("figs/"*runName*"_dynamic.png",fig3)
    
    # Plot comparison between Muscade and RIFLEX results. 
    fig4      = Figure(size = (1000,500))
    
    ax2 = Axis(fig4[1,1], ylabel="Axial force [kN]", xlabel="Time [s]", ylabelsize=20, xlabelsize=20)
    lines!(ax2, loadSteps, Fgp2_/1e3,                                color = :blue,             linestyle = :solid ,    label="Muscade2")
    lines!(ax2, df[:,"time"], df[:,"TensionTopChainL2 [N]"]/1e3,            color = :blue,             linestyle = :dot   ,  label="SIMA2")
    lines!(ax2, df_w[:,"time"], df_w[:,"TensionTopChainL2 [N]"]/1e3,        color = :purple,         linestyle = :dot   ,  label="SIMA2_wavesHydroForces")
    lines!(ax2, loadSteps, Fgp3_/1e3,                                color = :green,            linestyle = :solid ,     label="Muscade3")
    lines!(ax2, df[:,"time"], df[:,"TensionTopChainL3 [N]"]/1e3,            color = :green,            linestyle = :dot   ,   label="SIMA3")
    lines!(ax2, df_w[:,"time"], df_w[:,"TensionTopChainL3 [N]"]/1e3,        color = :grey,        linestyle = :dot   ,   label="SIMA3_wavesHydroForces")
    axislegend(position=:rb)
    
    Fgp3_interp = linear_interpolation(
    df[:,"time"],
    df[:,"TensionTopChainL3 [N]"],
    )
    t_min, t_max = 300, 400 # seconds
    xlims!(ax2, t_min, t_max)
    ylims!(ax2, minimum(Fgp3_interp(t_min:1:t_max)/1e3)-100, maximum(Fgp3_interp(t_min:1:t_max)/1e3)+200)
    save("figs/"*runName*"_dynamic_zoom.png",fig4)
end