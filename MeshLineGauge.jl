function MeshLineGauge(model,rawNode,azimuth,eltype,elgaugetype,xSection,segLength,nel)
    # Assumes at least 2 segments
    # For each segment, build a vector of (matrices describing node coordinates)
    # Need to pass element arguments (orient2 for EulerBeam3D for example)
    topNode = model.nod[rawNode]
    aNode = addnode!(model,[0.,0.])
    nseg        = length(nel)
    accLength   = [0;cumsum(segLength)]
    nnodes      = nel.+1   
    topNodeCoord    = coord([topNode])[1]
    bottomNodeCoord = topNodeCoord .+ [accLength[end] * cos(azimuth), accLength[end] * sin(azimuth), 0.]
    nodeCoord   =   [
        hcat(   topNodeCoord[1] .+ cos(azimuth).*(accLength[seg] .+ ((1:nnodes[seg]).-1)/(nnodes[seg]-1)*segLength[seg]),
                topNodeCoord[2] .+ sin(azimuth).*(accLength[seg] .+ ((1:nnodes[seg]).-1)/(nnodes[seg]-1)*segLength[seg]),
                topNodeCoord[3] .+ zeros(Float64,nnodes[seg],1)) for seg=1:nseg
    ];

    # Lists with First and last node of each segment, etc.

    firstNode   = Vector{Muscade.NodID}(undef,nseg) 
    lastNode    = Vector{Muscade.NodID}(undef,nseg)
    nodeList  =   Vector{Vector{Muscade.NodID}}(undef,nseg)
    elementList = Vector{Muscade.EleID};

    # Populate lists for Segment 1
    nodid       = addnode!(model,nodeCoord[1][2:end,:])
    mesh        = hcat(nodid[1:nnodes[1]-2],nodid[2:nnodes[1]-1])
    elementList = addelement!(model,elgaugetype,[topNode.ID,nodid[1],aNode];  P=SMatrix{3,1}(0.,.5,0.),D=SMatrix{3,1}(1.,0.,0.), ElementType = eltype, elementkwargs=(mat=xSection[1],))
    elementList = vcat(elementList,addelement!(model,eltype,mesh;       mat=xSection[1]))
    firstNode[1] = topNode.ID
    lastNode[1]  = nodid[size(nodid,1)]
    nodeList[1]  =  vcat(topNode.ID, nodid);

    # Populate list for the intermediate segments (if they exist)
    if nseg>2
        for segid ∈ 2:nseg-1
            local nodid         = addnode!(model,nodeCoord[segid][2:end,:])
            firstNode[segid]    = lastNode[segid-1]
            lastNode[segid]     = nodid[size(nodid,1)]
            local mesh          = hcat(nodid[1:(nnodes[segid]-2)],nodid[2:(nnodes[segid]-1)])
            elementList=vcat(elementList,addelement!(model,eltype,[firstNode[segid],nodid[1]];  mat=xSection[segid]))
            elementList=vcat(elementList,addelement!(model,eltype,mesh;                         mat=xSection[segid]))
            nodeList[segid] = nodid
        end
    end

    # Populate list for last segment
    nodid         = addnode!(model,nodeCoord[nseg][2:end-1,:])
    firstNode[nseg]    = lastNode[nseg-1]
    lastNode[nseg]     = addnode!(model, bottomNodeCoord)
    mesh          = hcat(nodid[1:(nnodes[nseg]-3)],nodid[2:(nnodes[nseg]-2)])
    elementList=vcat(elementList,addelement!(model,eltype,[firstNode[nseg],nodid[1]];  mat=xSection[nseg]))
    elementList=vcat(elementList,addelement!(model,eltype,mesh;                        mat=xSection[nseg]))
    elementList=vcat(elementList,addelement!(model,eltype,[nodid[end],lastNode[nseg]];     mat=xSection[nseg]))
    nodeList[nseg] = vcat(nodid,lastNode[nseg])


    return nodeList,elementList,aNode,nodeCoord
end 