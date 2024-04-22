"""
    production(M productionrate, cellNumber, dt)    
"""
function production(M, productionrate, cellNumber, dt)
    # M = (M + productionrate * cellNumber/(cellNumberInitial/2) * dt) 
    M = (M + productionrate * cellNumber * dt) 
    return M
end
"""
    degradation(M,degradationrate, dt)
"""
function degradation(M, degradationrate, dt)
    M = (M - degradationrate*M*dt)
    return M
end

"""
    free_anker_cells(cells_anker, bM, inhibited_anker_bound_M )

free anker cells that have not bound to GFP
"""
function free_anker_cells(cells_anker, bM )
    cells_anker_free = max(0,cells_anker - bM)
    return cells_anker_free
end

"""
    anchor_binding(fM, k_bind, k_off, dt, cells_anker, bM)

Dependencies:
free_anker_cells(cells_anker, bM )
    max(0,cells_anker - bM)
"""
function anchor_binding(fM, k_bind, k_off, dt, cells_anker, bM)
    fM = (fM - k_bind*fM*free_anker_cells(cells_anker,bM)*dt # what is binding
        + k_off*bM*dt # what is unbinding
        )
    bM = (bM 
        + k_bind*fM*free_anker_cells(cells_anker,bM)*dt # what is binding
        - k_off*bM*dt # what is unbinding
        )
    return fM, bM
end

"""
    inhibitor_binding(fM, k_bind, k_off, dt, iM, ibM)
"""
function inhibitor_binding(fM, k_bind, k_off, dt, iM, ibM)
    iM = (iM
    - k_bind*fM*iM*dt # what is binding to ibM
    + k_off*ibM*dt # what is unbinding
    )
    fM = (fM
    - k_bind*fM*iM*dt # what is binding to ibM
    + k_off*ibM*dt # what is unbinding
    )
    ibM = (ibM
        + k_bind*fM*iM*dt # what is binding to ibM
        - k_off*ibM*dt # what is unbinding
    )
    return iM, fM, ibM
end

"""
    logistic_growth(Cells)
Verhulst equation:
"""
function logistic_growth(Cells, bM, growthRate,maxCellNumber,dt )
   Cells = (Cells 
   + growthRate*bM*(1-Cells/maxCellNumber) * dt
   )
    return Cells
end
