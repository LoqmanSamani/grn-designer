#### diffusion

"""

    diffusion2d(M,length, depth, diffusionparam, dt, compartment_length, compartment_depth)

Diffusion in 2D with "von Neumann" condition only for neighbouring pixels are taken into account

    ll: lower left corner
    lr: lower right corner
    ul: uppert left corner
    ur: uppert right corner

    # Arguments:
    M:              molecule / morphogen
    length:         compartment_length
    depth:          compartment_depth
    diffusionparam: diffusion parameter
    dt:             timestep size

    # Usage:
    GFP[length,depth] = diffusion2d(M,length, depth, diffusionparam, dt, compartment_length, compartment_depth);
"""
function diffusion2d(M,length, depth, diffusionparam, dt, compartment_length, compartment_depth)
    # # handle 4 corner cases with 2 degrees of freedom (see function content)
    if length == 1 && depth == 1
        M[length,depth] = (
            lower_left_corner_diff(M,length, depth, diffusionparam, dt)
        )
    elseif length == compartment_length && depth == 1
        M[length,depth] = (
            lower_rigth_corner_diff(M,length, depth, diffusionparam, dt)
        )
    elseif length == 1 && depth == compartment_depth
        M[length,depth] = (
            upper_left_corner_diff(M,length, depth, diffusionparam, dt)
        )
    elseif length == compartment_length && depth == compartment_depth
        M[length,depth] = (
            upper_right_corner_diff(M,length, depth, diffusionparam, dt)
        )
    # # handle 4 side cases with 3 degrees of freedom (see function content)
    elseif depth == 1 && length != 1 && length != compartment_length
        M[length,depth] = (
            lower_side_diff(M,length, depth, diffusionparam, dt)
            )
    elseif length == 1 && depth != 1 && depth != compartment_depth
        M[length,depth] = (
            left_side_diff(M,length, depth, diffusionparam, dt)
            )
    elseif length == compartment_length && depth != 1 && depth != compartment_depth
        M[length,depth] = (
            right_side_diff(M,length, depth, diffusionparam, dt)
            )
    elseif depth == compartment_depth && length != 1 && length != compartment_length
        M[length,depth] = (
            upper_side_diff(M,length, depth, diffusionparam, dt)
            )
    else
    ## Case without edges  - 4 degrees of freedom
    # length = [2:end-1];
    # depth  = [2:end-1];
        M[length,depth] = (M[length,depth]
        + dt*diffusionparam*(
                +M[length+1,depth]     # Diffusion into the compartment from i+1
                +M[length-1,depth]     # Diffusion into the compartment from i-1
                +M[length,depth+1]     # Diffusion into the compartment from n+1
                +M[length,depth-1]     # Diffusion into the compartment from n-1
                -M[length,depth]*4     # Diffusion from this compartment into all 4 neighbouring ones
        )
        )
    end
    return M[length,depth]
end
## 4 side cases with 3 degrees of freedom
function lower_side_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase #2 1,2:end-1
    # length = [2:end-1];
    # depth  = [1];
    M[length,depth] = (M[length,depth]   # current value/concentration
        +dt*diffusionparam*(
        +M[length+1,depth]     # Diffusion into the compartment from i+1
        +M[length-1,depth]     # Diffusion into the compartment from i-1
        +M[length,depth+1]     # Diffusion into the compartment from n+1
        -M[length,depth]*3     # Diffusion from this compatment into another
        )
    )
    return M[length,depth]
end
function upper_side_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase # [2:end-1],end
    # length = [2:end-1];
    # depth  = [end];
    M[length,depth] = (M[length,depth]
        + dt*diffusionparam*(
            +M[length+1,depth]     # Diffusion into the compartment from i+1
            +M[length-1,depth]     # Diffusion into the compartment from n-1
            +M[length,depth-1]     # Diffusion into the compartment from n-1
            -M[length,depth]*3     # Diffusion from this compartment into the 2 neighbouring ones
        )
    )
    return M[length,depth]
end
function left_side_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase # 1,end
    # length = [1];
    # depth  = [2:end-1];
    M[length,depth] = (M[length,depth]
        + dt*diffusionparam*(
            +M[length+1,depth]     # Diffusion into the compartment from i+1
            +M[length,depth+1]     # Diffusion into the compartment from n+1
            +M[length,depth-1]     # Diffusion into the compartment from n-1
            -M[length,depth]*3     # Diffusion from this compartment into the 2 neighbouring ones
        )
    )
    return M[length,depth]
end
function right_side_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase # end, [2:end-1]
    # length = [end];
    # depth  = [2:end-1];
    M[length,depth] = ( M[length,depth] # current value/concentration
        + dt*diffusionparam*(
            +M[length-1,depth]        # Diffusion into the compartment from i+1
            +M[length,depth+1]        # Diffusion into the compartment from n+1
            +M[length,depth-1]        # Diffusion into the compartment from n-1
            -M[length,depth]*3          # Diffusion from this compatment into another
        )
    )
    return M[length,depth]
end
## 4 corner cases with 2 degrees of freedom
function lower_left_corner_diff(M, length, depth, diffusionparam, dt)
    ## Edgecase #1 on 1,1
    # length = 1;
    # depth  = 1;
    M[length,depth] = (M[length,depth]   # current value/concentration
        + dt*diffusionparam*(
            +M[length+1,depth]    # Diffusion into the compartment from i+1
            +M[length,depth+1]    # Diffusion into the compartment from n+1
            -M[length,depth]*2    # Diffusion from this compatment into another
        )
    )
    return M[length,depth]
end
function lower_rigth_corner_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase # on end,1
    # length = end;
    # depth  = 1;
    M[length,depth] = ( M[length,depth]         # current value/concentration
        + dt*diffusionparam*(
            +M[length-1,depth]        # Diffusion into the compartment from i-1
            +M[length,depth+1]        # Diffusion into the compartment from n+1
            -M[length,depth]*2        # Diffusion from this compatment into another
            )
    )
    return M[length,depth]
end
function upper_left_corner_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase 1,end
    # length = 1
    # depth  = end
    M[length,depth] = (M[length,depth]       # current value/concentration
        + dt*diffusionparam*(
            +M[length+1,depth]       # Diffusion into the compartment from i-1
            +M[length,depth-1]       # Diffusion into the compartment from n-1
            -M[length,depth]*2       # Diffusion from this compartment into the 2 neighbouring ones
            )
        # -degradationrate*M[length,depth]*dt # Degradation
    )
    return M[length,depth]
end
function upper_right_corner_diff(M,length, depth, diffusionparam, dt)
    ## Edgecase end,end
    #  length = end
    #  depth  = end
    M[length,depth] = (M[length,depth]
               + dt*diffusionparam*(
               +M[length-1,depth]                # Diffusion into the compartment from i-1
               +M[length,depth-1]                # Diffusion into the compartment from n-1
               -2*M[length,depth]                # Diffusion from this compartment into the 2 neighbouring ones
               )
    )
    return M[length,depth]
end