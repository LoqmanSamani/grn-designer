#---
using DrWatson
#@quickactivate 
# include helper functions from src folder
using HDF5
include("set_initial_condition.jl")
include("diffuse2D.jl")
include("reactions.jl")

"""
    simulate2DMS(theta)

simultate finite differences in a compartment_length times compartment_depth 2D environment.

# Examples
    results = simulate2DMS(theta);

# Dependencies        
    set_initial_condition.jl
    diffuse2D.jl
    myoptions

"""
runs_expectedvalue_filename = "/home/samani/Documents/sim/sim6.h5"

function simulate2DMS(theta::Dict)

    saveStepInterval = theta["saveStepInterval"];
    cellNumberInitial = theta["cellSeed"]/(theta["compartment_length"]*theta["compartment_depth"] );
    growthRate    = theta["growthRate"];
    maxCellNumber = theta["maxCellNumber"] ;
    cl = theta["compartment_length"]+1; # +1 since julia is index 1
    cd = theta["compartment_depth"]+1;
    t = 0.0+1;
    tmax = theta["tmax"]; 
    dt = theta["dt"];
    num_timesteps = Int(ceil(tmax / dt)) + 1;
    timestep = 1;
    
    fM  = zeros(cl,cd);
    bM  = zeros(cl,cd);
    iM  = zeros(cl,cd);
    ibM = zeros(cl,cd);
    cells_anker     = zeros(cl,cd);
    cells_GFP       = zeros(cl,cd);
    cells_mCherry   = zeros(cl,cd);
    cells_iM        = zeros(cl,cd);

    bM_all  = zeros(cl,cd, Int(ceil(num_timesteps/saveStepInterval)));
    fM_all  = zeros(cl,cd, Int(ceil(num_timesteps/saveStepInterval)));
    iM_all  = zeros(cl,cd, Int(ceil(num_timesteps/saveStepInterval)));
    ibM_all = zeros(cl,cd, Int(ceil(num_timesteps/saveStepInterval)));
    cells_anker_all     = zeros(cl,cd, Int32(ceil(num_timesteps/saveStepInterval)));      
    cells_GFP_all       = zeros(cl,cd, Int32(ceil(num_timesteps/saveStepInterval)));        
    cells_mCherry_all   = zeros(cl,cd, Int32(ceil(num_timesteps/saveStepInterval)));    
    cells_iM_all        = zeros(cl,cd, Int32(ceil(num_timesteps/saveStepInterval)));
    # load initial cell numbers from the ic-.. file defined in myoptions
    # cells_anker, cells_GFP,cells_mCherry,cells_iM = initialcellseed(cl,cd,cellNumberInitial,cells_anker, cells_GFP,cells_mCherry,cells_iM);

    fM_all[:,:, 1] = fM;
    bM_all[:,:, 1] = bM;
    iM_all[:,:, 1] = iM;
    ibM_all[:,:, 1] = ibM;
    cells_anker_all[:,:, 1] = cells_anker;
    cells_GFP_all[:,:, 1] = cells_GFP;
    cells_mCherry_all[:,:, 1] = cells_mCherry;
    cells_iM_all[:,:, 1] = cells_iM;

    tmp_fM  = zeros(cl,cd);
    tmp_bM  = zeros(cl,cd);
    tmp_iM  = zeros(cl,cd);
    tmp_ibM = zeros(cl,cd);
    tmp_cells_anker   = zeros(cl,cd);
    tmp_cells_GFP     = zeros(cl,cd);
    tmp_cells_mCherry = zeros(cl,cd);
    tmp_cells_iM      = zeros(cl,cd);

    while t <= tmax
        tmp_fM  = fM;
        tmp_bM  = bM;
        tmp_iM  = iM;
        tmp_ibM = ibM;
        tmp_cells_anker   = cells_anker;
        tmp_cells_GFP     = cells_GFP;
        tmp_cells_mCherry = cells_mCherry;
        tmp_cells_iM      = cells_iM;
        for length in 1:cl
        for depth in 1:cd
            # ## production
            fM[length,depth] = production(tmp_fM[length,depth], theta["k_fM_src"], cellNumberInitial, dt)
            iM[length,depth] = production(tmp_iM[length,depth], theta["k_iM_src"], cellNumberInitial, dt)
            # ## anchor_binding
            fM[length, depth], 
            bM[length, depth] = anchor_binding(tmp_fM[length,depth],  theta["k_fM_bind"], theta["k_fM_off"], dt, tmp_cells_anker[length,depth], tmp_bM[length,depth]);
            # ## inhibitor_binding
            iM[length, depth], 
            fM[length, depth], 
            ibM[length, depth] = inhibitor_binding(tmp_fM[length,depth], theta["k_iM_bind"], theta["k_iM_off"], dt, tmp_iM[length,depth], tmp_ibM[length,depth])
            # ## degradation
             fM[length,depth] = degradation(tmp_fM[length,depth], theta["k_fM_deg"], dt);
             bM[length,depth] = degradation(tmp_bM[length,depth], theta["k_bM_deg"], dt);
             iM[length,depth] = degradation(tmp_iM[length,depth], theta["k_iM_deg"], dt);
            ibM[length,depth] = degradation(tmp_ibM[length,depth], theta["k_ibM_deg"], dt);
            # ## diffusion
             fM[length,depth] = diffusion2d(tmp_fM, length, depth, theta["d_free"], dt, cl, cd);
             iM[length,depth] = diffusion2d(tmp_iM, length, depth, theta["d_i"], dt, cl, cd);
            ibM[length,depth] = diffusion2d(tmp_ibM,length, depth, theta["d_i"], dt, cl, cd);
            
            # ## Cell growth Verhulst 
            cells_anker[length,depth] = logistic_growth(tmp_cells_anker[length,depth],tmp_bM[length,depth], growthRate,maxCellNumber,dt)
            
        end # depth
        end # length
        # ## save every saveStepInterval
        if timestep % (saveStepInterval) == 0 
             fM_all[:,:, 1+Int((timestep/saveStepInterval))] = fM;
             bM_all[:,:, 1+Int((timestep/saveStepInterval))] = bM;
             iM_all[:,:, 1+Int((timestep/saveStepInterval))] = iM;
            ibM_all[:,:, 1+Int((timestep/saveStepInterval))] = ibM;
              cells_anker_all[:,:, 1+Int((timestep/saveStepInterval))] = cells_anker;
                cells_GFP_all[:,:, 1+Int((timestep/saveStepInterval))] = cells_GFP;
            cells_mCherry_all[:,:, 1+Int((timestep/saveStepInterval))] = cells_mCherry;
                 cells_iM_all[:,:, 1+Int((timestep/saveStepInterval))] = cells_iM;
        end
        # ## update time
        t += dt;
        timestep += 1
    end # t <= tmax
    simoutput = @strdict fM_all bM_all iM_all ibM_all cells_anker_all cells_GFP_all cells_mCherry_all cells_iM_all ; 
    # ## save simulation to HDF5 file in datadir
    # runs_expectedvalue_filename is defined in myoptions file 
    h5open(runs_expectedvalue_filename, "w") do file   
        # Write the updated data to the file
        write(file, "expected_fM",  simoutput["fM_all"][:,:,1:end])
        write(file, "expected_bM",  simoutput["bM_all"][:,:,1:end])
        write(file, "expected_iM",  simoutput["iM_all"][:,:,1:end])
        write(file, "expected_ibM",simoutput["ibM_all"][:,:,1:end])
    
        write(file, "expected_cells_anker_all",    simoutput["cells_anker_all"][:,:,1:end])
        write(file, "expected_cells_GFP_all",        simoutput["cells_GFP_all"][:,:,1:end])
        write(file, "expected_cells_mCherry_all",simoutput["cells_mCherry_all"][:,:,1:end])
        write(file, "expected_cells_iM_all",          simoutput["cells_iM_all"][:,:,1:end])
    end
    println("done writing to hdf5 file ", runs_expectedvalue_filename)
    return simoutput
end

theta = Dict(

"growthRate" => 0,
"maxCellNumber" => 10000,
"compartment_length" => 1000,
"compartment_depth" => 1000,
"tmax" => 20,
"dt" => 0.01,
"cellSeed" => 100000000,
"saveStepInterval" => 50,
"k_fM_src" => 0.4,
"k_iM_src" => 0.3,
"k_fM_bind" => 0.17,
"k_fM_off" => 0.12,
"k_iM_bind" => 0.09,
"k_iM_off" => 0.01,
"k_fM_deg" => 0.004,
"k_iM_deg" => 0.004,
"k_bM_deg" => 0.004,
"k_ibM_deg" => 0.042,
"d_free" => 0.2,
"d_i" => 0.15

)
print("start")

simulate2DMS(theta)
print("it is done")
