"""
    set_initial_condition(theta::Dict)

# Examples
    ic = set_initial_condition(theta);
Uses theta wich loads the parameters from the parameters file in the src directory
- sets up the starting conditions
- called in simulate2DMS.jl
"""
function set_initial_condition(theta::Dict)

    cl = theta["compartment_length"]+1; # +1 since julia is index 1
    cd = theta["compartment_depth"]+1;
    t = 0.0+1;
    tmax = theta["tmax"];
    dt = theta["dt"];
    num_timesteps = Int(ceil(tmax / dt)) + 1;
    timestep = 1;
    # cellNumberInitial = theta["cellSeed"]/(theta["compartment_length"] - theta["compartment_length"]/2);
    cellNumberInitial = theta["cellSeed"]/(theta["compartment_length"]*theta["compartment_depth"] );
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
    # bM_all  = zeros(cl,cd, num_timesteps);
    # fM_all  = zeros(cl,cd, num_timesteps);
    # iM_all  = zeros(cl,cd, num_timesteps);
    # ibM_all = zeros(cl,cd, num_timesteps);
    # cells_anker_all     = zeros(cl,cd, num_timesteps);
    # cells_GFP_all       = zeros(cl,cd, num_timesteps);
    # cells_mCherry_all   = zeros(cl,cd, num_timesteps);
    # cells_iM_all        = zeros(cl,cd, num_timesteps);


    # ## change cell numbers with initial seed
    cells_anker, cells_GFP,cells_mCherry,cells_iM = initialcellseed(cl,cd,cellNumberInitial,cells_anker, cells_GFP,cells_mCherry,cells_iM);
    # ## save in Dictionary/ NamedTuple
    simulation_input = @strdict cl cd t tmax dt num_timesteps timestep cellNumberInitial fM bM iM ibM bM_all fM_all iM_all ibM_all cells_anker_all cells_GFP_all cells_mCherry_all cells_iM_all cells_anker cells_GFP cells_mCherry cells_iM
    # # simulation_input_namedtuple = NamedTuple((Symbol(key), value) for (key, value) in theta_dict);
    return simulation_input
end
