from genetic_algorithm import genetic_algorithm
import h5py
import numpy as np





full_path = "/home/samani/Documents/sim"


sp1 = np.zeros((20, 20))
sp2 = np.zeros((20, 20))
sp1_cells = np.zeros((20, 20))
sp2_cells = np.zeros((20, 20))
params = np.array([[.5, .5, 4, 4, .1, .1]])
dt = 0.01
sim_start = 1
sim_stop = 20
epochs = 500
target_shape = (20, 20)

file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
target = np.array(file["sp2"])

precision_bits = {"sp1": (0, 10, 10), "sp2": (0, 10, 10), "sp1_cells": (0, 10, 10), "sp2_cells": (0, 10, 10), "params": (0, 10, 10)}



genetic_algorithm(
    population_size=100,
    specie_matrix_shape=(20, 20),
    precision_bits=precision_bits,
    num_params=6,
    generations=20,
    mutation_rates=[.01, .01, .01, .01, .01],
    crossover_rates=[.8, .8, .8, .8, .8],
    num_crossover_points=[2, 2, 2, 2, 1],
    target=target,
    target_precision_bits=(0, 10, 10),
    result_path="/home/samani/Documents/sim",
    selection_method="tournament",
    tournament_size=4,
    file_name="ga",
    dt=0.01,
    sim_start=1,
    sim_stop=20,
    epochs=500,
    fitness_trigger=2300
)

"""
=====================================================================================
                             *** Genetic Algorithm ***                               
=====================================================================================
Generation 1; Best/Max Fitness: 2222/2300; Generation Duration: 9.714666604995728
Generation 2; Best/Max Fitness: 2171/2300; Generation Duration: 12.408564329147339
Generation 3; Best/Max Fitness: 2151/2300; Generation Duration: 9.80402159690857
Generation 4; Best/Max Fitness: 2160/2300; Generation Duration: 12.663629531860352
Generation 5; Best/Max Fitness: 2179/2300; Generation Duration: 11.003901243209839
Generation 6; Best/Max Fitness: 2163/2300; Generation Duration: 12.41233229637146
Generation 7; Best/Max Fitness: 2072/2300; Generation Duration: 13.012815237045288
Generation 8; Best/Max Fitness: 2142/2300; Generation Duration: 12.628184080123901
Generation 9; Best/Max Fitness: 2173/2300; Generation Duration: 13.167481660842896
Generation 10; Best/Max Fitness: 2073/2300; Generation Duration: 19.208094120025635
Generation 11; Best/Max Fitness: 2205/2300; Generation Duration: 19.16652822494507
Generation 12; Best/Max Fitness: 2040/2300; Generation Duration: 12.865525960922241
Generation 13; Best/Max Fitness: 2033/2300; Generation Duration: 11.919644594192505
Generation 14; Best/Max Fitness: 2194/2300; Generation Duration: 12.871639966964722
Generation 15; Best/Max Fitness: 2167/2300; Generation Duration: 12.438245058059692
Generation 16; Best/Max Fitness: 2214/2300; Generation Duration: 12.933037996292114
Generation 17; Best/Max Fitness: 2230/2300; Generation Duration: 12.738369941711426
Generation 18; Best/Max Fitness: 2231/2300; Generation Duration: 12.997871160507202
Generation 19; Best/Max Fitness: 1985/2300; Generation Duration: 12.281800985336304
Generation 20; Best/Max Fitness: 2207/2300; Generation Duration: 12.802549362182617
                   -----------------------------------------------
                     Simulation Complete!
                     The best found fitness: 2231
                     Total Generations: 20
                     Average Fitness: 2150.60
                     Total Simulation Duration: 259 seconds
                   -----------------------------------------------
"""




genetic_algorithm(
    population_size=100,
    specie_matrix_shape=(20, 20),
    precision_bits=precision_bits,
    num_params=6,
    generations=20,
    mutation_rates=[.01, .01, .01, .01, .01],
    crossover_rates=[.8, .8, .8, .8, .8],
    num_crossover_points=[2, 2, 2, 2, 1],
    target=target,
    target_precision_bits=(0, 10, 10),
    result_path="/home/samani/Documents/sim",
    selection_method="roulette",
    file_name="ga",
    dt=0.01,
    sim_start=1,
    sim_stop=20,
    epochs=500,
    fitness_trigger=2300
)


"""
Simulation using Roulette Wheel Selection method
=====================================================================================
                             *** Genetic Algorithm ***                               
=====================================================================================
Generation 1; Best/Max Fitness: 2222/4000; Generation Duration: 9.793171405792236
Generation 2; Best/Max Fitness: 2048/4000; Generation Duration: 12.060910940170288
Generation 3; Best/Max Fitness: 2063/4000; Generation Duration: 10.285106897354126
Generation 4; Best/Max Fitness: 2052/4000; Generation Duration: 12.655268907546997
Generation 5; Best/Max Fitness: 1988/4000; Generation Duration: 12.633676290512085
Generation 6; Best/Max Fitness: 2040/4000; Generation Duration: 12.684281587600708
Generation 7; Best/Max Fitness: 1992/4000; Generation Duration: 12.742217063903809
Generation 8; Best/Max Fitness: 2079/4000; Generation Duration: 12.718709707260132
Generation 9; Best/Max Fitness: 2040/4000; Generation Duration: 12.927294969558716
Generation 10; Best/Max Fitness: 2252/4000; Generation Duration: 12.905120849609375
Generation 11; Best/Max Fitness: 1960/4000; Generation Duration: 12.2701256275177
Generation 12; Best/Max Fitness: 2085/4000; Generation Duration: 12.91312026977539
Generation 13; Best/Max Fitness: 2049/4000; Generation Duration: 12.151027917861938
Generation 14; Best/Max Fitness: 2025/4000; Generation Duration: 12.77616810798645
Generation 15; Best/Max Fitness: 2027/4000; Generation Duration: 12.219391584396362
Generation 16; Best/Max Fitness: 2051/4000; Generation Duration: 12.75226902961731
Generation 17; Best/Max Fitness: 2016/4000; Generation Duration: 12.153728723526001
Generation 18; Best/Max Fitness: 2114/4000; Generation Duration: 12.957456350326538
Generation 19; Best/Max Fitness: 1979/4000; Generation Duration: 12.235981941223145
Generation 20; Best/Max Fitness: 2052/4000; Generation Duration: 12.546692848205566
Generation 21; Best/Max Fitness: 2072/4000; Generation Duration: 13.124621629714966
Generation 22; Best/Max Fitness: 2049/4000; Generation Duration: 12.551552534103394
Generation 23; Best/Max Fitness: 2063/4000; Generation Duration: 12.674489498138428
Generation 24; Best/Max Fitness: 2042/4000; Generation Duration: 12.846903800964355
Generation 25; Best/Max Fitness: 1976/4000; Generation Duration: 12.661106824874878
Generation 26; Best/Max Fitness: 2057/4000; Generation Duration: 12.777565240859985
Generation 27; Best/Max Fitness: 2080/4000; Generation Duration: 12.112154722213745
Generation 28; Best/Max Fitness: 2020/4000; Generation Duration: 12.866113424301147
Generation 29; Best/Max Fitness: 2071/4000; Generation Duration: 12.78225326538086
Generation 30; Best/Max Fitness: 2044/4000; Generation Duration: 12.715431451797485
Generation 31; Best/Max Fitness: 1963/4000; Generation Duration: 11.999856233596802
Generation 32; Best/Max Fitness: 2088/4000; Generation Duration: 12.853062629699707
Generation 33; Best/Max Fitness: 2053/4000; Generation Duration: 12.461349725723267
Generation 34; Best/Max Fitness: 2053/4000; Generation Duration: 12.702967405319214
Generation 35; Best/Max Fitness: 2134/4000; Generation Duration: 12.666137456893921
Generation 36; Best/Max Fitness: 2045/4000; Generation Duration: 12.918286085128784
Generation 37; Best/Max Fitness: 2116/4000; Generation Duration: 12.817355394363403
Generation 38; Best/Max Fitness: 2110/4000; Generation Duration: 12.924347639083862
Generation 39; Best/Max Fitness: 2032/4000; Generation Duration: 13.04643440246582
Generation 40; Best/Max Fitness: 2072/4000; Generation Duration: 12.939663171768188
Generation 41; Best/Max Fitness: 2061/4000; Generation Duration: 12.26184868812561
Generation 42; Best/Max Fitness: 2098/4000; Generation Duration: 12.820005893707275
Generation 43; Best/Max Fitness: 2075/4000; Generation Duration: 12.321414947509766
Generation 44; Best/Max Fitness: 2026/4000; Generation Duration: 12.821568965911865
Generation 45; Best/Max Fitness: 2060/4000; Generation Duration: 13.216521263122559
Generation 46; Best/Max Fitness: 2047/4000; Generation Duration: 12.89412546157837
Generation 47; Best/Max Fitness: 2007/4000; Generation Duration: 12.660538911819458
Generation 48; Best/Max Fitness: 2072/4000; Generation Duration: 13.515820980072021
Generation 49; Best/Max Fitness: 2073/4000; Generation Duration: 12.772275447845459
Generation 50; Best/Max Fitness: 2041/4000; Generation Duration: 13.120394945144653
Generation 51; Best/Max Fitness: 2116/4000; Generation Duration: 12.709864139556885
Generation 52; Best/Max Fitness: 2040/4000; Generation Duration: 13.120708703994751
Generation 53; Best/Max Fitness: 2110/4000; Generation Duration: 13.018434286117554
Generation 54; Best/Max Fitness: 2077/4000; Generation Duration: 13.007287979125977
Generation 55; Best/Max Fitness: 2097/4000; Generation Duration: 12.780468225479126
Generation 56; Best/Max Fitness: 2129/4000; Generation Duration: 13.084178447723389
Generation 57; Best/Max Fitness: 2113/4000; Generation Duration: 12.26901125907898
Generation 58; Best/Max Fitness: 2112/4000; Generation Duration: 13.070876598358154
Generation 59; Best/Max Fitness: 2133/4000; Generation Duration: 13.993165016174316
Generation 60; Best/Max Fitness: 2139/4000; Generation Duration: 12.96047329902649
Generation 61; Best/Max Fitness: 2129/4000; Generation Duration: 15.03727674484253
Generation 62; Best/Max Fitness: 2116/4000; Generation Duration: 13.174480438232422
Generation 63; Best/Max Fitness: 2131/4000; Generation Duration: 13.14793062210083
Generation 64; Best/Max Fitness: 2115/4000; Generation Duration: 13.05058741569519
Generation 65; Best/Max Fitness: 2144/4000; Generation Duration: 12.535450458526611
Generation 66; Best/Max Fitness: 2140/4000; Generation Duration: 13.184065341949463
Generation 67; Best/Max Fitness: 2110/4000; Generation Duration: 12.91594648361206
Generation 68; Best/Max Fitness: 2167/4000; Generation Duration: 13.17602825164795
Generation 69; Best/Max Fitness: 2168/4000; Generation Duration: 12.202023983001709
Generation 70; Best/Max Fitness: 2117/4000; Generation Duration: 13.186075925827026
Generation 71; Best/Max Fitness: 2130/4000; Generation Duration: 12.395509481430054
Generation 72; Best/Max Fitness: 2033/4000; Generation Duration: 12.932381629943848
Generation 73; Best/Max Fitness: 2122/4000; Generation Duration: 12.561432838439941
Generation 74; Best/Max Fitness: 2133/4000; Generation Duration: 12.69325590133667
Generation 75; Best/Max Fitness: 2054/4000; Generation Duration: 12.952064037322998
Generation 76; Best/Max Fitness: 2285/4000; Generation Duration: 13.075181722640991
Generation 77; Best/Max Fitness: 2046/4000; Generation Duration: 12.209217309951782
Generation 78; Best/Max Fitness: 2042/4000; Generation Duration: 13.026135921478271
Generation 79; Best/Max Fitness: 2032/4000; Generation Duration: 13.013145685195923
Generation 80; Best/Max Fitness: 1967/4000; Generation Duration: 13.769348382949829
Generation 81; Best/Max Fitness: 1996/4000; Generation Duration: 12.486492395401001
Generation 82; Best/Max Fitness: 2047/4000; Generation Duration: 14.358282089233398
Generation 83; Best/Max Fitness: 2053/4000; Generation Duration: 16.830693244934082
Generation 84; Best/Max Fitness: 2059/4000; Generation Duration: 14.482231616973877
Generation 85; Best/Max Fitness: 2068/4000; Generation Duration: 13.848066568374634
Generation 86; Best/Max Fitness: 2033/4000; Generation Duration: 12.883932113647461
Generation 87; Best/Max Fitness: 2058/4000; Generation Duration: 13.059886932373047
Generation 88; Best/Max Fitness: 2045/4000; Generation Duration: 12.954988956451416
Generation 89; Best/Max Fitness: 2053/4000; Generation Duration: 13.170358180999756
Generation 90; Best/Max Fitness: 2067/4000; Generation Duration: 13.488936424255371
Generation 91; Best/Max Fitness: 2060/4000; Generation Duration: 13.230575561523438
Generation 92; Best/Max Fitness: 2087/4000; Generation Duration: 14.094774961471558
Generation 93; Best/Max Fitness: 2055/4000; Generation Duration: 14.290200471878052
Generation 94; Best/Max Fitness: 2060/4000; Generation Duration: 12.929114818572998
Generation 95; Best/Max Fitness: 2072/4000; Generation Duration: 12.204010009765625
Generation 96; Best/Max Fitness: 2057/4000; Generation Duration: 12.801678895950317
Generation 97; Best/Max Fitness: 2044/4000; Generation Duration: 12.89694333076477
Generation 98; Best/Max Fitness: 2148/4000; Generation Duration: 12.950957536697388
Generation 99; Best/Max Fitness: 2097/4000; Generation Duration: 13.088786125183105
Generation 100; Best/Max Fitness: 2099/4000; Generation Duration: 12.929802179336548
                   -----------------------------------------------
                     Simulation Complete!
                     The best found fitness: 2285
                     Total Generations: 100
                     Average Fitness: 2073.39
                     Total Simulation Duration: 1288 seconds
                   -----------------------------------------------

                   
"""




