from main import ex

seeds = [1, 2, 3, 4, 5]
init_bias_options = [True, False]
fsvgd_options = [True, False]
rpf_options = [True, False]
N_options = [10, 20, 30, 40, 50, 60]

for seed in seeds:
    for init_bias in init_bias_options:
        for rpf_option in rpf_options:
            for fsvgd_option in fsvgd_options:
                for N in N_options:
                    if (seed == 1) and (init_bias == True) and (rpf_option == True) and (fsvgd_option) == True and (N < 40):
                        continue
                    else:
                        ex.run(config_updates={
                            'N': N,
                            'random_seed': seed,
                            'use_fsvgd': fsvgd_option,
                            'use_rpf': rpf_option,
                            'bias_init': init_bias,
                        })
