import covasim as cv


def additive_vs_classic_on_altai_with_interventions():
    def runn():
        start_day = "2022-01-06"
        end_day = "2022-05-14"

        pars = dict({
            'pop_size': 2163693,
            'start_day': start_day,
            'end_day': end_day,
            'rand_seed': 42,
            'pop_infected': 200,
            'interventions': cv.test_num(daily_tests='data')
        })

        pars_classic = {**pars, 'is_additive_formula': False}
        pars_additive = {**pars, 'is_additive_formula': True}

        sim_classic = cv.Sim(pars_classic, beta=0.010, datafile="tmp_epidemy_altai.csv", popfile='synthpop-altai-people.ppl', label='classic formula ')
        sim_additive = cv.Sim(pars_additive, beta=0.005, datafile="tmp_epidemy_altai.csv", popfile='synthpop-altai-people.ppl', label='additive formula ')
        msim = cv.parallel([sim_classic, sim_additive])
        fig = msim.plot(to_plot=['new_diagnoses'])
        fig.savefig("fig_name.pdf", format='pdf')

    runn()

additive_vs_classic_on_altai_with_interventions()

