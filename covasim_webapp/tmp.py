import covasim as cv
import pandas as pd
import numpy as np
import sciris as sc
import synthpops as sp
import json
import location_preprocessor as lp
from covasim import utils as cvu
from sus_prob_analyzer import store_seir



def create_population(pop_size, filename):
    pars = dict(n_agents=pop_size, pop_type='synthpops')
    sim = cv.Sim(pars).init_people()
    print(f"Initiation with {pop_size} is successful!")
    sim.people.save(filename)
    print(f"Population with {pop_size} is ready!")
    print()

#def tutorial_example():
#    import sciris as sc # We'll use this to time how long each one takes
#    new_pars = dict(n_agents=10000, pop_type='synthpops', n_days=100, rand_seed=0, beta=3.0)
#    sim2 = cv.Sim(new_pars, popfile='synth_pop_10k.ppl')
#    sim2.run()
#    print("Runn")


def print_contact_stats(people):
    contacts = people.contacts
    total_contacts_count = 0
    for lkey, layer in contacts.items():
        p1 = layer['p1']
        contacts_count_by_layer = len(p1)
        print(f"For layer {lkey} contact count is {contacts_count_by_layer}")
        total_contacts_count += contacts_count_by_layer 
    print(f"All contact count is : {total_contacts_count}")

def print_age_sex_stats(people):
    ages = np.array(people.age)
    sexes = np.array(people.sex)
    df_describe_ages = pd.DataFrame(ages)
    print(df_describe_ages.describe())
    df_describe_sex = pd.DataFrame(sexes)
    print(df_describe_sex.describe())

def print_stat(pars, popfile):
    sim2 = cv.Sim(pars, popfile=popfile).init_people()
    people = sim2.people
    print_contact_stats(people)
    print_age_sex_stats(people)


def print_conts():
    import sciris as sc
    obj_path = '/home/kozlov_ie/web_cov_exp/covasim_webapp/covasim_webapp/synthpops/synthpops/data/MUestimates_work.obj'
    data = sc.loadobj(obj_path)
    print(data['Zimbabwe'])

def tmp_fun():
    pop_size = 100000
    
    random_pars = dict(n_agents=pop_size, pop_type='random', contacts=dict(a=35))
    random_popfile = 'random_100k_pop.pkl'
    print_stat(random_pars, random_popfile)
    #
    synthpop_pars = dict(n_agents=pop_size, pop_type='synthpops')
    synthpop_popfile = 'synthpops_files/synth_pop_100K.ppl'
    print_stat(synthpop_pars, synthpop_popfile)


def make_dict_dist2fig():
    import pickle

    res_dict = dict()
    distnames = [
        'constants', 
        'normal_pos',
        'lognormal',
        'neg_binomial_all',
        'beta_1_all',
        'lognormal_lite_all',
        'beta_3_all',
        'beta_2_all',
        'normal_pos_all',
        'lognormal_hot_all',
        'uniform',
        'uniform_all'
    ]
    for distname in distnames:
        res_dict[distname] = plotly_dist_sus(distname)
    
    with open('susdist2figs.pkl', 'wb') as f:
        pickle.dump(res_dict, f)


def plotly_dist_sus(distname):
    import plotly.graph_objects as go

    age_dist = dict({'0-9': 2430, '10-19': 2441, '20-29': 1861, '30-39': 3334, '40-49': 3190, '50-59': 2646, '60-69': 3391, '70-79': 1524, '80+': 815})

    fig = go.Figure()
    agent_count = np.sum(list(age_dist.values()))
    rel_sus = np.zeros(agent_count, dtype=np.float32)
    progs = dict(
            age_cutoffs   = np.array([0,       10,      20,      30,      40,      50,      60,      70,      80,      90,]),     # Age cutoffs (lower limits)
            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from Zhang et al., https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
    )
    # generate simple age dist
    ppl_age = np.zeros(agent_count, dtype=np.int64)
    tmp_ind = 0
    for (ind, cnt) in enumerate(age_dist.values()):
        ppl_age[tmp_ind:tmp_ind+cnt] = ind * 10
        tmp_ind += cnt
    inds = np.digitize(ppl_age, progs['age_cutoffs'])-1

    constants_sum = np.sum(progs['sus_ORs'][inds])
    def normalize_rel_sus(rel_sus_b):
        coef = float(constants_sum) / (np.sum(rel_sus_b))
        return rel_sus_b * coef

    if distname == 'constants':
        rel_sus[:] = progs['sus_ORs'][inds]  # Default susceptibilities
    elif distname == 'normal_pos':
        for i in range(1, 10):
            inds_age = np.where((progs['age_cutoffs'][i - 1] <= ppl_age) * (ppl_age < progs['age_cutoffs'][i]))[0]
            rel_sus[inds_age] = cvu.sample(dist='normal_pos', par1=progs['sus_ORs'][i - 1], par2=0.2,
                                                size=inds_age.size)
        inds_age = np.where(ppl_age > progs['age_cutoffs'][9])[0]
        rel_sus[inds_age] = cvu.sample(dist='normal_pos', par1=progs['sus_ORs'][9], par2=0.2,
                                            size=inds_age.size)
    elif distname == 'lognormal':
        for i in range(1, 10):
            inds_age = np.where((progs['age_cutoffs'][i - 1] <= ppl_age) * (ppl_age < progs['age_cutoffs'][i]))[0]
            rel_sus[inds_age] = cvu.sample(dist='lognormal', par1=progs['sus_ORs'][i - 1], par2=0.2,
                                                size=inds_age.size)
        inds_age = np.where(ppl_age > progs['age_cutoffs'][9])[0]
        rel_sus[inds_age] = cvu.sample(dist='lognormal', par1=progs['sus_ORs'][9], par2=0.2,
                                            size=inds_age.size)
    elif distname == 'uniform':
        my_unfiform_intervals = [(0.17, 0.5), (0.5, 0.8), (0.8, 1.1), (0.8, 1.1), (0.8, 1.1), (0.8, 1.1),
                                    (1.1, 1.34), (1.34, 1.5), (1.34, 1.5), (1.34, 1.5)]
        for i in range(1, 10):
            inds_age = np.where((progs['age_cutoffs'][i - 1] <= ppl_age) * (ppl_age < progs['age_cutoffs'][i]))[0]
            rel_sus[inds_age] = np.random.uniform(
                my_unfiform_intervals[i - 1][0], my_unfiform_intervals[i - 1][1], size=inds_age.size)
        inds_age = np.where(ppl_age > progs['age_cutoffs'][9])[0]
        rel_sus[inds_age] = np.random.uniform(
            my_unfiform_intervals[9][0], my_unfiform_intervals[9][1], size=inds_age.size)
    elif distname == 'uniform_all':
        rel_sus[:]     = np.random.uniform(0, 1.47, size=rel_sus.size) # Uniform susceptibilities
    elif distname == 'lognormal_lite_all':
        rel_sus[:]     = cvu.sample(dist='lognormal', par1=0.65, par2=0.5, size=rel_sus.size) # lognormal susceptibilities
        ss = rel_sus[rel_sus > 1.5].size
        rel_sus[rel_sus > 1.5] = cvu.sample(dist='lognormal', par1=0.5, par2=0.5, size=ss)
    elif distname == 'lognormal_hot_all':
        rel_sus[:]     = cvu.sample(dist='lognormal', par1=1.0, par2=0.5, size=rel_sus.size) # lognormal susceptibilities
        ss = rel_sus[rel_sus > 1.5].size
        rel_sus[rel_sus > 1.5] = cvu.sample(dist='lognormal', par1=1.0, par2=0.5, size=ss)        
    elif distname == 'normal_pos_all':
        rel_sus[:]     = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=rel_sus.size) # normal susceptibilities
        ss = rel_sus[rel_sus > 1.5].size
        rel_sus[rel_sus > 1.5] = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=ss)
    elif distname == 'beta_1_all':
        rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=1.2, par2=6.2, step=1.5, size=rel_sus.size)
    elif distname == 'beta_2_all':
        rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=1.8, par2=6.2, step=1.5, size=rel_sus.size)
    elif distname == 'beta_3_all':
        rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=2.2, par2=5.6, step=1.5, size=rel_sus.size)
    elif distname == 'neg_binomial_all':
        rel_sus[:]     = 0.1 + cvu.sample(dist='neg_binomial', par1=0.21, par2=3, step=0.07, size=rel_sus.size)
    #elif pars['rel_sus_type'] == 'binom_2_all':
    #    rel_sus[:]     = cvu.sample(dist='beta', par1=2.2, par2=5.6, step=1.5, size=rel_sus.size)    
    else:
        raise RuntimeError("Not corrected type of rel_sus")

    rel_sus[rel_sus > 2.5] = 2.5
    rel_sus = normalize_rel_sus(rel_sus)

    bins = np.arange(0, 2.5, 0.2)
    hist_rel_sus = np.histogram(rel_sus, bins=bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    summ = np.sum(list(age_dist.values()))
    y_ttotal = hist_rel_sus[0] / summ
    if distname == 'constants':
        import plotly.express as px
        print(bins)
        print(y_ttotal)
        fig = px.bar(x=bins, y=y_ttotal, 
                     labels={"x": "Susceptibility",
                            "y": "Fraction"
                            })
    else:
        fig.add_trace(
            go.Scatter(x=bins, y=y_ttotal, name="hist sus", line_shape='spline'))

    plotly_legend = dict(legend_orientation='h', legend=dict(x=0.0, y=1.18))
    fig.update_layout(title={'text': '<b>Histogram susceptibility</b>'}, 
                      yaxis_title='Fraction', height=400, width=500,
                      yaxis_range=[-0.03, np.max(y_ttotal) + 0.1],
                      
                      xaxis_title='Susceptibility',  **plotly_legend)
    return fig


def plot_nabs():
    import plotly.express as px

    #analyzer = store_seir(show_contact_stat=False, label='seir')
    ## dict(form='nab_decay', decay_rate1=0.007701635339554948, decay_time1=250, decay_rate2=0.001)
    ## dict(form='exp_decay', init_val=-0.2, half_life=40)
    #pars = {
    #    'nab_decay': dict(form='nab_decay', decay_rate1=0.007701635339554948, decay_time1=250, decay_rate2=0.001)
    #}
    #sim = cv.Sim(n_days=300, n_agents=100000, pop_type='hybrid', analyzers=analyzer, label="default")
    #sim.run()
    #print(sim.people.peak_nab[:100])
    #print("_______________")
    nabs_arr = np.load("nab_arrs.pkl.npy")
    nabs_arr = nabs_arr.T
    for i in range(nabs_arr.shape[0]):
        print(np.max(nabs_arr[i]))
        if np.max(nabs_arr[i]) > 20:
            nabs_arr[i] = np.zeros(shape=nabs_arr.shape[1])
    
    nabs_arr = nabs_arr.T
    fig = px.line(nabs_arr)
    fig.update_layout(
        xaxis_title="time", yaxis_title="nab"
    )
    fig.show()



if __name__ == '__main__':
    #make_people_from_file('Population_Nsk.xlsx')
    #pars = dict(n_agents=100000, pop_type='synthpops')
    #people_nsk = sp.Pop.load("synthpops_files/synth_pop_Novosibirsk.ppl")
    #sim2 = cv.Sim(pars).init_people(prepared_pop=people_nsk)
    ##sim = cv.Sim(pop_size=100000, pop_type='synthpops', popfile=)
    #sim2.run()
    #pop = lp.make_people_from_pars()
    #make_dict_dist2fig()
    plot_nabs()
