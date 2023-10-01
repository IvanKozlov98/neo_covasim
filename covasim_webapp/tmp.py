import covasim as cv
import pandas as pd
import numpy as np
import sciris as sc
import synthpops as sp
import json
import location_preprocessor as lp


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




if __name__ == '__main__':
    #make_people_from_file('Population_Nsk.xlsx')
    #pars = dict(n_agents=100000, pop_type='synthpops')
    #people_nsk = sp.Pop.load("synthpops_files/synth_pop_Novosibirsk.ppl")
    #sim2 = cv.Sim(pars).init_people(prepared_pop=people_nsk)
    ##sim = cv.Sim(pop_size=100000, pop_type='synthpops', popfile=)
    #sim2.run()
    pop = lp.make_people_from_pars()

