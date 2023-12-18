'''
Sciris app to run the web interface.
'''

#%% Housekeeping

# Key imports
import os
import sys
import json
import base64
import copy
import tempfile
import traceback
import numpy as np
import sciris as sc
import scirisweb as sw
import covasim as cv
import shutil as sh
from pathlib import Path
import plotly.figure_factory as ff
from sus_prob_analyzer import store_seir
from functools import partial
import pandas as pd
import concurrent.futures
from itertools import repeat
import location_preprocessor as lp
import pickle

# Create the app
app = sw.ScirisApp(__name__, name="NeoCovasim")
flask_app = app.flask_app


# Set defaults
max_pop  = 20e5   # Maximum population size
max_days = 360    # Maximum number of days
max_time = 1500     # Maximum of seconds for a run
max_city_count = 20
die      = False  # Whether or not to raise exceptions instead of continuing
bgcolor  = '#eee' # Background color for app
plotbg   = '#dde'
location2filename = dict()

Incidence_and_outcomes = 'Incidence and outcomes'
General_spread_parameters = 'General spread parameters'
Spread_parameters_by_layer = 'Spread parameters by layer'
Spread_parameters_by_age = 'Spread parameters by age'
Immunity = 'Immunity'
Rest = 'Rest'
graph_groups = [
    Incidence_and_outcomes,
    General_spread_parameters,
    Immunity,
    Spread_parameters_by_layer,
    Spread_parameters_by_age,
    Rest
]

susdist2figs = dict()
with open('susdist2figs.pkl', 'rb') as f:
    susdist2figs = pickle.load(f)


#%% Define the API helper functions

@app.route('/healthcheck')
def healthcheck():
    ''' Check that the server is up '''
    return sw.robustjsonify({'status':'ok'})


def log_err(message, ex):
    ''' Compile error messages to send to the frontend '''
    tex = traceback.TracebackException.from_exception(ex)
    output = {
        "message": message,
        "exception": ''.join(traceback.format_exception(tex.exc_type, tex, tex.exc_traceback))
    }
    sc.pp(output)
    return output


@app.register_RPC()
def get_defaults(region=None, merge=False, die=die):
    ''' Get parameter defaults '''

    if region is None:
        region = 'Default'

    regions = {
        # 'n_imports': {
        #     'Default': 0,
        #     'Optimistic': 0,
        #     'Pessimistic': 10,
        # },
        'beta': {
            'Default': 0.015,
            'Optimistic': 0.010,
            'Pessimistic': 0.025,
        },
        #'web_exp2inf': {
        #    'Default': 4.0,
        #    'Optimistic': 5.0,
        #    'Pessimistic': 3.0,
        #},
        #'web_inf2sym': {
        #    'Default': 1.0,
        #    'Optimistic': 0.0,
        #    'Pessimistic': 3.0,
        #},
        #'rel_symp_prob': {
        #    'Default': 1.0,
        #    'Optimistic': 1.2,
        #    'Pessimistic': 0.5,
        #},
        #'rel_severe_prob': {
        #    'Default': 1.0,
        #    'Optimistic': 0.3,
        #    'Pessimistic': 3.0,
        #},
        #'rel_crit_prob': {
        #    'Default': 1.0,
        #    'Optimistic': 0.7,
        #    'Pessimistic': 5.0,
        #},
        #'rel_death_prob': {
        #    'Default': 1.0,
        #    'Optimistic': 0.5,
        #    'Pessimistic': 2.0,
        #},
    }

    sim_pars = dict(
        #pop_size     = dict(best=10000, min=1, max=max_pop,  name='Population size',            tip='Number of agents simulated in the model'),
        #pop_infected = dict(best=[10] * max_city_count,    min=0, max=max_pop,  name='Initial infections',         tip='Number of initial seed infections in the model'),
        n_imports    = dict(best=[0] * max_city_count,     min=0, max=100,      name='Daily imported infections',  tip='Number of infections that are imported each day'),
        rand_seed    = dict(best=[0] * max_city_count,     min=0, max=100,      name='Random seed',                tip='The parameter of the random number generator; with a single stimulation, it does not matter'),
        n_days       = dict(best=90,     min=1, max=max_days, name="Simulation duration",        tip='Total duration (in days) of the simulation'),
    )

    epi_pars = dict(
        beta            = dict(best=[0.015] * max_city_count, min=0.0, max=0.2, name='Beta (infectiousness)',              tip ='Probability of infection per contact per day'),
        #web_exp2inf     = dict(best=[4.0] * max_city_count,   min=0.0, max=30,  name='Time to infectiousness (days)',      tip ='Average number of days between exposure and being infectious'),
        #web_inf2sym     = dict(best=[1.0] * max_city_count,   min=0.0, max=30,  name='Asymptomatic period (days)',         tip ='Average number of days between exposure and developing symptoms'),
        #web_dur         = dict(best=[10.0] * max_city_count,  min=0.0, max=30,  name='Infection duration (days)',          tip ='Average number of days between infection and recovery (viral shedding period)'),
        #web_timetodie   = dict(best=[6.0] * max_city_count,   min=0.0, max=30,  name='Time until death (days)',            tip ='Average number of days between becoming critically ill and death'),
        #rel_symp_prob   = dict(best=[1.0] * max_city_count,   min=0.0, max=10,  name='Symptomatic probability multiplier', tip ='Adjustment factor on literature-derived values for proportion of infected people who become symptomatic'),
        #rel_severe_prob = dict(best=[1.0] * max_city_count,   min=0.0, max=10,  name='Severe probability multiplier',      tip ='Adjustment factor on literature-derived values for proportion of symptomatic people who develop severe disease'),
        #rel_crit_prob   = dict(best=[1.0] * max_city_count,   min=0.0, max=10,  name='Critical probability multiplier',    tip ='Adjustment factor on literature-derived values for proportion of people with severe disease who become crtiically ill'),
        #rel_death_prob  = dict(best=[1.0] * max_city_count,   min=0.0, max=10,  name='Death probability multiplier',       tip ='Adjustment factor on literature-derived values for proportion of critically ill people who die'),
        tourist_contact_count  = dict(best=[4] * max_city_count,   min=0.0, max=10,  name='Tourist contact number multiplier',       tip ='Tourist contact count is equal this coefficient multiple by number of random contacts'),
    )

    #for parkey,valuedict in regions.items():
    #    if parkey in sim_pars:
    #        sim_pars[parkey]['best'] = valuedict[region] # NB, needs to be refactored -- 'Default' is just a placeholder until we have actual regions
    #    elif parkey in epi_pars:
    #        epi_pars[parkey]['best'] = valuedict[region]
    #    else:
    #        raise Exception(f'Key {parkey} not found')
    if merge:
        output = {**sim_pars, **epi_pars}
    else:
        output = {'sim_pars': sim_pars, 'epi_pars': epi_pars}

    return output


@app.register_RPC()
def get_version():
    ''' Get the version '''
    output = f'Version {cv.__version__} ({cv.__versiondate__})'
    return output


@app.register_RPC()
def get_licenses():
    cwd = Path(__file__).parent
    repo = cwd.joinpath('..')
    license = repo.joinpath('LICENSE').read_text(encoding='utf-8')
    notice = repo.joinpath('licenses/NOTICE').read_text(encoding='utf-8')
    return {
        'license': license,
        'notice': notice
    }


@app.register_RPC()
def get_location_options(enable=False):
    ''' Get the list of options for the location select '''
    locations = cv.data.show_locations(output=True).age_distributions
    if enable:
        return locations
    else:
        return []


@app.register_RPC(call_type='upload')
def upload_pars(fname):
    parameters = sc.loadjson(fname)
    if not isinstance(parameters, dict):
        raise TypeError(f'Uploaded file was a {type(parameters)} object rather than a dict')
    if  'sim_pars' not in parameters or 'epi_pars' not in parameters:
        raise KeyError(f'Parameters file must have keys "sim_pars" and "epi_pars", not {parameters.keys()}')
    return parameters


@app.register_RPC(call_type='upload')
def upload_city(fname):
    import pandas as pd
    import os
    import shutil
    df_common = pd.read_excel(fname, header=None, sheet_name="Parameters")
    location_name = df_common.iloc[0, 1]
    new_filename = f"tmp_excels/{os.path.basename(fname)}"
    shutil.copy(fname, new_filename)
    location2filename[location_name] = new_filename
    return location_name



@app.register_RPC(call_type='upload')
def upload_file(file):
    stem, ext = os.path.splitext(file)
    fd, path = tempfile.mkstemp(suffix=ext, prefix="input_", dir=tempfile.mkdtemp())
    sh.copyfile(file, path)
    return path


@app.register_RPC()
def get_vaccine_pars(vaccine_choice):
    from covasim import parameters as cvpar
    tmp_res = cvpar.get_vaccine_dose_pars(vaccine_choice)
    tmp_res['nab_init_par1'] = tmp_res['nab_init']['par1']
    tmp_res['nab_init_par2'] = tmp_res['nab_init']['par2']
    return tmp_res

@app.register_RPC()
def get_variant_pars(variant_choice):
    from covasim import parameters as cvpar
    tmp_res = cvpar.get_variant_pars(variant_choice)
    for k in tmp_res.keys():
        if 'dur' in k:
            tmp_res[k] = tmp_res[k]['par1']  
    return tmp_res

@app.register_RPC()
def get_default_cross_immunity():
    from covasim import parameters as cvpar
    return cvpar.get_cross_immunity()

@app.register_RPC()
def get_dist_figs(rel_sus_choice_list=None, tabs=None):
    cur_rel_sus_fig = {}
    for city_ind in tabs:
        rel_sus_choice = parse_rel_sus_type(rel_sus_choice_list[city_ind])
        fig = susdist2figs[rel_sus_choice]
        cur_rel_sus_fig[city_ind] = dict({'id': f'dist_sus_{city_ind}', 'json': fig.to_json()})
    return cur_rel_sus_fig

@app.register_RPC()
def get_graph_svg(fig):
    import plotly
    ff = plotly.io.from_json(fig['json'])
    return ff.to_image(format="svg")

@app.register_RPC()
def get_gantt(int_pars_list=None, intervention_config=None, n_days=90, tabs=None):
    intervention_figs = []
    for (city_ind, int_pars) in zip(tabs, int_pars_list):
        df = []
        response = {'id': f'test: {city_ind}'}
        for key,scenario in int_pars.items():
            for timeline in scenario:
                task = intervention_config[key]['formTitle'] + '(' + str(timeline.get('intervention_choice', '')) + ')'
                level = task + ' ' + str(timeline.get('level', '')) + '%'
                df.append(dict(Task=task, Start=timeline['start'], Finish=timeline['end'], Level= level))
        if len(df) > 0:
            fig = ff.create_gantt(df, height=400, index_col='Level', title='Intervention timeline',
                                show_colorbar=True, group_tasks=True, showgrid_x=True, showgrid_y=True)
            fig.update_xaxes(type='linear', range=[0, n_days])
            response['json'] = fig.to_json()
        intervention_figs.append(response)

    return intervention_figs


@app.register_RPC()
def get_gantt_variant(introduced_variants_list=None, n_days=90, tabs=None):
    variants_figs = []


    for (city_ind, introduced_variants) in zip(tabs, introduced_variants_list):
        df = []
        response = {'id': f'test: {city_ind}'}
        for variant_dict in introduced_variants:
            variant = variant_dict['variant_name']
            n_import = variant_dict['n_import']
            start_day = variant_dict['start_day']
            df.append(dict(Task=variant, Start=start_day, Finish=n_days, Level=f"{variant} with {n_import}"))
        if len(df) > 0:
            fig = ff.create_gantt(df, height=400, index_col='Level', title='Variant timeline',
                                show_colorbar=True, group_tasks=True, showgrid_x=True, showgrid_y=True)
            fig.update_xaxes(type='linear', range=[0, n_days])
            response['json'] = fig.to_json()
        variants_figs.append(response)

    return variants_figs


#%% Define the core API

def vaccinate_by_age(prob, min_age, max_age, sim):
    target_inds = cv.true((sim.people.age >= min_age) * (sim.people.age < max_age))
    inds = sim.people.uid
    vals = np.zeros(len(sim.people))
    vals[target_inds] = prob
    output = dict(inds=inds, vals=vals)
    return output

def parse_vaccine_parameters(int_str_pars): 
    prob = float(int_str_pars['level']) / 100
    min_age = 0 if int_str_pars['min_age'] == '' else int(int_str_pars['min_age'])
    max_age = 100 if int_str_pars['max_age'] == '' else int(int_str_pars['max_age'])

    return {
        #'rel_sus': int(int_str_pars['rel_sus_vaccine']) / 100,
        #'rel_symp': int(int_str_pars['rel_symp_vaccine']) / 100,
        'subtarget': partial(vaccinate_by_age, prob, min_age, max_age)
    }, {
        'prob': prob,
        'min_age': min_age,
        'max_age': max_age
    }

def parse_age_group(age_group_str):
    "possible values: 'All', '0-7', '7-18', '18-25', '25-60', '60-'"
    if age_group_str == 'All':
        return (0, 100)
    elif age_group_str == '0-7':
        return(0, 7)
    elif age_group_str == '7-18':
        return(7, 18)
    elif age_group_str == '18-25':
        return(18, 25)
    elif age_group_str == '25-60':
        return(25, 60)
    elif age_group_str == '60-':
        return(60, 100)
    else:
        raise ValueError("Develop error :(")
    

def get_people_isolated_inds(min_age, max_age, sim):
    return sim.people.uid[cv.true((sim.people.age >= min_age) * (sim.people.age < max_age))]

def parse_interventions(int_pars, is_random=False):
    '''
    Parse interventions. Format

    Args:
        int_pars = {
            'social_distance': [
                {'start': 1,  'end': 19, 'level': 'aggressive'},
                {'start': 20, 'end': 30, 'level': 'mild'},
                ],
            'school_closures': [
                {'start': 12, 'end': 14}
                ],
            'symptomatic_testing': [
                {'start': 8, 'end': 25, 'level': 60}
                ]}

    '''
    intervs = []

    if int_pars is not None:
        masterlist = []
        for ikey,intervlist in int_pars.items():
            for iconfig in intervlist:
                iconfig['ikey'] = ikey
                masterlist.append(dict(iconfig))

        for iconfig in masterlist:
            ikey  = iconfig['ikey']
            start = iconfig['start']
            end   = iconfig['end']
            level = float(iconfig['level'])/100
            
            if ikey not in ('symptomatic_testing', 'vaccinate_closures', 'contact_tracing'):
                intervention_choice = iconfig['intervention_choice']
                intervention_function = cv.change_beta if intervention_choice == "Masks" else cv.clip_edges
                intervention_base_name = "masks:" if intervention_choice == "Masks" else "removing contacts:"
            if ikey == 'social_distance':
                age_group = iconfig['age_group_choice']
                min_age, max_age = parse_age_group(age_group)
                change = 1.0-level
                # TODO
                tmp_layers = 'a' if is_random else 'c'
                if intervention_choice == "Removing contacts":
                    interv = intervention_function(days=[start, end], changes=[change, 1.0], layers=tmp_layers, func_people_isolated_inds=partial(get_people_isolated_inds, min_age, max_age), label=f"{intervention_base_name} social distance change on " + "{:.2f}".format(change))
                else:
                    interv = intervention_function(days=[start, end], changes=[change, 1.0], layers=tmp_layers, label=f"{intervention_base_name} social distance change on " + "{:.2f}".format(change))
            elif ikey == 'school_closures':
                change = 1.0-level
                interv = intervention_function(days=[start, end], changes=[change, 1.0], layers='s', label=f"{intervention_base_name} school closures change on " + "{:.2f}".format(change))
            elif ikey == 'work_closures':
                change = 1.0-level
                interv = intervention_function(days=[start, end], changes=[change, 1.0], layers='w', label=f"{intervention_base_name} work closures change on " + "{:.2f}".format(change))
            elif ikey == 'home_closures':
                change = 1.0-level
                interv = intervention_function(days=[start, end], changes=[change, 1.0], layers='h', label=f"{intervention_base_name} home closures change on " + "{:.2f}".format(change))
            elif ikey == 'symptomatic_testing':
                asymp_prob = float(iconfig['alevel'])/100
                interv = cv.test_prob(start_day=start, end_day=end, symp_prob=level, asymp_prob=asymp_prob, label=f"symptomatic testing with " + "{:.2f}".format(asymp_prob))
            elif ikey == 'vaccinate_closures':
                pars_vac, label_d = parse_vaccine_parameters(iconfig)
                vaccine_dict = dict(
                    nab_init  = dict(dist='normal', par1=float(iconfig['nab_init_par1']), par2=float(iconfig['nab_init_par2'])),
                    nab_boost = float(iconfig['nab_boost']),
                    doses     = int(iconfig['doses']),
                    interval  = int(iconfig['interval']),
                )
                interv = cv.historical_vaccinate_prob(
                    vaccine=vaccine_dict,
                    days=np.arange(start, end), 
                    subtarget=pars_vac['subtarget'],
                    label=f"""
vaccinate {(label_d['min_age'], label_d['max_age'])}
with prob """ + "{:.2f}".format(label_d['prob']),
                    prob=label_d['prob']
                )

            elif ikey == 'contact_tracing':
                trace_prob = {k:level for k in 'hswc'}
                trace_time = {k:1.0 for k in 'hswc'}
                interv = cv.contact_tracing(start_day=start, end_day=end, trace_probs=trace_prob, trace_time=trace_time)
            else:
                raise NotImplementedError

            intervs.append(interv)

    return intervs


def parse_rel_sus_type(rel_sus_type):
    if rel_sus_type == "Constant (Covasim default)":
        return 'constants'
    elif rel_sus_type == "Normal by age":
        return "normal_pos"
    elif rel_sus_type == "Normal all":
        return "normal_pos_all"
    elif rel_sus_type == "Lognormal by age":
        return "lognormal"
    elif rel_sus_type == "Lognormal lite all":
        return "lognormal_lite_all"
    elif rel_sus_type == "Lognormal hot all":
        return "lognormal_hot_all"
    elif rel_sus_type == "Beta 1 all":
        return "beta_1_all"
    elif rel_sus_type == "Beta 3 all":
        return "beta_3_all"
    elif rel_sus_type == 'Constant all':
        return "uniform"
    else:
        raise Exception(f"Unrecognised sus type: {rel_sus_type}")


def parse_parameters(sim_pars, epi_pars, int_pars, n_days, location, verbose, errs, die, infection_step, rel_sus_type, rel_trans_type, infectiousTableConfig, population_volume, city_ind):
    ''' Sanitize web parameters into actual simulation ones '''
    orig_pars = cv.make_pars()

    defaults = get_defaults(merge=True)
    web_pars = {}
    web_pars['verbose'] = verbose # Control verbosity here

    for key,entry in {**sim_pars, **epi_pars}.items():
        best   = defaults[key]['best']
        minval = defaults[key]['min']
        maxval = defaults[key]['max']

        try:
            web_pars[key] = np.clip(float(entry['best']), minval, maxval)
        except Exception as E:
            user_key = entry['name']
            user_val = entry['best']
            err = f'Could not convert parameter "{user_key}" from value "{user_val}"; using default value instead.'
            errs.append(log_err(err, E))
            web_pars[key] = best
            if die: raise

        if key in sim_pars:
            sim_pars[key]['best'] = web_pars[key]
        else:
            epi_pars[key]['best'] = web_pars[key]

    # Convert durations
    #web_pars['dur'] = sc.dcp(orig_pars['dur']) # This is complicated, so just copy it
    #web_pars['dur']['exp2inf']['par1']  = web_pars.pop('web_exp2inf')
    #web_pars['dur']['inf2sym']['par1']  = web_pars.pop('web_inf2sym')
    #web_pars['dur']['crit2die']['par1'] = web_pars.pop('web_timetodie')
    #web_dur = web_pars.pop('web_dur')
    #for key in ['asym2rec', 'mild2rec', 'sev2rec', 'crit2rec']:
    #    web_pars['dur'][key]['par1'] = web_dur

    # Add n_days
    web_pars['n_days'] = n_days

    # Add demographic
    web_pars['location'] = location

    # Add the intervention
    web_pars['interventions'] = parse_interventions(int_pars, population_volume == '100K(Random)')

    # Handle CFR -- ignore symptoms and set to 1
    #if web_pars['rand_seed'] == 0:
    #    web_pars['rand_seed'] = None
    web_pars['timelimit'] = max_time  # Set the time limit
    #web_pars['pop_size'] = int(web_pars['pop_size'])  # Set data type
    
    web_pars['beta_layer'] = parseInfectiousConfig(infectiousTableConfig, city_ind)
    web_pars['is_additive_formula'] = (infection_step == "Cumulative")
    web_pars['rel_sus_type'] = parse_rel_sus_type(rel_sus_type)
    web_pars['rel_trans_type'] = 'beta_dist' if rel_trans_type == 'Independent(sus)' else 'eq_res_type'

    return web_pars

def parse_population_size(population_option):
    if population_option == "10K":
        return 10000
    elif population_option == "100K" or population_option == "100K(Random)":
        return 100000
    elif population_option == "500K":
        return 500000
    elif population_option == "1M":
        return 1000000
    elif population_option == "3M":
        return 3000000
    else:
        df_common = pd.read_excel(location2filename[population_option], header=None, sheet_name="Parameters")
        return int(df_common.iloc[1, 1])

def parseInfectiousConfig(infectiousConfig, city_ind):
    regular_dict = dict(infectiousConfig)
    key2key = dict(home_infectious='h', school_infectious='s', work_infectious='w', random_infectious='c')

    # Извлеките значения 'value'
    result = {}
    for key, value in regular_dict.items():
        fields = value['fields']
        field_values = {field['key']: field['value'] for field in fields}
        result[key2key[key]] = float(field_values['beta_layer'][city_ind])

    return result


def get_description(key):
    descriptions = {
        "variant_proportion": """
Пропорции количества инфицированных каждым вариантом
""",
        "rs": """
График R - Наблюдаемое репродуктивное число по динамике регистрации случаев. 
В динамике.
График R_eff (from Covasim) - Истинное репродуктивное число по данным модели Covasim, в реальности не может быть вычислено 
            """,
        "ars": """
По формуле https://www.mdapp.co/attack-rate-formula-calculator-587/
А именно {число заболеваний за текущие 3 дня} / {среднее число восприимчивых за 3 дня} * 100.0
            """
        ,
        "sars": """
По формуле https://www.mdapp.co/attack-rate-formula-calculator-587/
А именно {кол-во агентов заразившихся на уровне layer} / {кол-во контактов на уровне layer, в которых один здоровый, а другой заразный} * 100.0
                """
        ,
        "nabs": """
Гистограмма NABs агентов.
                    """
        ,
                "immunities": """
Гистограмма иммунитета агентов.
                    """
        ,
        "hist_sus": """
Гистограмма распределения восприимчивости агентов в исходной популяции.
Включает суммарную оценку восприимчивости, определяемую генетическими, физиологическими, иммунологическими, поведенческими факторами. Ось Х - условная восприимчивость, ось Y - число агентов с данным уровнем восприимчивости.
                    """
        ,
        "part_80": """
Процент агентов, которые обеспечивают 80% заражений.
Точка (x, y) означает, что y процентов инфицированных агентов обеспечили 80% всех заражений, произошедших до дня x
        """,
        "hist_number_source_per_day": """
Распределение количества агентов, зараженных одним инфицированным, по дням. 
Точка (x, y)[day] означает, по y - сколько агентов заразили x других агентов в день [day].
        """
        ,
        "hist_number_source_cum": """
Кумулятивное распределение количества агентов, зараженных одним инфицированным.
Точка (x, y)[day] означает, по y - сколько агентов заразили x других агентов до дня [day].
        """,
        "not_infected_people_by_sus": """
Распределение неинфицированных агентов по возрасту и по уровню восприимчивости в динамике. Исходное соотношение возрастных групп и восприимчивости определяется настройкой “Distribution of susceptibility”
        """
        ,
        "not_infected_people_by_sus_norm": """
Соотношение неинфицированных агентов по возрасту для каждого уровня восприимчивости в динамике. Исходное соотношение возрастных групп и восприимчивости определяется настройкой “Distribution of susceptibility”
        """
        ,
        "infected_non_infected_group": """
Кумулятивное соотношение переболевших и неинфицированных агентов в разных возрастных группах в динамике. 
        """,
        "viral_load_per_day": """
Если схема заражения - накапливаемая, то точка (x, y) означает что за день x участники на заданном уровне обменялись друг с другом y фракций патогена.
Если схема заражения - классическая, то  точка (x, y) означает что за день x было заражено y участников на заданном уровне.
        """
        ,
        "viral_load_cum": """
Если схема заражения - накапливаемая, то точка (x, y) означает что суммарно до дня x участники на заданном уровне обменялись друг с другом y фракций патогена.
Если схема заражения - классическая, то  точка (x, y) означает что суммарно до дня x было заражено y участников на заданном уровне.
        """
        ,
        "risk_infected_by_age_group_cum": """
{Кумулятивное количество случаев заражения агентов возрастной группы k} / {Количество человек в возрастной группе k}
        """
        ,
        "risk_infected_by_age_group_per_day": """
{Количество новых случаев заражения агентов возрастной группы k} / {Количество человек в возрастной группе k}
        """
        ,
        "contact_to_sus_trans": """
Графическое представление условного среза популяции  (2000 агентов) по статусу в динамике. Каждая точка соответствует одному агенту, по оси Х распределение агентов по восприимчивости, по оси Y распределение агентов по числу контактов. 
        """
        ,
        "nab_common": "Информация о защищенности популяции от патогена",
        "people": "Состав популяции по состоянию и исходам",
        "common_sim": ["Кумулятивное число инфицированных агентов и зарегистрированных случаев (с поправкой на отсрочку и неполную эффективность регистрации). Вертикальные пунктирные линии обозначают дни начала и окончания интервенций.", 
                       "Число инфицированных агентов и зарегистрированных случаев (с поправкой на отсрочку и неполную эффективность регистрации) в сутки. Вертикальные пунктирные линии обозначают дни начала и окончания интервенций.", 
                       "Кумулятивное число заболевших по тяжести заболевания. Вертикальные пунктирные линии обозначают дни начала и окончания интервенций.",
                       "Процент вакцинированных людей"
                       ]
    }
    return descriptions[key] 


def parse_interaction_records(interaction_records, tabs):
    """
    Returns: adjancy matrix
    """
    def get_city_key(city_name):
        return int(city_name[-1])

    city_count = len(tabs)
    adjacency_matrix = np.full((city_count, city_count), 0.0) / city_count
    city_key2city_ind = dict(zip(tabs, range(city_count)))
    for interaction_record in interaction_records:
        from_city_ind = city_key2city_ind[get_city_key(interaction_record['from_city_choice'])]
        to_city_ind = city_key2city_ind[get_city_key(interaction_record['to_city_choice'])]
        adjacency_matrix[from_city_ind][to_city_ind] = float(interaction_record['level'])
    
    return adjacency_matrix


def separate_by_tabs(sim_pars, epi_pars, int_pars, infection_step_list, rel_sus_type_list, rel_trans_type_list, population_volume_list, tabs, introduced_variants_list, cross_immunity_data):
    def filter_inds(ll):
        return list(ll[i] for i in tabs)
    sim_pars_list = []
    epi_pars_list = []

    # separate sim_pars
    for city_ind in tabs:
        sim_pars_copy = copy.deepcopy(sim_pars)
        for k in sim_pars_copy.keys():
            if k == 'n_days':
                sim_pars_copy[k]['best'] = sim_pars_copy[k]['best']
            else:
                sim_pars_copy[k]['best'] = sim_pars_copy[k]['best'][city_ind]
        sim_pars_list.append(sim_pars_copy)

    # separate epi_pars
    for city_ind in tabs:
        epi_pars_copy = copy.deepcopy(epi_pars)
        for k in epi_pars_copy.keys():
            epi_pars_copy[k]['best'] = epi_pars_copy[k]['best'][city_ind]
        epi_pars_list.append(epi_pars_copy)

    # filter other
    int_pars_list = filter_inds(int_pars)
    infection_step_list = filter_inds(infection_step_list) 
    rel_sus_type_list = filter_inds(rel_sus_type_list) 
    rel_trans_type_list = filter_inds(rel_trans_type_list) 
    population_volume_list = filter_inds(population_volume_list)
    introduced_variants_list = filter_inds(introduced_variants_list)
    cross_immunity_data = filter_inds(cross_immunity_data)
    return sim_pars_list, epi_pars_list, int_pars_list, infection_step_list, rel_sus_type_list, rel_trans_type_list, population_volume_list, introduced_variants_list, cross_immunity_data

msim_with = None
prev_time = []

def get_time_from_log(sim):
    with open(f'{sim.label}_timelog.txt', 'r') as file:
        line = file.readline()
    return int(line)

@app.register_RPC()
def get_current_time():
    global msim_with
    global prev_time
    if msim_with is None:
        return 0
    min_percent = 100
    for (i, sim) in enumerate(msim_with.sims):
        try:
            tt = get_time_from_log(sim)
            prev_time[i] = tt
        except Exception as E:
            print(f"was exception {E}")
            tt = prev_time[i]
        min_percent = tt if tt < min_percent else min_percent
    return min_percent


def init_log_files(sims):
    global prev_time
    prev_time = []
    for sim in sims:
        prev_time.append(0)
        with open(f'{sim.label}_timelog.txt', 'w') as file:
            file.write('0') 

# Core plotting
def process_graphs(figs, description):
    jsons = []
    list_descr = sc.promotetolist(description)
    for (fig, descr) in zip(sc.promotetolist(figs), list_descr):
        fig.update_layout(paper_bgcolor=bgcolor, plot_bgcolor=plotbg)
        output = {'json': fig.to_json(), 'id': str(sc.uuid()), 'description': descr}
        d = json.loads(output['json'])
        d['config'] = {'responsive': True}
        output['json'] = json.dumps(d)
        jsons.append(output)
    return jsons

def plot_all_graphs(cur_sim, show_contact_stat):
    graphs = {}
    
    for graph_group in graph_groups:
        graphs[graph_group] = []
    graphs[Incidence_and_outcomes] += process_graphs(cv.plotly_sim([cur_sim]), get_description('common_sim'))
    graphs[Incidence_and_outcomes] += process_graphs(cv.plotly_people(cur_sim), get_description('people'))
    graphs[Incidence_and_outcomes] += process_graphs(cv.plot_by_variant([cur_sim]), get_description('common_sim')[:2])
    graphs[Incidence_and_outcomes] += process_graphs(cv.plotly_states_people(cur_sim), get_description('people'))
    graphs[Incidence_and_outcomes] += process_graphs(cv.plotly_states_of_recovered_people(cur_sim), get_description('people'))
    graphs[Incidence_and_outcomes] += process_graphs(cv.plot_by_variant_rel(cur_sim), get_description('variant_proportion'))

    # Basic 2
    graphs[General_spread_parameters] += process_graphs(cv.plotly_rs([cur_sim]), get_description('rs'))
    graphs[General_spread_parameters] += process_graphs(cv.plotly_ars([cur_sim]), get_description('ars'))
    # Immunity
    graphs[Immunity] += process_graphs(cv.plotly_hist_nab_per_day(cur_sim), get_description('nabs'))
    graphs[Immunity] += process_graphs(cv.plotly_hist_immunity_per_day(cur_sim), get_description('immunities'))
    graphs[Immunity] += process_graphs(cv.plotly_nabs([cur_sim]), get_description('nab_common'))

    if show_contact_stat:
        graphs[General_spread_parameters] += process_graphs(cv.plotly_part_80([cur_sim]), get_description('part_80'))
        graphs[General_spread_parameters] += process_graphs(cv.plotly_hist_number_source_per_day(cur_sim), get_description('hist_number_source_per_day'))
        graphs[General_spread_parameters] += process_graphs(cv.plotly_hist_number_source_cum(cur_sim), get_description('hist_number_source_cum'))

    # By layers
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_sars([cur_sim]), get_description('sars'))
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_viral_load_per_day([cur_sim]), get_description('viral_load_per_day'))
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_viral_load_cum([cur_sim]), get_description('viral_load_cum'))

    # By group ages
    graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_not_infected_people_by_sus(cur_sim), get_description('not_infected_people_by_sus'))
    graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_not_infected_people_by_sus_norm(cur_sim), get_description('not_infected_people_by_sus_norm'))
    graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_risk_infected_by_age_group_per_day(cur_sim), get_description('risk_infected_by_age_group_per_day'))
    graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_risk_infected_by_age_group_cum(cur_sim), get_description('risk_infected_by_age_group_cum'))
    # Rest
    graphs[Rest] += process_graphs(cv.plotly_infected_non_infected_group(cur_sim), get_description('infected_non_infected_group'))
    graphs[Rest] += process_graphs(cv.plotly_contact_to_sus_trans(cur_sim), get_description('contact_to_sus_trans'))
    
    return graphs

def plot_comparing(sims, show_contact_stat):
    graphs = {}
    for graph_group in graph_groups:
        graphs[graph_group] = []
    graphs[Incidence_and_outcomes] = []
    graphs[Incidence_and_outcomes] += process_graphs(cv.plotly_sim(sims), get_description('common_sim'))
    graphs[Incidence_and_outcomes] += process_graphs(cv.plot_by_variant(sims), get_description('common_sim')[:2])

    graphs[General_spread_parameters] = []
    graphs[General_spread_parameters] += process_graphs(cv.plotly_rs(sims), get_description('rs'))
    graphs[General_spread_parameters] += process_graphs(cv.plotly_ars(sims), get_description('ars'))
    if show_contact_stat:
        graphs[General_spread_parameters] += process_graphs(cv.plotly_part_80(sims), get_description('part_80'))

    graphs[Immunity] += process_graphs(cv.plotly_nabs(sims), get_description('nab_common'))

    graphs[Spread_parameters_by_layer] = []
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_sars(sims), get_description('sars'))
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_viral_load_per_day(sims), get_description('viral_load_per_day'))
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_viral_load_cum(sims), get_description('viral_load_cum'))
    return graphs

def execute_function(func, *args):
    return func(*args)


def build_city(sim):
    pop = lp.make_people_from_file(location2filename[sim['label']], sim['popfile'])
    return cv.Sim(pars=sim['pars'], variants=sim['variants'], datafile=sim['datafile'], analyzers=sim['analyzers'], label=sim['label']).init_people(prepared_pop=pop)


def build_parallel_cities(sims):
    inds = []
    not_ready_sims = []
    for (i, sim) in enumerate(sims):
        if isinstance(sim, dict):            
            inds.append(i)
            not_ready_sims.append(sim)
    if len(inds) > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(inds)) as pool:
            ready_sims = pool.map(build_city, not_ready_sims)
        for (ind, ready_sim) in enumerate(ready_sims):
            sims[inds[ind]] = ready_sim

    return sims        

def var_d_raw2var_d(variant_dict_raw):
    return dict(
                {"rel_beta": float(variant_dict_raw['rel_beta']), 
                "rel_symp_prob": float(variant_dict_raw['rel_symp_prob']), 
                "rel_severe_prob": float(variant_dict_raw['rel_severe_prob']), 
                "rel_crit_prob": float(variant_dict_raw['rel_crit_prob']), 
                "rel_death_prob": float(variant_dict_raw['rel_death_prob']),
                'dur_exp2inf': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_exp2inf']), par2=1.5),
                'dur_inf2sym': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_inf2sym']), par2=0.9),
                'dur_sym2sev': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_sym2sev']), par2=4.9),
                'dur_sev2crit': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_sev2crit']), par2=2.0),
                'dur_asym2rec': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_asym2rec']),  par2=2.0),
                'dur_mild2rec': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_mild2rec']),  par2=2.0),
                'dur_sev2rec': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_sev2rec']), par2=6.3),
                'dur_crit2rec': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_crit2rec']), par2=6.3),
                'dur_crit2die': dict(dist='lognormal_int', par1=float(variant_dict_raw['dur_crit2die']), par2=4.8),
                "oral_microbiota_percent": float(variant_dict_raw['oral_microbiota_percent']),
                "oral_microbiota_factor": float(variant_dict_raw['oral_microbiota_factor']),
                }
            )

def get_variant_and_cross_for_not_multi(introduced_variants_list, cross_immunity_data):
    variants_list = []
    cross_list = []
    for introduced_variants, cross_rows in zip(introduced_variants_list, cross_immunity_data):
        # make variants
        variants = []
        for variant_dict_raw in introduced_variants:
            variant_dict = var_d_raw2var_d(variant_dict_raw)
            label = variant_dict_raw['variant_name'] + '.'
            variants.append(cv.variant(
                variant=variant_dict, 
                n_imports=variant_dict_raw['n_import'], 
                days=int(variant_dict_raw['start_day']), 
                label=label
                )
            )
        variants_list.append(variants)
        # make cross matrix
        variants_count = len(introduced_variants)
        immunity = np.ones((variants_count, variants_count), dtype=float) 
        print(f"cross_rows = {cross_rows}")
        for (j_v, cross_row) in enumerate(cross_rows):
            for (i_v, variant_dict_raw) in enumerate(introduced_variants):
                immunity[j_v, i_v] = float(cross_row[variant_dict_raw['variant_name']])
        cross_list.append(immunity)
    return variants_list, cross_list


def is_equal_var_dicts(d1, d2):
    for k in d1.keys():
        if k == "variant_name":
            continue
        if d1[k] != d2[k]:
            return False
    return True


def hasnt_list_this_variant(var_list, var_dict):
    for vv in var_list:
        if is_equal_var_dicts(vv, var_dict):
            return False
    return True

def get_variant_and_cross_for_multi(introduced_variants_list, tabs):
    # preprocessing
    for (i, introduced_variants) in enumerate(introduced_variants_list):
        for variant_dict_raw in introduced_variants:
            variant_dict_raw['variant_name'] = variant_dict_raw['variant_name'] + f' of City {tabs[i]}'
 
    # make introduced_variants_set 
    introduced_variants_set = []
    for (i, introduced_variants) in enumerate(introduced_variants_list):
        for variant_dict_raw in introduced_variants:
            if hasnt_list_this_variant(introduced_variants_set, variant_dict_raw):
                introduced_variants_set.append(variant_dict_raw)
            else:
                for vv in introduced_variants_set:
                    if is_equal_var_dicts(vv, variant_dict_raw):
                        vv['variant_name'] = vv['variant_name'] + f', {tabs[i]}'
    variants_list = []
    # make variant_list
    for i in range(len(tabs)):
        variants = []
        for variant_dict_raw in introduced_variants_set:
            variant_dict = var_d_raw2var_d(variant_dict_raw)
            variants.append(cv.variant(
                variant=variant_dict, 
                n_imports=( 0 if hasnt_list_this_variant(introduced_variants_list[i], variant_dict_raw) else variant_dict_raw['n_import']), 
                days=int(variant_dict_raw['start_day']), 
                label=variant_dict_raw['variant_name']
                )
            )
        variants_list.append(variants)
    return variants_list, None


def get_variants_and_cross(introduced_variants_list, cross_immunity_data, is_multi, tabs):
    if is_multi:
        return get_variant_and_cross_for_multi(introduced_variants_list, tabs)
    return get_variant_and_cross_for_not_multi(introduced_variants_list, cross_immunity_data)



@app.register_RPC()
def run_sim(sim_pars=None, epi_pars=None, int_pars=None, datafile=None, multiple_cities=False, show_contact_stat=False, n_days=None, location=None, infection_step_list=None, rel_sus_type_list=None, rel_trans_type_list=None, population_volume_list=None, infectiousTableConfig=None, introduced_variants_list=None, tabs=None, cross_immunity_data=None, interaction_records=None, verbose=True, die=die):
    ''' Create, run, and plot everything '''
    global msim_with
    errs = []
    sim_pars_out, epi_pars_out, int_pars_out = copy.deepcopy(sim_pars), copy.deepcopy(epi_pars), copy.deepcopy(int_pars)
    (sim_pars_list, epi_pars_list, int_pars_list, infection_step_list, rel_sus_type_list, rel_trans_type_list, population_volume_list, introduced_variants_list, cross_immunity_data) = separate_by_tabs(sim_pars, epi_pars, int_pars, infection_step_list, rel_sus_type_list, rel_trans_type_list, population_volume_list, tabs, introduced_variants_list, cross_immunity_data)
    try:
        web_pars_list = []

        for (sim_pars, epi_pars, int_pars, infection_step, rel_sus_type, rel_trans_type, population_volume, city_ind) in \
            zip(sim_pars_list, epi_pars_list, int_pars_list, infection_step_list, rel_sus_type_list, rel_trans_type_list, population_volume_list, tabs):            
            web_pars = parse_parameters(sim_pars=sim_pars, epi_pars=epi_pars, int_pars=int_pars, n_days=n_days, location=location, verbose=verbose, errs=errs, die=die, infection_step=infection_step, rel_sus_type=rel_sus_type, rel_trans_type=rel_trans_type, infectiousTableConfig=infectiousTableConfig, population_volume=population_volume, city_ind=city_ind)
            if False:
                print(f'Input parameters for {city_ind}:')
                print(web_pars)
                print("---------------")
            web_pars_list.append(web_pars)
    except Exception as E:
        print(E)
        errs.append(log_err('Parameter conversion failed!', E))
        if die: raise

    predefined_pops = ['100K', '100K(Random)', '500K', '1M', '3M']
    variants_list, cross_list = get_variants_and_cross(introduced_variants_list, cross_immunity_data, multiple_cities, tabs)
    print("_______|)))))))))")
    print(variants_list)
    print(cross_list)
    print("ddd")
    # Create the sim and update the parameters
    try:
        sims = []
        for pars, population_volume, variants, cross_matrix, city_ind in zip(web_pars_list, population_volume_list, variants_list, cross_list, tabs):
            new_pop_size = parse_population_size(population_volume)
            pars['pop_size'] = new_pop_size
            pars['pop_type'] = 'synthpops' if population_volume != "100K(Random)" else 'random'
            pars['n_variants'] = len(variants)
            pars['immunity'] = cross_matrix
            popfile = f"synthpops_files/synth_pop_{population_volume}.ppl"
            analyzer = store_seir(show_contact_stat=show_contact_stat, label='seir')
            pars['pop_infected'] = 0
            if population_volume not in predefined_pops:
                sim = dict(pars=pars, datafile=datafile, analyzers=analyzer, variants=variants, label=population_volume, popfile=popfile)
            else:
                lbl = f"City {city_ind}"
                if pars['pop_type'] != 'random':
                    sim = cv.Sim(pars=pars, datafile=datafile, variants=variants, popfile=popfile, analyzers=analyzer, label=lbl)
                else:
                    sim = cv.Sim(pars=pars, datafile=datafile, variants=variants, popfile=popfile, contacts=dict(a=35), beta_layer=dict(a=pars['beta_layer']['c']), analyzers=analyzer, label=lbl)
            sims.append(sim)
        sims = build_parallel_cities(sims)
    except Exception as E:
        print('Sim creation failed!')
        print(E)
        errs.append(log_err('Sim creation failed!', E))
        if die: raise

    # Core algorithm
    try:
        cites_count = len(tabs)
        adjacency_matrix = parse_interaction_records(interaction_records, tabs) if multiple_cities else None
        msim_with = cv.MultiSim(sims=sims)
        init_log_files(msim_with.sims)
        msim_with.run(n_cpus=cites_count, mulitple_cities=multiple_cities, adjacency_matrix=adjacency_matrix, keep_people=True, verbose=False)
        init_log_files(msim_with.sims)

    except TimeoutError as TE:
        err = f"The simulation stopped on day {sim.t} because run time limit ({sim['timelimit']} seconds) was exceeded. Please reduce the population size and/or number of days simulated."
        errs.append(log_err(err, TE))
        if die: raise
    except Exception as E:
        errs.append(log_err('Sim run failed!', E))
        if die: raise
    
    try:
        is_several_cities = len(tabs) > 1
        cites_count = len(tabs) 
        n_cpus = cites_count + is_several_cities
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as pool:
            results_graph = pool.map(plot_all_graphs, msim_with.sims, repeat(show_contact_stat, cites_count))

        graphs = {}
        for graph_group in graph_groups:
            graphs[graph_group] = {}
        for (city_ind, graph) in enumerate(results_graph):
            for (k, v) in graph.items():
                graphs[k][city_ind] = v
        # comparing graph
        if is_several_cities:
            comparing_graph = plot_comparing(msim_with.sims, show_contact_stat)
            for (k, v) in comparing_graph.items():
                graphs[k]['comparing'] = v
            graphs[Rest].pop('comparing')
            graphs[Spread_parameters_by_age].pop('comparing')


    except Exception as E:
        errs.append(log_err('Plotting failed!', E))
        if die: raise

    # Create and send output files (base64 encoded content)
    #try:
    #    files,summary = get_output_files(sim)
    #except Exception as E:
    #    files = {}
    #    summary = {}
    #    errs.append(log_err('Unable to save output files!', E))
    #    if die: raise

    output = {}
    output['errs']     = errs
    output['sim_pars'] = sim_pars_out
    output['epi_pars'] = epi_pars_out
    output['int_pars'] = int_pars_out
    output['graphs']   = graphs
    output['files']    = None
    output['summary']  = None

    return output



def get_output_files(sim):
    ''' Create output files for download '''

    datestamp = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')
    ss = sim.to_excel()

    files = {}
    files['xlsx'] = {
        'filename': f'covasim_results_{datestamp}.xlsx',
        'content': 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + base64.b64encode(ss.blob).decode("utf-8"),
    }

    json_string = sim.to_json(tostring=True, verbose=False)
    files['json'] = {
        'filename': f'covasim_results_{datestamp}.json',
        'content': 'data:application/text;base64,' + base64.b64encode(json_string.encode()).decode("utf-8"),
    }

    # Summary output
    summary = {
        'days': sim.npts-1,
        'cases': round(sim.results['cum_infections'][-1]),
        'deaths': round(sim.results['cum_deaths'][-1]),
    }
    return files, summary


#%% Run the server using Flask

if __name__ == "__main__":

    os.chdir(sc.thisdir(__file__))

    if len(sys.argv) > 1:
        app.config['SERVER_PORT'] = int(sys.argv[1])
    else:
        app.config['SERVER_PORT'] = 8232
    if len(sys.argv) > 2:
        autoreload = int(sys.argv[2])
    else:
        autoreload = 1

    app.run(autoreload=autoreload)
