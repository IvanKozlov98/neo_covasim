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
import time
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
from covasim.parameters import VirusParameters
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
virus2filename = dict()
weather2filename = dict()

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
        'beta': {
            'Default': 0.015,
            'Optimistic': 0.010,
            'Pessimistic': 0.025,
        },
    }

    sim_pars = dict(
        n_imports    = dict(best=[0] * max_city_count,     min=0, max=100,      name='Daily imported infections',  tip='Number of infections that are imported each day'),
        rand_seed    = dict(best=[0] * max_city_count,     min=0, max=100,      name='Random seed',                tip='The parameter of the random number generator; with a single stimulation, it does not matter'),
        n_days       = dict(best=90,     min=1, max=max_days, name="Simulation duration",        tip='Total duration (in days) of the simulation'),
    )

    epi_pars = dict(
        beta            = dict(best=[0.015] * max_city_count, min=0.0, max=0.2, name='Beta (infectiousness)',              tip ='Probability of infection per contact per day'),
        tourist_contact_count  = dict(best=[4] * max_city_count,   min=0.0, max=10,  name='Tourist contact number multiplier',       tip ='Tourist contact count is equal this coefficient multiple by number of random contacts'),
    )

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

def get_virus_name(filename):
    df = pd.read_excel(filename, header=None)
    virus_name_ind = df[df[0] == 'Virus name'].index[0]
    virus_name = df.iloc[virus_name_ind + 1][0]
    return virus_name

@app.register_RPC(call_type='upload')
def upload_virus(fname):
    import pandas as pd
    import os
    import shutil
    df_common = pd.read_excel(fname, header=None)
    virus_name = get_virus_name(fname)
    new_filename = f"tmp_excels/{os.path.basename(fname)}"
    shutil.copy(fname, new_filename)
    virus2filename[virus_name] = new_filename
    return virus_name

@app.register_RPC(call_type='upload')
def upload_weather(fname):
    import pandas as pd
    import os
    import shutil
    weather_name = fname
    new_filename = f"tmp_excels/{os.path.basename(fname)}"
    shutil.copy(fname, new_filename)
    weather2filename[weather_name] = new_filename
    return weather_name


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
        response = {'id': f'intervention_test: {city_ind}'}
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
        response = {'id': f'variant_test: {city_ind}'}
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


def parse_parameters(sim_pars, epi_pars, int_pars, n_days, verbose, die, infection_step, rel_sus_type, rel_trans_type, infectiousTableConfig, population_volume, month_choice, monthly_weather_file, city_ind):
    ''' Sanitize web parameters into actual simulation ones '''
    orig_pars = cv.make_pars()

    defaults = get_defaults(merge=True)
    web_pars = {}
    web_pars['verbose'] = verbose # Control verbosity here

    for key,entry in {**sim_pars, **epi_pars}.items():
        best   = defaults[key]['best']
        minval = defaults[key]['min']
        maxval = defaults[key]['max']

        web_pars[key] = np.clip(float(entry['best']), minval, maxval)

        if key in sim_pars:
            sim_pars[key]['best'] = web_pars[key]
        else:
            epi_pars[key]['best'] = web_pars[key]

    # Add n_days
    web_pars['n_days'] = n_days

    # Add the intervention
    web_pars['interventions'] = parse_interventions(int_pars, population_volume == '100K(Random)')

    # Handle CFR -- ignore symptoms and set to 1
    web_pars['timelimit'] = max_time  # Set the time limit
    
    web_pars['beta_layer'] = parseInfectiousConfig(infectiousTableConfig, city_ind)
    web_pars['is_additive_formula'] = (infection_step == "Cumulative")
    web_pars['rel_sus_type'] = parse_rel_sus_type(rel_sus_type)
    web_pars['rel_trans_type'] = 'beta_dist' if rel_trans_type == 'Independent(sus)' else 'eq_res_type'
    web_pars['starting_month'] = month_choice if month_choice != "No seasonality" else None 
    if monthly_weather_file != "":
        web_pars['monthly_weather'] = pd.read_csv(weather2filename[monthly_weather_file]).to_dict('list')

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


def separate_by_tabs(kwargs):
    tabs = kwargs['tabs']

    def filter_inds(ll):
        return list(ll[i] for i in tabs)
    
    def separate_sim_pars(sim_pars_list): 
        res = []
        for city_ind in tabs:
            sim_pars_copy = copy.deepcopy(sim_pars_list)
            for k in sim_pars_copy.keys():
                if k == 'n_days':
                    sim_pars_copy[k]['best'] = sim_pars_copy[k]['best']
                else:
                    sim_pars_copy[k]['best'] = sim_pars_copy[k]['best'][city_ind]
            res.append(sim_pars_copy)
        return res

    def separate_epi_pars(epi_pars_list):
        res = []
        for city_ind in tabs:
            epi_pars_copy = copy.deepcopy(epi_pars_list)
            for k in epi_pars_copy.keys():
                epi_pars_copy[k]['best'] = epi_pars_copy[k]['best'][city_ind]
            res.append(epi_pars_copy)
        return res

    res_dict = {}
    for par_k, par_v in kwargs.items():
        if par_k == "sim_pars_list":
            res_dict[par_k] = separate_sim_pars(par_v)
        elif par_k == "epi_pars_list":
            res_dict[par_k] = separate_epi_pars(par_v)
        elif "_list" in par_k:
            res_dict[par_k] = filter_inds(par_v)
        else:
            res_dict[par_k] = par_v
    return res_dict


neo_simulator = None
prev_time = []

def get_time_from_log(sim):
    with open(sim.logfile, 'r') as file:
        line = file.readline()
    return int(line)

@app.register_RPC()
def get_current_time():
    global neo_simulator
    global prev_time
    if neo_simulator is None or neo_simulator.msim_with is None:
        return -1
    min_percent = 100
    for (i, sim) in enumerate(neo_simulator.msim_with.sims):
        try:
            tt = get_time_from_log(sim)
            prev_time[i] = tt
        except Exception as E:
            tt = prev_time[i]
        min_percent = tt if tt < min_percent else min_percent
    return min_percent


def init_log_files(sims):
    global prev_time
    prev_time = []
    for sim in sims:
        prev_time.append(0)
        with open(sim.logfile, 'w') as file:
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

def plot_all_graphs(cur_sim, show_all):
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

    if show_all:
        graphs[General_spread_parameters] += process_graphs(cv.plotly_part_80([cur_sim]), get_description('part_80'))
        graphs[General_spread_parameters] += process_graphs(cv.plotly_hist_number_source_per_day(cur_sim), get_description('hist_number_source_per_day'))
        graphs[General_spread_parameters] += process_graphs(cv.plotly_hist_number_source_cum(cur_sim), get_description('hist_number_source_cum'))
        # By group ages
        graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_not_infected_people_by_sus(cur_sim), get_description('not_infected_people_by_sus'))
        graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_not_infected_people_by_sus_norm(cur_sim), get_description('not_infected_people_by_sus_norm'))
        graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_risk_infected_by_age_group_per_day(cur_sim), get_description('risk_infected_by_age_group_per_day'))
        graphs[Spread_parameters_by_age] += process_graphs(cv.plotly_risk_infected_by_age_group_cum(cur_sim), get_description('risk_infected_by_age_group_cum'))
        # Rest
        graphs[Rest] += process_graphs(cv.plotly_infected_non_infected_group(cur_sim), get_description('infected_non_infected_group'))
        graphs[Rest] += process_graphs(cv.plotly_contact_to_sus_trans(cur_sim), get_description('contact_to_sus_trans'))
    # By layers
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_sars([cur_sim]), get_description('sars'))
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_viral_load_per_day([cur_sim]), get_description('viral_load_per_day'))
    graphs[Spread_parameters_by_layer] += process_graphs(cv.plotly_viral_load_cum([cur_sim]), get_description('viral_load_cum'))



    return graphs

def plot_comparing(sims, show_all):
    graphs = {}
    for graph_group in graph_groups:
        graphs[graph_group] = []
    graphs[Incidence_and_outcomes] = []
    graphs[Incidence_and_outcomes] += process_graphs(cv.plotly_sim(sims), get_description('common_sim'))
    graphs[Incidence_and_outcomes] += process_graphs(cv.plot_by_variant(sims), get_description('common_sim')[:2])

    graphs[General_spread_parameters] = []
    graphs[General_spread_parameters] += process_graphs(cv.plotly_rs(sims), get_description('rs'))
    graphs[General_spread_parameters] += process_graphs(cv.plotly_ars(sims), get_description('ars'))
    if show_all:
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
    #pop = sp.Pop.load('synthpops_files/synth_pop_N.Novgorod.ppl')
    print("Befire firiii")
    res = cv.Sim(
        pars=sim['pars'], datafile=sim['datafile'], analyzers=sim['analyzers'], 
        virus_parameters=sim['virus_parameters'], variants=sim['variants'], 
        label=sim['label']
    ).init_people(prepared_pop=pop)
    return res


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
        for (j_v, cross_row) in enumerate(cross_rows):
            for (i_v, variant_dict_raw) in enumerate(introduced_variants):
                immunity[j_v, i_v] = float(cross_row[variant_dict_raw['variant_name']])
        cross_list.append(immunity)
    return variants_list, cross_list


def is_equal_var_dicts(d1, d2):
    for k in d1.keys():
        if k == "variant_name" or k == "possible_name":
            continue
        if d1[k] != d2[k]:
            return False
    return True


def hasnt_list_this_variant(var_list, var_dict):
    for vv in var_list:
        if is_equal_var_dicts(vv, var_dict):
            return False
    return True

def get_variant_and_cross_for_multi(introduced_variants_list, cross_immunity_data, tabs):
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
    cross_list = []
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
        # make cross matrix
        variants_count = len(introduced_variants_set)
        immunity = np.ones((variants_count, variants_count), dtype=float) 
        for j_v in range(variants_count):
            for (i_v, variant_dict_raw) in enumerate(introduced_variants_set):
                before_ind = variant_dict_raw['variant_name'].find(' of')
                # find value in cross_immunity_data
                variant_name_ = variant_dict_raw['variant_name'][:before_ind]
                for cross_row in cross_immunity_data:
                    if variant_name_ in cross_row:
                        immunity[j_v, i_v] = float(cross_row[variant_name_])
                        
        cross_list.append(immunity)

        variants_list.append(variants)
    return variants_list, cross_list


def get_variants_and_cross(introduced_variants_list, cross_immunity_data, is_multi, tabs):
    if is_multi:
        return get_variant_and_cross_for_multi(introduced_variants_list, cross_immunity_data, tabs)
    return get_variant_and_cross_for_not_multi(introduced_variants_list, cross_immunity_data)


class NeoSimulator:

    class MultiParametersSim:
        predefined_pops = ['100K', '100K(Random)', '500K', '1M', '3M']

        def __init__(self, params_dict):
            # sim parameters
            self.parss = None
            self.datafiles = None 
            self.analyzers = None 
            self.virus_parameterss = None 
            self.variantss = None 
            self.popfiles = None 
            self.labels = None
            # help parameters
            self.n_cities = None
            self.is_predefined_pop_s = None
            # common(for sims) parameters
            self.adjacency_matrix = None
            self.mulitple_cities = None
            self.show_all = None
            self.save_people_history = None
            # Initialize all
            self.initialize(params_dict)

        def initialize(self, params_dict):
            tabs = params_dict['tabs']
            # make n_cities
            self.n_cities = len(tabs)
            # make is_predefined_pop_s
            self.is_predefined_pop_s = self.get_is_predefined_pop_s(params_dict['population_volume_list'])
            # make datafiles
            self.datafiles = self.get_datafiles(params_dict['datafile'], self.n_cities)
            # make analyzers
            self.analyzers = self.get_analyzers(self.n_cities, params_dict['show_all'], params_dict['save_people_history'])
            # make params_dict['show_all']
            self.show_all = params_dict['show_all']
            # make save_people_history 
            self.save_people_history = params_dict['save_people_history']
            # make virus_parameterss
            self.virus_parameterss = self.get_virus_parameterss(params_dict['virus_name_list'])
            # make variantss
            self.variantss, cross_list = self.get_variantss(
                params_dict['introduced_variants_list'], 
                params_dict['cross_immunity_data'], 
                params_dict['multiple_cities'], 
                tabs)
            # make popfiles
            self.popfiles = self.get_popfiles(params_dict['population_volume_list'])
            # make labels
            self.labels = self.get_labels(tabs, params_dict['population_volume_list'])
            # make adjacency_matrix
            self.adjacency_matrix = parse_interaction_records(params_dict['interaction_records'], tabs) if params_dict['multiple_cities'] else None
            # make mulitple_cities
            self.multiple_cities = params_dict['multiple_cities']
            # make parss
            self.parss = self.get_parss(params_dict, cross_list, self.variantss)

        def get_parss(self, params_dict, cross_list, variant_list):
            web_pars_list = []
            for (sim_pars, epi_pars, int_pars, infection_step, rel_sus_type, 
                rel_trans_type, population_volume, month_choice, monthly_weather_file, city_ind) in \
                zip(params_dict['sim_pars_list'], params_dict['epi_pars_list'], params_dict['int_pars_list'], 
                    params_dict['infection_step_list'], params_dict['rel_sus_type_list'], 
                    params_dict['rel_trans_type_list'], params_dict['population_volume_list'], 
                    params_dict['month_choice_list'], params_dict['monthly_weather_file_list'], params_dict['tabs']):            
                web_pars = parse_parameters(
                    sim_pars=sim_pars, epi_pars=epi_pars, int_pars=int_pars, 
                    n_days=params_dict['n_days'], verbose=params_dict['verbose'], die=params_dict['die'], 
                    infection_step=infection_step, rel_sus_type=rel_sus_type, rel_trans_type=rel_trans_type,
                    infectiousTableConfig=params_dict['infectiousTableConfig'], population_volume=population_volume, 
                    month_choice=month_choice, monthly_weather_file=monthly_weather_file, city_ind=city_ind)
                web_pars_list.append(web_pars)
            
            parss = []
            for pars, population_volume, variants, cross_matrix in \
                zip(web_pars_list, params_dict['population_volume_list'], variant_list, cross_list):
                #
                new_pop_size = parse_population_size(population_volume)
                pars['pop_size'] = new_pop_size
                pars['pop_type'] = 'synthpops' if population_volume != "100K(Random)" else 'random'
                pars['n_variants'] = len(variants)
                pars['immunity'] = cross_matrix
                pars['pop_infected'] = 0
                parss.append(pars)
            return parss


        def get_datafiles(self, datafile, n_cities):
            datafiles = []
            for _ in range(n_cities):
                datafiles.append(datafile)
            return datafiles


        def get_variantss(self, introduced_variants_list, cross_immunity_data, multiple_cities, tabs):
            variants_list, cross_list = get_variants_and_cross(introduced_variants_list, cross_immunity_data, multiple_cities, tabs)
            return variants_list, cross_list

        def get_virus_parameterss(self, virus_name_list):
            res = []
            for virus_name in virus_name_list:
                if virus_name in virus2filename:
                    virus_pars = VirusParameters.load(virus2filename[virus_name])
                else:
                    virus_pars = VirusParameters()
                res.append(virus_pars)
            return res

        def get_popfiles(self, population_volume_list):
            popfiles = []
            for population_volume in population_volume_list:
                popfiles.append(f"synthpops_files/synth_pop_{population_volume}.ppl")
            return popfiles

        def get_is_predefined_pop_s(self, population_volume_list):
            res = []
            for population_volume in population_volume_list:
                is_predefined_pop = population_volume not in self.predefined_pops
                res.append(is_predefined_pop)
            return res


        def get_analyzers(self, n_cities, show_all, save_people_history):
            analyzers = []
            for _ in range(n_cities):
                analyzers.append(store_seir(show_all=show_all, save_people_history=save_people_history, label='seir'))
            return analyzers

        def get_labels(self, tabs, population_volume_list):
            labels = []
            for i in range(len(tabs)):
                city_ind = tabs[i]
                population_volume = population_volume_list[i]
                if population_volume not in self.predefined_pops:
                    labels.append(population_volume)
                else:
                    labels.append(f"City {city_ind}")
            return labels

        
        def get_city_parameters(self, i):
            pars = self.parss[i]
            basic_res = dict(
                pars=pars, datafile=self.datafiles[i], analyzers=self.analyzers[i], 
                virus_parameters=self.virus_parameterss[i], variants=self.variantss[i], 
                popfile=self.popfiles[i], label=self.labels[i])
            if pars['pop_type'] == 'random':
                basic_res["contacts"] = dict(a=35)
                basic_res["beta_layer"] = dict(a=pars['beta_layer']['c'])
            return basic_res


    def __init__(self, **params_dict):
        self.all_params = None
        self.sims = None
        self.msim_with = None
        self.graphs = None
        self.output = None
        self.initialize(params_dict)


    def initialize(self, params_dict):
        self._copy_pars_for_result(
            params_dict['sim_pars_list'], 
            params_dict['epi_pars_list'], 
            params_dict['int_pars_list']
        )
        self.create_simulation(params_dict)


    def _copy_pars_for_result(self, sim_pars_list, epi_pars_list, int_pars_list):
        self.output = {}
        self.output['sim_pars'] = copy.deepcopy(sim_pars_list)
        self.output['epi_pars'] = copy.deepcopy(epi_pars_list)
        self.output['int_pars'] = copy.deepcopy(int_pars_list)

    def construct_sims(self, parameters_sim):
        sims = []
        for i in range(parameters_sim.n_cities):
            city_parameters = parameters_sim.get_city_parameters(i)
            if parameters_sim.is_predefined_pop_s[i]:
                sims.append(city_parameters)
            else:
                sims.append(cv.Sim(
                    pars=city_parameters['pars'], datafile=city_parameters['datafile'], analyzers=city_parameters['analyzers'], 
                    virus_parameters=city_parameters['virus_parameters'], variants=city_parameters['variants'], 
                    popfile=city_parameters['popfile'], label=city_parameters['label']))

        return sims

    def create_sims(self, parameters_sims):
        sims = self.construct_sims(parameters_sims)
        self.sims = build_parallel_cities(sims)

    def create_simulation(self, params_dict):
        # separate by tabs every city
        separated_parameters = separate_by_tabs(params_dict)
        # pass it in order parsing and treat all
        self.all_params = self.MultiParametersSim(separated_parameters)
        # Create the sim and update the parameters
        self.create_sims(self.all_params)

    def _run_impl(self):
        self.msim_with = cv.MultiSim(sims=self.sims)
        init_log_files(self.msim_with.sims)
        self.msim_with.run(
            n_cpus=self.all_params.n_cities, 
            mulitple_cities=self.all_params.multiple_cities, 
            adjacency_matrix=self.all_params.adjacency_matrix, 
            keep_people=True, verbose=False)
    
    def _plot_impl(self):
        n_cities = self.all_params.n_cities
        is_several_cities = n_cities > 1
        n_cpus = n_cities + is_several_cities
        show_all = self.all_params.show_all
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as pool:
            results_graph = pool.map(
                plot_all_graphs, self.msim_with.sims, 
                repeat(show_all, n_cities))

        self.graphs = {}
        for graph_group in graph_groups:
            self.graphs[graph_group] = {}
        for (city_ind, graph) in enumerate(results_graph):
            for (k, v) in graph.items():
                self.graphs[k][city_ind] = v
        # comparing graph
        if is_several_cities:
            comparing_graph = plot_comparing(self.msim_with.sims, show_all)
            for (k, v) in comparing_graph.items():
                self.graphs[k]['comparing'] = v
            self.graphs[Rest].pop('comparing')
            self.graphs[Spread_parameters_by_age].pop('comparing')
        # write result graph to output
        self.output['graphs'] = self.graphs

    def _result_prepare_impl(self):
        files_all, summary_all = get_output_files(self.msim_with.sims, self.all_params.save_people_history)
        self.output['files_all']    = files_all
        self.output['summary_all']  = summary_all


def get_error_obj(message, exception):
    return dict(errs=[dict(
            message=message, 
            exception=str(exception))])


@app.register_RPC()
def create_simulation(sim_pars_list=None, epi_pars_list=None, int_pars_list=None, datafile=None, multiple_cities=False, show_all=False, save_people_history=False, n_days=None, infection_step_list=None, rel_sus_type_list=None, rel_trans_type_list=None, population_volume_list=None, infectiousTableConfig=None, introduced_variants_list=None, tabs=None, cross_immunity_data=None, interaction_records=None, virus_name_list=None, month_choice_list=None, monthly_weather_file_list=None, verbose=True, die=die):
    global neo_simulator
    try:
        neo_simulator = NeoSimulator(
            sim_pars_list=sim_pars_list, 
            epi_pars_list=epi_pars_list, 
            int_pars_list=int_pars_list, 
            datafile=datafile, 
            multiple_cities=multiple_cities, 
            show_all=show_all, 
            save_people_history=save_people_history,
            n_days=n_days,
            infection_step_list=infection_step_list, 
            rel_sus_type_list=rel_sus_type_list, 
            rel_trans_type_list=rel_trans_type_list,
            population_volume_list=population_volume_list, 
            infectiousTableConfig=infectiousTableConfig,
            introduced_variants_list=introduced_variants_list, 
            tabs=tabs, 
            cross_immunity_data=cross_immunity_data, 
            interaction_records=interaction_records, 
            virus_name_list=virus_name_list, 
            month_choice_list=month_choice_list, 
            monthly_weather_file_list=monthly_weather_file_list, 
            verbose=verbose, 
            die=die
        )
        return "success"
    except Exception as E:
        raise E
        return get_error_obj("Create simulation error", E)

@app.register_RPC()
def run_simulation():
    global neo_simulator
    try:
        neo_simulator._run_impl()
        return "success"
    except TimeoutError as TE:
        return get_error_obj("Timeout error", TE)
    except Exception as E:
        raise E
        return get_error_obj("Run simulation error", E)

@app.register_RPC()
def plot_simulation():
    global neo_simulator
    try:
        neo_simulator._plot_impl()
        return "success"
    except Exception as E:
        return get_error_obj("Plotting simulation error", E)


@app.register_RPC()
def results_prepare_simulation():
    global neo_simulator
    try:
        neo_simulator._result_prepare_impl()
        return neo_simulator.output
    except Exception as E:
        return get_error_obj("Result prepared error", E)


def get_output_files_impl(sim, save_people_history):
    ''' Create output files for download '''

    datestamp = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')
    del sim.pars['starting_month']
    del sim.pars['multipleir_humidity_coef']
    del sim.pars['monthly_weather']

    ss = sim.to_excel()

    files = {}
    files['xlsx'] = {
        'filename': f'covasim_results_{sim.label}_{datestamp}.xlsx',
        'content': 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + base64.b64encode(ss.blob).decode("utf-8"),
    }

    json_string = sim.to_json(tostring=True, verbose=False)
    files['json'] = {
        'filename': f'covasim_results_{sim.label}_{datestamp}.json',
        'content': 'data:application/text;base64,' + base64.b64encode(json_string.encode()).decode("utf-8"),
    }

    if save_people_history:
        history_people = sim.get_analyzer('seir').get_excel_people_history()
        files['xlsx_people_history'] = {
            'filename': f'covasim_results_{sim.label}_{datestamp}_people_history.xlsx',
            'content': 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + base64.b64encode(history_people.blob).decode("utf-8"),
        }

    # Summary output
    summary = {
        'days': sim.npts-1,
        'cases': round(sim.results['cum_infections'][-1]),
        'deaths': round(sim.results['cum_deaths'][-1]),
    }
    return files, summary


def get_output_files(sims, save_people_history):
    files_all = []
    summary_all = []
    for sim in sims:
        files_sim, summary_sim = get_output_files_impl(sim, save_people_history)
        files_all.append(files_sim)
        summary_all.append(summary_sim)
    return files_all, summary_all


#%% Run the server using Flask

if __name__ == "__main__":

    os.chdir(sc.thisdir(__file__))

    if len(sys.argv) > 1:
        app.config['SERVER_PORT'] = int(sys.argv[1])
    else:
        app.config['SERVER_PORT'] = 8239
    if len(sys.argv) > 2:
        autoreload = int(sys.argv[2])
    else:
        autoreload = 1

    app.run(autoreload=autoreload)
