'''
Set the parameters for Covasim.
'''

import numpy as np
import pandas as pd
import math
import sciris as sc
from .settings import options as cvo # For setting global options
from . import misc as cvm
from . import defaults as cvd

__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses', 'get_variant_choices', 'get_vaccine_choices',
           'get_variant_pars', 'get_cross_immunity', 'get_vaccine_variant_pars', 'get_vaccine_dose_pars']


def _check_prognoses(prognoses):
    # Check that lengths match
    expected_len = len(prognoses['age_cutoffs'])
    for key,val in prognoses.items():
        this_len = len(prognoses[key])
        if this_len != expected_len: # pragma: no cover
            errormsg = f'Lengths mismatch in prognoses: {expected_len} age bins specified, but key "{key}" has {this_len} entries'
            raise ValueError(errormsg)


def get_humidity(filename):
    df = pd.read_excel(filename, header=None)
    virus_name_ind = df[df[0] == 'Humidity table'].index[0]
    return np.array(list(map(float, df.iloc[virus_name_ind + 2].values[1:])))

class VirusParameters:
    def __init__(self, 
        nab_init        = dict(dist='normal', par1=0, par2=2),  # Parameters for the distribution of the initial level of log2(nab) following natural infection, taken from fig1b of https://doi.org/10.1101/2021.03.09.21252641
        nab_decay       = dict(form='nab_growth_decay', growth_time=21, decay_rate1=np.log(2) / 50, decay_time1=150, decay_rate2=np.log(2) / 250, decay_time2=365),
        trans_redux     = 0.59,  # Reduction in transmission for breakthrough infections, https://www.medrxiv.org/content/10.1101/2021.07.13.21260393v
        nab_eff         = dict(alpha_inf=1.08, alpha_inf_diff=1.812, beta_inf=0.967, alpha_symp_inf=-0.739, beta_symp_inf=0.038, alpha_sev_symp=-0.014, beta_sev_symp=0.079), # Parameters to map nabs to efficacy
        beta_dist       = dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01), # Distribution to draw individual level transmissibility; dispersion from https://www.researchsquare.com/article/rs-29548/v1     
        rel_sus_type = 'constants',
        multipleir_humidity_coef = np.array([1.8, 1.6, 1.44, 1.22, 1.11, 1.11, 1.22, 1.44, 1.6, 1.8]),
        prognoses = dict(
            age_cutoffs   = np.array([0,       10,      20,      30,      40,      50,      60,      70,      80,      90,]),     # Age cutoffs (lower limits)
            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from Zhang et al., https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
            trans_ORs     = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Odds ratios for relative transmissibility -- no evidence of differences
            comorbidities = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Comorbidities by age -- set to 1 by default since already included in disease progression rates
            symp_probs    = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.90]),    # Overall probability of developing symptoms (based on https://www.medrxiv.org/content/10.1101/2020.03.24.20043018v1.full.pdf, scaled for overall symptomaticity)
            severe_probs  = np.array([0.00050, 0.00165, 0.00720, 0.02080, 0.03430, 0.07650, 0.13280, 0.20655, 0.24570, 0.24570]), # Overall probability of developing severe symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420, 0.17420]), # Overall probability of developing critical symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            death_probs   = np.array([0.00002, 0.00002, 0.00010, 0.00032, 0.00098, 0.00265, 0.00766, 0.02439, 0.08292, 0.16190]), # Overall probability of dying -- from O'Driscoll et al., https://www.nature.com/articles/s41586-020-2918-0; last data point from Brazeau et al., https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-34-ifr/
        )    
    ):
        self.nab_init = nab_init 
        self.nab_decay = nab_decay 
        self.trans_redux = trans_redux 
        self.nab_eff = nab_eff 
        self.beta_dist = beta_dist 
        self.rel_sus_type = rel_sus_type
        self.multipleir_humidity_coef = multipleir_humidity_coef
        # set prognoses
        self.prognoses = prognoses 
        self.prognoses = relative_prognoses(prognoses) # Convert to conditional probabilities
        _check_prognoses(prognoses)

    def load(filename):
        df = pd.read_excel(filename, header=None)
        # set main parameters
        # set nab_init searching by key
        nab_init_str = df[df[0] == 'Initial NABs after infection distribution (in neutralization titre relative to convalescent plasma)'][1].values[0]
        # parse string like {dist_name=normal, mean=0, std=2} to dict
        def parse_str_to_dict(s):
            s = s.replace('{', '').replace('}', '')
            tmp_res = dict([tuple(x.split('=')) for x in s.split(',')])
            # remove start and end spaces
            res = dict()
            for k, v in tmp_res.items():
                res[k.strip()] = v.strip()
            return res
        
        def parse_method_str_to_dict(s):
            # parse string like {method=nab_growth_decay   parameters: growth_time=21, decay_rate1=log(2) / 50, decay_time1=150, decay_rate2=log(2) / 250, decay_time2=365} to dict
            s = s.replace('{', '').replace('}', '')
            # split by method and parameters
            method, params = s.split('parameters:')
            # parse method
            method = method.split('=')[1].strip()[:-1]
            # parse parameters
            params = params.split(',')
            params = [x.split('=') for x in params]
            params = [(x[0].strip(), x[1].strip()) for x in params]
            params = dict(params)
            return dict(method=method, params=params)
        def str_to_float(s):
            # convert string like log(2) / 50 to float
            s = s.replace('log', 'math.log')
            s = s.replace('exp', 'math.exp')
            s = s.replace('^', '**')
            result = eval(s)
            return result


        nab_init_dict = parse_str_to_dict(nab_init_str)
        nab_init = dict()
        nab_init['dist'] = nab_init_dict['dist_name']
        nab_init['par1'] = str_to_float(nab_init_dict['mean'])
        nab_init['par2'] = str_to_float(nab_init_dict['std'])
        # set nab_decay
        nab_decay_str = df[df[0] == 'Decay NABs function'][1].values[0]
        nab_decay_dict = parse_method_str_to_dict(nab_decay_str)
        nab_decay = dict()
        nab_decay['form'] = nab_decay_dict['method']
        nab_decay['growth_time'] = str_to_float(nab_decay_dict['params']['growth_time'])
        nab_decay['decay_rate1'] = str_to_float(nab_decay_dict['params']['decay_rate1'])
        nab_decay['decay_time1'] = str_to_float(nab_decay_dict['params']['decay_time1'])
        nab_decay['decay_rate2'] = str_to_float(nab_decay_dict['params']['decay_rate2'])
        nab_decay['decay_time2'] = str_to_float(nab_decay_dict['params']['decay_time2'])
        # set rel_sus_type
        #rel_sus_type = str_to_float(df[df[0] == 'Reduction in transmission for breakthrough infections'][1].values[0])
        # set trans_redux
        trans_redux = str_to_float(df[df[0] == 'Reduction in transmission for breakthrough infections'][1].values[0])
        # set nab_eff
        nab_eff_succ_str = df[df[0] == 'Function that determine how many NABs person need to have to be susceptible'][1].values[0]
        nab_eff_succ_dict = parse_method_str_to_dict(nab_eff_succ_str)
        nab_eff_symp_str = df[df[0] == 'Function that determine how many NABs person need to have to be symptomatic'][1].values[0]
        nab_eff_symp_dict = parse_method_str_to_dict(nab_eff_symp_str)
        nab_eff_sev_str = df[df[0] == 'Function that determine how many NABs person need to have to be severe'][1].values[0]
        nab_eff_sev_dict = parse_method_str_to_dict(nab_eff_sev_str)
        nab_eff = dict()
        nab_eff['alpha_inf_diff'] = 1.812 # TODO(IvanKozlov98) get know what does mean this parameter
        nab_eff['alpha_inf'] = str_to_float(nab_eff_succ_dict['params']['a'])
        nab_eff['beta_inf'] = str_to_float(nab_eff_succ_dict['params']['b'])
        nab_eff['alpha_symp_inf'] = str_to_float(nab_eff_symp_dict['params']['a'])
        nab_eff['beta_symp_inf'] = str_to_float(nab_eff_symp_dict['params']['b'])
        nab_eff['alpha_sev_symp'] = str_to_float(nab_eff_sev_dict['params']['a'])
        nab_eff['beta_sev_symp'] = str_to_float(nab_eff_sev_dict['params']['b'])
        # set beta_dist
        beta_dist_str = df[df[0] == 'Initial transmisibility distribution'][1].values[0]
        beta_dist_dict = parse_str_to_dict(beta_dist_str)
        beta_dist = dict()
        beta_dist['dist'] = beta_dist_dict['dist_name']
        beta_dist['par1'] = str_to_float(beta_dist_dict['mean'])
        beta_dist['par2'] = str_to_float(beta_dist_dict['std'])
        beta_dist['step'] = 0.01 # TODO (IvanKozlov98)
        # set prognoses
        prognoses = dict()
        # convert list of strings like ['0-9' '10-19' '20-29' '30-39' '40-49' '50-59' '60-69' '70-79' '80-89' '90+'] to numpy like [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        age_cutoffs_strs = df[df[0] == 'Age cutoffs'].values[0][1:]
        age_cutoffs = [int(x.split('-')[0]) for x in age_cutoffs_strs[:-1]] + [age_cutoffs_strs[-1].replace('+', '')]
        prognoses['age_cutoffs'] = np.array(age_cutoffs, dtype=float)
        prognoses['symp_probs'] = np.array(df[df[0] == 'Probability of being symptomatic'].values[0][1:], dtype=float)
        prognoses['severe_probs'] = np.array(df[df[0] == 'Probability of being severe'].values[0][1:], dtype=float)
        prognoses['sus_ORs'] = np.array(df[df[0] == 'Susceptibility by age'].values[0][1:], dtype=float)
        prognoses['trans_ORs'] = np.array(df[df[0] == 'Transmisibility by age'].values[0][1:], dtype=float)
        prognoses['crit_probs'] = np.array(df[df[0] == 'Probability of being critical'].values[0][1:], dtype=float)
        prognoses['death_probs'] = np.array(df[df[0] == 'Probability of being death'].values[0][1:], dtype=float)
        prognoses['comorbidities'] = np.array(df[df[0] == 'Comorbidities by age'].values[0][1:], dtype=float)
        prognoses = prognoses

        multipleir_humidity_coef = get_humidity(filename)

        virus_pars = VirusParameters(
            nab_init=nab_init,
            nab_decay=nab_decay,
            trans_redux=trans_redux,
            nab_eff=nab_eff,
            beta_dist=beta_dist,
            rel_sus_type='constants',
            multipleir_humidity_coef=multipleir_humidity_coef,
            prognoses=prognoses
        )
        return virus_pars


def make_pars(set_prognoses=False, prog_by_age=True, version=None, **kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = cv.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        set_prognoses (bool): whether or not to create prognoses (else, added when the population is created)
        prog_by_age   (bool): whether or not to use age-based severity, mortality etc.
        kwargs        (dict): any additional kwargs are interpreted as parameter names
        version       (str):  if supplied, use parameters from this Covasim version

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Weather parameters
    pars['monthly_weather'] = {
        'Outside Temperature (°C)': [-6.9, -6.1, -0.9, 6.9, 13.2, 17.2, 20.0, 17.9, 11.8, 5.8, -0.1, -4.3], 
        'Outside Humidity (%)': [0.86, 0.82, 0.74, 0.66, 0.65, 0.68, 0.72, 0.74, 0.8, 0.82, 0.85, 0.87], 
        'Inside Temperature (°C)': [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
    }
    pars['starting_month'] = None
    pars['multipleir_humidity_coef'] = np.array([1.8, 1.6, 1.44, 1.22, 1.11, 1.11, 1.22, 1.44, 1.6, 1.8])

    # Population parameters
    pars['pop_size']     = 20e3     # Number of agents, i.e., people susceptible to SARS-CoV-2
    pars['pop_infected'] = 20       # Number of initial infections
    pars['pop_type']     = 'random' # What type of population data to use -- 'random' (fastest), 'synthpops' (best), 'hybrid' (compromise)
    pars['location']     = None     # What location to load data from -- default Seattle

    # Simulation parameters
    pars['start_day']  = '2020-03-01' # Start day of the simulation
    pars['end_day']    = None         # End day of the simulation
    pars['n_days']     = 60           # Number of days to run, if end_day isn't specified
    pars['rand_seed']  = 1            # Random seed, if None, don't reset
    pars['verbose']    = cvo.verbose  # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

    # Rescaling parameters
    pars['pop_scale']         = 1    # Factor by which to scale the population -- e.g. pop_scale=10 with pop_size=100e3 means a population of 1 million
    pars['scaled_pop']        = None # The total scaled population, i.e. the number of agents times the scale factor
    pars['rescale']           = True # Enable dynamic rescaling of the population -- starts with pop_scale=1 and scales up dynamically as the epidemic grows
    pars['rescale_threshold'] = 0.05 # Fraction susceptible population that will trigger rescaling if rescaling
    pars['rescale_factor']    = 1.2  # Factor by which the population is rescaled on each step
    pars['frac_susceptible']  = 1.0  # What proportion of the population is susceptible to infection

    # Network parameters, generally initialized after the population has been constructed
    pars['contacts']        = None  # The number of contacts per layer; set by reset_layer_pars() below
    pars['dynam_layer']     = None  # Which layers are dynamic; set by reset_layer_pars() below
    pars['beta_layer']      = None  # Transmissibility per layer; set by reset_layer_pars() below

    pars['tourist_contact_count'] = 4
    # Basic disease transmission parameters
    pars['viral_dist']   = dict(frac_time=0.3, load_ratio=2, high_cap=4) # The time varying viral load (transmissibility); estimated from Lescure 2020, Lancet, https://doi.org/10.1016/S1473-3099(20)30200-0
    pars['beta']         = 0.016  # Beta per symptomatic contact; absolute value, calibrated
    pars['asymp_factor'] = 1.0  # Multiply beta by this factor for asymptomatic cases; no statistically significant difference in transmissibility: https://www.sciencedirect.com/science/article/pii/S1201971220302502

    # Parameters that control settings and defaults for multi-variant runs
    pars['n_imports']  = 0 # Average daily number of imported cases (actual number is drawn from Poisson distribution)
    pars['n_variants'] = 1 # The number of variants circulating in the population

    pars['rel_trans_type'] = 'beta_dist'

    # Parameters used to calculate immunity
    pars['use_waning']   = True # Whether to use dynamically calculated immunity
    pars['nab_kin']      = None # Constructed during sim initialization using the nab_decay parameters
    pars['nab_boost']    = 1.5 # Multiplicative factor applied to a person's nab levels if they get reinfected. No data on this, assumption.
    pars['rel_imm_symp'] = dict(asymp=0.85, mild=1, severe=1.5) # Relative immunity from natural infection varies by symptoms. Assumption.
    pars['immunity']     = None  # Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py

    # Variant-specific disease transmission parameters. By default, these are set up for a single variant, but can all be modified for multiple variants
    pars['rel_beta']        = 1.0 # Relative transmissibility varies by variant

    # Duration parameters: time for disease progression
    pars['dur'] = {}
    pars['dur_exp2inf']  = dict(dist='lognormal_int', par1=4.5, par2=1.5) # Duration from exposed to infectious; see Lauer et al., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/, appendix table S2, subtracting inf2sym duration
    pars['dur_inf2sym']  = dict(dist='lognormal_int', par1=1.1, par2=0.9) # Duration from infectious to symptomatic; see Linton et al., https://doi.org/10.3390/jcm9020538, from Table 2, 5.6 day incubation period - 4.5 day exp2inf from Lauer et al.
    pars['dur_sym2sev']  = dict(dist='lognormal_int', par1=6.6, par2=4.9) # Duration from symptomatic to severe symptoms; see Linton et al., https://doi.org/10.3390/jcm9020538, from Table 2, 6.6 day onset to hospital admission (deceased); see also Wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044, 7 days (Table 1)
    pars['dur_sev2crit'] = dict(dist='lognormal_int', par1=1.5, par2=2.0) # Duration from severe symptoms to requiring ICU; average of 1.9 and 1.0; see Chen et al., https://www.sciencedirect.com/science/article/pii/S0163445320301195, 8.5 days total - 6.6 days sym2sev = 1.9 days; see also Wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044, Table 3, 1 day, IQR 0-3 days; std=2.0 is an estimate

    # Duration parameters: time for disease recovery
    pars['dur_asym2rec'] = dict(dist='lognormal_int', par1=8.0,  par2=2.0) # Duration for asymptomatic people to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['dur_mild2rec'] = dict(dist='lognormal_int', par1=8.0,  par2=2.0) # Duration for people with mild symptoms to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['dur_sev2rec']  = dict(dist='lognormal_int', par1=18.1, par2=6.3) # Duration for people with severe symptoms to recover, 24.7 days total; see Verity et al., https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext; 18.1 days = 24.7 onset-to-recovery - 6.6 sym2sev; 6.3 = 0.35 coefficient of variation * 18.1; see also https://doi.org/10.1017/S0950268820001259 (22 days) and https://doi.org/10.3390/ijerph17207560 (3-10 days)
    pars['dur_crit2rec'] = dict(dist='lognormal_int', par1=18.1, par2=6.3) # Duration for people with critical symptoms to recover; as above (Verity et al.)
    pars['dur_crit2die'] = dict(dist='lognormal_int', par1=10.7, par2=4.8) # Duration from critical symptoms to death, 18.8 days total; see Verity et al., https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext; 10.7 = 18.8 onset-to-death - 6.6 sym2sev - 1.5 sev2crit; 4.8 = 0.45 coefficient of variation * 10.7

    # Severity parameters: probabilities of symptom progression
    pars['rel_symp_prob']   = 1.0  # Scale factor for proportion of symptomatic cases
    pars['rel_severe_prob'] = 1.0  # Scale factor for proportion of symptomatic cases that become severe
    pars['rel_crit_prob']   = 1.0  # Scale factor for proportion of severe cases that become critical
    pars['rel_death_prob']  = 1.0  # Scale factor for proportion of critical cases that result in death
    pars['prog_by_age']     = prog_by_age # Whether to set disease progression based on the person's age

    # Efficacy of protection measures
    pars['iso_factor']   = None # Multiply beta by this factor for diagnosed cases to represent isolation; set by reset_layer_pars() below
    pars['quar_factor']  = None # Quarantine multiplier on transmissibility and susceptibility; set by reset_layer_pars() below
    pars['quar_period']  = 14   # Number of days to quarantine for; assumption based on standard policies

    # Events and interventions
    pars['interventions'] = []   # The interventions present in this simulation; populated by the user
    pars['analyzers']     = []   # Custom analysis functions; populated by the user
    pars['timelimit']     = None # Time limit for the simulation (seconds)
    pars['stopping_func'] = None # A function to call to stop the sim partway through

    # Health system parameters
    pars['n_beds_hosp']    = None # The number of hospital (adult acute care) beds available for severely ill patients (default is no constraint)
    pars['n_beds_icu']     = None # The number of ICU beds available for critically ill patients (default is no constraint)
    pars['no_hosp_factor'] = 2.0  # Multiplier for how much more likely severely ill people are to become critical if no hospital beds are available
    pars['no_icu_factor']  = 2.0  # Multiplier for how much more likely critically ill people are to die if no ICU beds are available

    pars['is_additive_formula'] = False

    # Handle vaccine and variant parameters
    pars['vaccine_pars'] = {} # Vaccines that are being used; populated during initialization
    pars['vaccine_map']  = {} #Reverse mapping from number to vaccine key
    pars['variants']     = [] # Additional variants of the virus; populated by the user, see immunity.py
    pars['variant_map']  = {} # Reverse mapping from number to variant key
    pars['variant_pars'] = dict() # Populated just below
    #for sp in cvd.variant_pars:
    #    if sp in pars.keys():
    #        pars['variant_pars']['wild'][sp] = pars[sp]

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)

    pars['nab_init']     = dict(dist='normal', par1=0, par2=2)  # Parameters for the distribution of the initial level of log2(nab) following natural infection, taken from fig1b of https://doi.org/10.1101/2021.03.09.21252641
    pars['nab_decay']    = dict(form='nab_growth_decay', growth_time=21, decay_rate1=np.log(2) / 50, decay_time1=150, decay_rate2=np.log(2) / 250, decay_time2=365)
    pars['trans_redux']  = 0.59  # Reduction in transmission for breakthrough infections, https://www.medrxiv.org/content/10.1101/2021.07.13.21260393v
    pars['nab_eff']      = dict(alpha_inf=1.08, alpha_inf_diff=1.812, beta_inf=0.967, alpha_symp_inf=-0.739, beta_symp_inf=0.038, alpha_sev_symp=-0.014, beta_sev_symp=0.079) # Parameters to map nabs to efficacy
    pars['beta_dist']    = dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01) # Distribution to draw individual level transmissibility; dispersion from https://www.researchsquare.com/article/rs-29548/v1
    pars['rel_sus_type'] = 'constants'
    pars['prognoses']       = None # The actual arrays of prognoses by age; this is populated later
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses(pars['prog_by_age'], version=version) # Default to age-specific prognoses



    # If version is specified, load old parameters
    if version is not None:
        version_pars = cvm.get_version_pars(version, verbose=pars['verbose'])
        if sc.compareversions(version, '<3.0.0'): # Waning was introduced in 3.0, so is always false before
            version_pars['use_waning'] = False
        for key in pars.keys(): # Only loop over keys that have been populated
            if key in version_pars: # Only replace keys that exist in the old version
                pars[key] = version_pars[key]

        # Handle code change migration
        if sc.compareversions(version, '<2.1.0') and 'migrate_lognormal' not in pars:
            cvm.migrate_lognormal(pars, verbose=pars['verbose'])

    return pars


# Define which parameters need to be specified as a dictionary by layer -- define here so it's available at the module level for sim.py
layer_pars = ['beta_layer', 'contacts', 'dynam_layer', 'iso_factor', 'quar_factor']


def reset_layer_pars(pars, layer_keys=None, force=False):
    '''
    Helper function to set layer-specific parameters. If layer keys are not provided,
    then set them based on the population type. This function is not usually called
    directly by the user, although it can sometimes be used to fix layer key mismatches
    (i.e. if the contact layers in the population do not match the parameters). More
    commonly, however, mismatches need to be fixed explicitly.

    Args:
        pars (dict): the parameters dictionary
        layer_keys (list): the layer keys of the population, if available
        force (bool): reset the parameters even if they already exist
    '''

    # Specify defaults for random -- layer 'a' for 'all'
    layer_defaults = {}
    layer_defaults['random'] = dict(
        beta_layer  = dict(a=1.0), # Default beta
        contacts    = dict(a=20),  # Default number of contacts
        dynam_layer = dict(a=0),   # Do not use dynamic layers by default
        iso_factor  = dict(a=0.2), # Assumed isolation factor
        quar_factor = dict(a=0.3), # Assumed quarantine factor
    )

    # Specify defaults for hybrid -- household, school, work, and community layers (h, s, w, c)
    layer_defaults['hybrid'] = dict(
        beta_layer  = dict(h=3.0, s=0.6, w=0.6, c=0.3),  # Per-population beta weights; relative; in part based on Table S14 of https://science.sciencemag.org/content/sci/suppl/2020/04/28/science.abb8001.DC1/abb8001_Zhang_SM.pdf
        contacts    = dict(h=2.0, s=20,  w=16,  c=20),   # Number of contacts per person per day, estimated
        dynam_layer = dict(h=0,   s=0,   w=0,   c=1),    # Which layers are dynamic -- none by default
        iso_factor  = dict(h=0.3, s=0.1, w=0.1, c=0.1),  # Multiply beta by this factor for people in isolation
        quar_factor = dict(h=0.6, s=0.2, w=0.2, c=0.2),  # Multiply beta by this factor for people in quarantine
    )

    # Specify defaults for SynthPops -- same as hybrid except for LTCF layer (l)
    l_pars = dict(beta_layer=1.5, contacts=10, dynam_layer=0, iso_factor=0.2, quar_factor=0.3)
    layer_defaults['synthpops'] = sc.dcp(layer_defaults['hybrid'])
    for key,val in l_pars.items():
        layer_defaults['synthpops'][key]['l'] = val

    # Choose the parameter defaults based on the population type, and get the layer keys
    try:
        defaults = layer_defaults[pars['pop_type']]
    except Exception as E:
        errormsg = f'Cannot load defaults for population type "{pars["pop_type"]}": must be hybrid, random, or synthpops'
        raise ValueError(errormsg) from E
    default_layer_keys = list(defaults['beta_layer'].keys()) # All layers should be the same, but use beta_layer for convenience

    # Actually set the parameters
    for pkey in layer_pars:
        par = {} # Initialize this parameter
        default_val = layer_defaults['random'][pkey]['a'] # Get the default value for this parameter

        # If forcing, we overwrite any existing parameter values
        if force:
            par_dict = defaults[pkey] # Just use defaults
        else:
            par_dict = sc.mergedicts(defaults[pkey], pars.get(pkey, None)) # Use user-supplied parameters if available, else default

        # Figure out what the layer keys for this parameter are (may be different between parameters)
        if layer_keys:
            par_layer_keys = layer_keys # Use supplied layer keys
        else:
            par_layer_keys = list(sc.odict.fromkeys(default_layer_keys + list(par_dict.keys())))  # If not supplied, use the defaults, plus any extra from the par_dict; adapted from https://www.askpython.com/python/remove-duplicate-elements-from-list-python

        # Construct this parameter, layer by layer
        for lkey in par_layer_keys: # Loop over layers
            par[lkey] = par_dict.get(lkey, default_val) # Get the value for this layer if available, else use the default for random
        pars[pkey] = par # Save this parameter to the dictionary

    return


def get_prognoses(by_age=True, version=None):
    '''
    Return the default parameter values for prognoses

    The prognosis probabilities are conditional given the previous disease state.

    Args:
        by_age (bool): whether to use age-specific values (default true)

    Returns:
        prog_pars (dict): the dictionary of prognosis probabilities
    '''

    if not by_age: # All rough estimates -- almost always, prognoses by age (below) are used instead
        prognoses = dict(
            age_cutoffs   = np.array([0]),
            sus_ORs       = np.array([1.00]),
            trans_ORs     = np.array([1.00]),
            symp_probs    = np.array([0.75]),
            comorbidities = np.array([1.00]),
            severe_probs  = np.array([0.10]),
            crit_probs    = np.array([0.04]),
            death_probs   = np.array([0.01]),
        )
    else:
        prognoses = dict(
            age_cutoffs   = np.array([0,       10,      20,      30,      40,      50,      60,      70,      80,      90,]),     # Age cutoffs (lower limits)
            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from Zhang et al., https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
            trans_ORs     = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Odds ratios for relative transmissibility -- no evidence of differences
            comorbidities = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Comorbidities by age -- set to 1 by default since already included in disease progression rates
            symp_probs    = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.90]),    # Overall probability of developing symptoms (based on https://www.medrxiv.org/content/10.1101/2020.03.24.20043018v1.full.pdf, scaled for overall symptomaticity)
            severe_probs  = np.array([0.00050, 0.00165, 0.00720, 0.02080, 0.03430, 0.07650, 0.13280, 0.20655, 0.24570, 0.24570]), # Overall probability of developing severe symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420, 0.17420]), # Overall probability of developing critical symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            death_probs   = np.array([0.00002, 0.00002, 0.00010, 0.00032, 0.00098, 0.00265, 0.00766, 0.02439, 0.08292, 0.16190]), # Overall probability of dying -- from O'Driscoll et al., https://www.nature.com/articles/s41586-020-2918-0; last data point from Brazeau et al., https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-34-ifr/
        )
    prognoses = relative_prognoses(prognoses) # Convert to conditional probabilities

    # If version is specified, load old parameters
    if by_age and version is not None:
        version_prognoses = cvm.get_version_pars(version, verbose=False)['prognoses']
        for key in version_prognoses.keys(): # Only loop over keys that have been populated
            if key in version_prognoses: # Only replace keys that exist in the old version
                prognoses[key] = np.array(version_prognoses[key])

    # Check that lengths match
    _check_prognoses(prognoses)

    return prognoses


def relative_prognoses(prognoses):
    '''
    Convenience function to revert absolute prognoses into relative (conditional)
    ones. Internally, Covasim uses relative prognoses.
    '''
    out = sc.dcp(prognoses)
    out['death_probs']  /= out['crit_probs']   # Conditional probability of dying, given critical symptoms
    out['crit_probs']   /= out['severe_probs'] # Conditional probability of symptoms becoming critical, given severe
    out['severe_probs'] /= out['symp_probs']   # Conditional probability of symptoms becoming severe, given symptomatic
    return out


def absolute_prognoses(prognoses):
    '''
    Convenience function to revert relative (conditional) prognoses into absolute
    ones. Used to convert internally used relative prognoses into more readable
    absolute ones.

    **Example**::

        sim = cv.Sim()
        abs_progs = cv.parameters.absolute_prognoses(sim['prognoses'])
    '''
    out = sc.dcp(prognoses)
    out['severe_probs'] *= out['symp_probs']   # Absolute probability of severe symptoms
    out['crit_probs']   *= out['severe_probs'] # Absolute probability of critical symptoms
    out['death_probs']  *= out['crit_probs']   # Absolute probability of dying
    return out


#%% Variant, vaccine, and immunity parameters and functions

def get_variant_choices():
    '''
    Define valid pre-defined variant names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'wild':  ['wild', 'default', 'pre-existing', 'original'],
        'alpha': ['alpha', 'b117', 'uk', 'united kingdom', 'kent'],
        'beta':  ['beta', 'b1351', 'sa', 'south africa'],
        'gamma': ['gamma', 'p1', 'b11248', 'brazil'],
        'delta': ['delta', 'b16172', 'india'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def get_vaccine_choices():
    '''
    Define valid pre-defined vaccine names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'default': ['default', None],
        'pfizer':  ['pfizer', 'biontech', 'pfizer-biontech', 'pf', 'pfz', 'pz', 'bnt162b2', 'comirnaty'],
        'moderna': ['moderna', 'md', 'spikevax'],
        'novavax': ['novavax', 'nova', 'covovax', 'nvx', 'nv'],
        'az':      ['astrazeneca', 'az', 'covishield', 'oxford', 'vaxzevria'],
        'jj':      ['jnj', 'johnson & johnson', 'janssen', 'jj'],
        'sinovac': ['sinovac', 'coronavac'],
        'sinopharm': ['sinopharm']
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def _get_from_pars(pars, default=False, key=None, defaultkey='default'):
    ''' Helper function to get the right output from vaccine and variant functions '''

    # If a string was provided, interpret it as a key and swap
    if isinstance(default, str):
        key, default = default, key

    # Handle output
    if key is not None:
        try:
            return pars[key]
        except Exception as E:
            errormsg = f'Key "{key}" not found; choices are: {sc.strjoin(pars.keys())}'
            raise sc.KeyNotFoundError(errormsg) from E
    elif default:
        return pars[defaultkey]
    else:
        return pars


def get_variant_pars(default=False, variant=None):
    '''
    Define the default parameters for the different variants
    '''
    pars = dict(

        wild = dict(
            rel_beta        = 1.0, # Default values
            rel_symp_prob   = 1.0, # Default values
            rel_severe_prob = 1.0, # Default values
            rel_crit_prob   = 1.0, # Default values
            rel_death_prob  = 1.0, # Default values
            dur_exp2inf     = dict(dist='lognormal_int', par1=4.5, par2=1.5),
            dur_inf2sym     = dict(dist='lognormal_int', par1=1.1, par2=0.9),
            dur_sym2sev     = dict(dist='lognormal_int', par1=6.6, par2=4.9),
            dur_sev2crit    = dict(dist='lognormal_int', par1=1.5, par2=2.0),
            dur_asym2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_mild2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_sev2rec     = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2rec    = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2die    = dict(dist='lognormal_int', par1=10.7, par2=4.8),
            oral_microbiota_percent = 0.0,
            oral_microbiota_factor = 1.0,
        ),

        alpha = dict(
            rel_beta        = 1.67, # Midpoint of the range reported in https://science.sciencemag.org/content/372/6538/eabg3055
            rel_symp_prob   = 1.0,  # Inconclusive evidence on the likelihood of symptom development. See https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(21)00055-4/fulltext
            rel_severe_prob = 1.64, # From https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3792894, and consistent with https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348 and https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/961042/S1095_NERVTAG_update_note_on_B.1.1.7_severity_20210211.pdf
            rel_crit_prob   = 1.0,  # Various studies have found increased mortality for B117 (summary here: https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(21)00201-2/fulltext#tbl1), but not necessarily when conditioned on having developed severe disease
            rel_death_prob  = 1.0,  # See comment above
            dur_exp2inf     = dict(dist='lognormal_int', par1=4.5, par2=1.5),
            dur_inf2sym     = dict(dist='lognormal_int', par1=1.1, par2=0.9),
            dur_sym2sev     = dict(dist='lognormal_int', par1=6.6, par2=4.9),
            dur_sev2crit    = dict(dist='lognormal_int', par1=1.5, par2=2.0),
            dur_asym2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_mild2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_sev2rec     = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2rec    = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2die    = dict(dist='lognormal_int', par1=10.7, par2=4.8),
            oral_microbiota_percent = 0.0,
            oral_microbiota_factor = 1.0,
        ),

        beta = dict(
            rel_beta        = 1.0, # No increase in transmissibility; B1351's fitness advantage comes from the reduction in neutralisation
            rel_symp_prob   = 1.0,
            rel_severe_prob = 3.6, # From https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
            dur_exp2inf     = dict(dist='lognormal_int', par1=4.5, par2=1.5),
            dur_inf2sym     = dict(dist='lognormal_int', par1=1.1, par2=0.9),
            dur_sym2sev     = dict(dist='lognormal_int', par1=6.6, par2=4.9),
            dur_sev2crit    = dict(dist='lognormal_int', par1=1.5, par2=2.0),
            dur_asym2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_mild2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_sev2rec     = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2rec    = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2die    = dict(dist='lognormal_int', par1=10.7, par2=4.8),
            oral_microbiota_percent = 0.0,
            oral_microbiota_factor = 1.0,
        ),

        gamma = dict(
            rel_beta        = 2.05, # Estimated to be 1.7–2.4-fold more transmissible than wild-type: https://science.sciencemag.org/content/early/2021/04/13/science.abh2644
            rel_symp_prob   = 1.0,
            rel_severe_prob = 2.6, # From https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.16.2100348
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
            dur_exp2inf     = dict(dist='lognormal_int', par1=4.5, par2=1.5),
            dur_inf2sym     = dict(dist='lognormal_int', par1=1.1, par2=0.9),
            dur_sym2sev     = dict(dist='lognormal_int', par1=6.6, par2=4.9),
            dur_sev2crit    = dict(dist='lognormal_int', par1=1.5, par2=2.0),
            dur_asym2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_mild2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_sev2rec     = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2rec    = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2die    = dict(dist='lognormal_int', par1=10.7, par2=4.8),
            oral_microbiota_percent = 0.0,
            oral_microbiota_factor = 1.0,
        ),

        delta = dict(
            rel_beta        = 2.2, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researchsquare.com/article/rs-637724/v1
            rel_symp_prob   = 1.0,
            rel_severe_prob = 3.2, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
            dur_exp2inf     = dict(dist='lognormal_int', par1=4.5, par2=1.5),
            dur_inf2sym     = dict(dist='lognormal_int', par1=1.1, par2=0.9),
            dur_sym2sev     = dict(dist='lognormal_int', par1=6.6, par2=4.9),
            dur_sev2crit    = dict(dist='lognormal_int', par1=1.5, par2=2.0),
            dur_asym2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_mild2rec    = dict(dist='lognormal_int', par1=8.0,  par2=2.0),
            dur_sev2rec     = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2rec    = dict(dist='lognormal_int', par1=18.1, par2=6.3),
            dur_crit2die    = dict(dist='lognormal_int', par1=10.7, par2=4.8),
            oral_microbiota_percent = 0.0,
            oral_microbiota_factor = 1.0,
        )
    )

    return _get_from_pars(pars, default, key=variant, defaultkey='wild')


def get_cross_immunity(default=False, variant=None):
    '''
    Get the cross immunity between each variant in a sim
    '''
    pars = dict(

        wild = dict(
            wild  = 1.0, # Default for own-immunity
            alpha = 0.5, # Assumption
            beta  = 0.5, # https://www.nature.com/articles/s41586-021-03471-w
            gamma = 0.34, # Assumption
            delta = 0.374, # Assumption
        ),

        alpha = dict(
            wild  = 0.5, # Assumption
            alpha = 1.0, # Default for own-immunity
            beta  = 0.8, # Assumption
            gamma = 0.8, # Assumption
            delta = 0.689  # Assumption
        ),

        beta = dict(
            wild  = 0.066, # https://www.nature.com/articles/s41586-021-03471-w
            alpha = 0.5,   # Assumption
            beta  = 1.0,   # Default for own-immunity
            gamma = 0.5,   # Assumption
            delta = 0.086    # Assumption
        ),

        gamma = dict(
            wild  = 0.34, # Previous (non-P.1) infection provides 54–79% of the protection against infection with P.1 that it provides against non-P.1 lineages: https://science.sciencemag.org/content/early/2021/04/13/science.abh2644
            alpha = 0.4,  # Assumption based on the above
            beta  = 0.4,  # Assumption based on the above
            gamma = 1.0,  # Default for own-immunity
            delta = 0.088   # Assumption
        ),

        delta = dict( # Parameters from https://www.cell.com/cell/fulltext/S0092-8674(21)00755-8
            wild  = 0.374,
            alpha = 0.689,
            beta  = 0.086,
            gamma = 0.088,
            delta = 1.0 # Default for own-immunity
        ),
    )

    return _get_from_pars(pars, default, key=variant, defaultkey='wild')


def get_vaccine_variant_pars(default=False, vaccine=None):
    '''
    Define the effectiveness of each vaccine against each variant
    '''
    pars = dict(

        default = dict(
            wild  = 1.0,
            alpha = 1.0,
            beta  = 1.0,
            gamma = 1.0,
            delta = 1.0,
        ),

        pfizer = dict(
            wild  = 1.0,
            alpha = 1/2.0, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
            beta  = 1/10.3, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
            gamma = 1/6.7, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
            delta = 1/2.9, # https://www.researchsquare.com/article/rs-637724/v1
        ),

        moderna = dict(
            wild  = 1.0,
            alpha = 1/1.8,
            beta  = 1/4.5,
            gamma = 1/8.6, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
            delta = 1/2.9,  # https://www.researchsquare.com/article/rs-637724/v1
        ),

        az = dict(
            wild  = 1.0,
            alpha = 1/2.3,
            beta  = 1/9,
            gamma = 1/2.9,
            delta = 1/6.2,  # https://www.researchsquare.com/article/rs-637724/v1
        ),

        jj = dict(
            wild  = 1.0,
            alpha = 1.0,
            beta  = 1/3.6,  # https://www.biorxiv.org/content/10.1101/2021.07.01.450707v1.full.pdf
            gamma = 1/3.4,  # https://www.biorxiv.org/content/10.1101/2021.07.01.450707v1.full.pdf
            delta = 1/1.6,  # https://www.biorxiv.org/content/10.1101/2021.07.01.450707v1.full.pdf
        ),

        novavax = dict( # Data from https://ir.novavax.com/news-releases/news-release-details/novavax-covid-19-vaccine-demonstrates-893-efficacy-uk-phase-3
            wild  = 1.0,
            alpha = 1/1.12,
            beta  = 1/4.7,
            gamma = 1/8.6, # Assumption, no data available yet
            delta = 1/6.2, # Assumption, no data available yet
        ),

        sinovac = dict(
            wild  = 1.0,
            alpha = 1/1.12,
            beta  = 1/4.7,
            gamma = 1/8.6, # Assumption, no data available yet
            delta = 1/6.2, # Assumption, no data available yet
        ),

        sinopharm = dict(
            wild  = 1.0,
            alpha = 1/1.12,
            beta  = 1/4.7,
            gamma = 1/8.6, # Assumption, no data available yet
            delta = 1/6.2, # Assumption, no data available yet
        )
    )

    return _get_from_pars(pars, default=default, key=vaccine)


def get_vaccine_dose_pars(default=False, vaccine=None):
    '''
    Define the parameters for each vaccine
    '''

    pars = dict(

        default = dict(
            nab_init  = dict(dist='normal', par1=0, par2=2), # Initial distribution of NAbs
            nab_boost = 2, # Factor by which a dose increases existing NABs
            doses     = 1, # Number of doses for this vaccine
            interval  = None, # Interval between doses
        ),

        pfizer = dict(
            nab_init  = dict(dist='normal', par1=-1, par2=2),
            nab_boost = 4,
            doses     = 2,
            interval  = 21,
        ),

        moderna = dict(
            nab_init  = dict(dist='normal', par1=-1, par2=2),
            nab_boost = 8,
            doses     = 2,
            interval  = 28,
        ),

        az = dict(
            nab_init  = dict(dist='normal', par1=-1.5, par2=2),
            nab_boost = 2,
            doses     = 2,
            interval  = 21,
        ),

        jj = dict(
            nab_init  = dict(dist='normal', par1=1, par2=2),
            nab_boost = 3,
            doses     = 1,
            interval  = None,
        ),

        novavax = dict(
            nab_init  = dict(dist='normal', par1=-0.9, par2=2),
            nab_boost = 3,
            doses     = 2,
            interval  = 21,
        ),

        sinovac = dict(
            nab_init  = dict(dist='normal', par1=-2, par2=2),
            nab_boost = 2,
            doses     = 2,
            interval  = 14,
        ),

        sinopharm = dict(
            nab_init  = dict(dist='normal', par1=-1, par2=2),
            nab_boost = 2,
            doses     = 2,
            interval  = 21,
        )
    )

    return _get_from_pars(pars, default, key=vaccine)