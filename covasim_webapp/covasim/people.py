'''
Defines the People class and functions associated with making people and handling
the transitions between states (e.g., from susceptible to infected).
'''

#%% Imports
import numpy as np
import sciris as sc
from collections import defaultdict
from . import version as cvv
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb
from . import plotting as cvplt
from . import immunity as cvi


__all__ = ['People']

class People(cvb.BasePeople):
    '''
    A class to perform all the operations on the people -- usually not invoked directly.

    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized. However, ages, contacts, etc. will need to be created separately --
    see ``cv.make_people()`` instead.

    Note that this class handles the mechanics of updating the actual people, while
    ``cv.BasePeople`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as pop_size
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::

        ppl1 = cv.People(2000)

        sim = cv.Sim()
        ppl2 = cv.People(sim.pars)
    '''

    def __init__(self, pars, strict=True, **kwargs):

        self.inds_new_infections = []
        # Handle pars and population size
        self.set_pars(pars)
        self.version = cvv.__version__ # Store version info

        # Other initialization
        self.t = 0 # Keep current simulation time
        self._lock = False # Prevent further modification of keys
        self.meta = cvd.PeopleMeta() # Store list of keys and dtypes
        self.contacts = None
        self.init_contacts() # Initialize the contacts
        self.infection_log = [] # Record of infections - keys for ['source','target','date','layer']

        # Set person properties -- all floats except for UID
        for key in self.meta.person:
            if key == 'uid':
                self[key] = np.arange(self.pars['pop_size'], dtype=cvd.default_int)
            elif key in ['n_infections', 'n_breakthroughs']:
                self[key] = np.zeros(self.pars['pop_size'], dtype=cvd.default_int)
            else:
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)

        # Set health states -- only susceptible is true by default -- booleans except exposed by variant which should return the variant that ind is exposed to
        for key in self.meta.states:
            val = (key in ['susceptible', 'naive']) # Default value is True for susceptible and naive, false otherwise
            self[key] = np.full(self.pars['pop_size'], val, dtype=bool)

        # Set variant states, which store info about which variant a person is exposed to
        for key in self.meta.variant_states:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)
        for key in self.meta.by_variant_states:
            self[key] = np.full((self.pars['n_variants'], self.pars['pop_size']), False, dtype=bool)

        # Set immunity and antibody states
        for key in self.meta.imm_states:  # Everyone starts out with no immunity
            self[key] = np.zeros((self.pars['n_variants'], self.pars['pop_size']), dtype=cvd.default_float)
        for key in self.meta.nab_states:  # Everyone starts out with no antibodies
            dtype = cvd.default_int if key == 't_nab_event' else cvd.default_float
            self[key] = np.zeros(self.pars['pop_size'], dtype=dtype)
        for key in self.meta.vacc_states:
            self[key] = np.zeros(self.pars['pop_size'], dtype=cvd.default_int)

        # Set dates and durations -- both floats
        for key in self.meta.dates + self.meta.durs:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cvd.default_float)

        # Store the dtypes used in a flat dict
        self._dtypes = {key:self[key].dtype for key in self.keys()} # Assign all to float by default
        if strict:
            self.lock() # If strict is true, stop further keys from being set (does not affect attributes)

        # Store flows to be computed during simulation
        self.init_flows()

        # Although we have called init(), we still need to call initialize()
        self.initialized = False

        # Handle contacts, if supplied (note: they usually are)
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts'))

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if strict:
                self.set(key, value)
            else:
                self[key] = value

        self._pending_quarantine = defaultdict(list)  # Internal cache to record people that need to be quarantined on each timestep {t:(person, quarantine_end_day)}
        return

    def _save_array_members(self):
        self.all_array_members = [attr for attr in dir(self) if
                                  not callable(getattr(self, attr)) and not attr.startswith(
                                      "__") and not attr.startswith("_") and isinstance(getattr(self, attr),
                                                                                        np.ndarray)]  #
        self.intercity_contacts = None
        self.array_members = []
        self.two_array_members = []
        for array_member in self.all_array_members:
            if len(self[array_member].shape) > 1:
                self.two_array_members.append(array_member)
            else:
                self.array_members.append(array_member)

        for array_member in self.array_members:
            if array_member == "uid":
                self[array_member] = np.arange(self.pop_size + self.tourist_count, dtype=cvd.default_int)
            else:
                self[array_member] = np.resize(self[array_member], self.pop_size + self.tourist_count)

        total_pop_size = self.pop_size + self.tourist_count
        for two_array_member in self.two_array_members:
            new_two_array = np.empty(shape=(self[two_array_member].shape[0], total_pop_size),
                                     dtype=self[two_array_member].dtype)
            for dim in range(self[two_array_member].shape[0]):
                new_two_array[dim] = np.resize(self[two_array_member][dim], total_pop_size)
            self[two_array_member] = new_two_array

    def extend_tourists_fields(self, own_ind_city, tourist_ratio=0.1):
        self.tourist_count = int(self.pop_size * tourist_ratio)
        # additional fields
        self.true_uid = np.arange(self.pop_size + self.tourist_count, dtype=cvd.default_int)
        self.in_city = np.full(self.pop_size, True, dtype=bool)
        self.rest_in_another_city_days = -np.ones(self.pop_size, dtype=cvd.default_int)
        self.own_city = np.full(self.pop_size, own_ind_city, dtype=cvd.default_int)
        # save arrays
        self._save_array_members()
        # make not existing people
        self.make_not_existing(self.pop_size + np.arange(self.tourist_count))


    def make_not_existing(self, inds):
        self.in_city[inds] = False
        self.rest_in_another_city_days[inds] = -1
        self.rel_trans[inds] = 0.0
        self.rel_sus[inds] = 0.0
        self.susceptible[inds] = False
        self.infectious[inds] = False


    def __len__(self):
        ''' This is just a scalar, but validate() and _resize_arrays() make sure it's right '''
        return int(self.pop_size + self.tourist_count)

    def init_flows(self):
        ''' Initialize flows to be zero '''
        self.flows = {key:0 for key in cvd.new_result_flows}
        self.flows_variant = {}
        for key in cvd.new_result_flows_by_variant:
            self.flows_variant[key] = np.zeros(self.pars['n_variants'], dtype=cvd.default_float)
        return

    def right_hist(self, arr):
        unique_values, counts = np.unique(arr, return_counts=True)
        tmp_x = np.arange(np.max(unique_values) + 1)
        tmp_y = np.zeros(shape=tmp_x.size)
        tmp_y[unique_values] = counts
        return tmp_x, tmp_y

    def get_contact_statistics(self):
        people2cnt = np.zeros(shape=len(self), dtype=np.int)
        for lk in self.contacts.keys():
            layer = self.contacts[lk]
            uids_1, counts_1 = np.unique(layer['p1'], return_counts=True)
            uids_2, counts_2 = np.unique(layer['p2'], return_counts=True)
            people2cnt[uids_1] += counts_1
            people2cnt[uids_2] += counts_2
        return people2cnt

    def plot_contact_statistics(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        total_hists = dict()
        people2cnt = np.zeros(shape=len(self), dtype=np.int)
        for lk in self.contacts.keys():
            layer = self.contacts[lk]
            uids_1, counts_1 = np.unique(layer['p1'], return_counts=True)
            uids_2, counts_2 = np.unique(layer['p2'], return_counts=True)
            people2cnt[uids_1] += counts_1
            people2cnt[uids_2] += counts_2
            # lk hists
            p1_x_hist, p1_y_hist = self.right_hist(counts_1)
            p2_x_hist, p2_y_hist = self.right_hist(counts_2)
            univ_size = np.max([p1_x_hist.size, p2_x_hist.size])
            p1_x_hist.resize(univ_size)
            p1_y_hist.resize(univ_size)
            p2_x_hist.resize(univ_size)
            p2_y_hist.resize(univ_size)
            total_hists[lk] = p1_x_hist + p2_x_hist, p1_y_hist + p2_y_hist
        total_hists['common'] = self.right_hist(people2cnt)

        fig = make_subplots(rows=2, cols=3)
        fig.add_trace(go.Bar(x=total_hists['c'][0], y=total_hists['c'][1], name='Random contacts'), row=1, col=1)
        fig.add_trace(go.Bar(x=total_hists['h'][0], y=total_hists['h'][1], name='Home contacts'), row=1, col=2)
        fig.add_trace(go.Bar(x=total_hists['w'][0], y=total_hists['w'][1], name='Work contacts'), row=1, col=3)
        fig.add_trace(go.Bar(x=total_hists['s'][0], y=total_hists['s'][1], name='School contacts'), row=2, col=1)
        fig.add_trace(go.Bar(x=total_hists['common'][0], y=total_hists['common'][1], name='Common contacts'), row=2, col=2)


        # Update the layout
        fig.update_layout(title='Contact histograms')
        fig.show()
        return fig

    def initialize_tourist_cells(self):
        self.pop_size = self.pars['pop_size']
        self.tourist_count = 0
        if 'own_ind_city' in self.pars:
            self.extend_tourists_fields(self.pars['own_ind_city'], tourist_ratio=self.pars['tourist_ratio'])


    def initialize(self, sim_pars=None):
        ''' Perform initializations '''
        # tourist init
        self.initialize_tourist_cells()
        # and otherwise
        self._test_delay = cvu.sample(dist='special_test_delay', size=len(self))
        self.inds_new_infections = []
        self.validate(sim_pars=sim_pars) # First, check that essential-to-match parameters match
        self.set_pars(sim_pars) # Replace the saved parameters with this simulation's
        self.set_prognoses()
        self.initialized = True
        return


    def set_prognoses(self):
        '''
        Set the prognoses for each person based on age during initialization. Need
        to reset the seed because viral loads are drawn stochastically.
        '''

        pars = self.pars # Shorten
        if 'prognoses' not in pars or 'rand_seed' not in pars:
            errormsg = 'This people object does not have the required parameters ("prognoses" and "rand_seed"). Create a sim (or parameters), then do e.g. people.set_pars(sim.pars).'
            raise sc.KeyNotFoundError(errormsg)

        cvu.set_seed(pars['rand_seed'])

        progs = pars['prognoses'] # Shorten the name
        inds = np.digitize(self.age, progs['age_cutoffs'])-1
        self.symp_prob[:]   = progs['symp_probs'][inds] # Probability of developing symptoms
        self.severe_prob[:] = progs['severe_probs'][inds]*progs['comorbidities'][inds] # Severe disease probability is modified by comorbidities
        self.crit_prob[:]   = progs['crit_probs'][inds] # Probability of developing critical disease
        self.death_prob[:]  = progs['death_probs'][inds] # Probability of death
        # self.rel_sus[:]     = progs['sus_ORs'][inds]  # Default susceptibilities
        # self.rel_sus[:]     = np.random.uniform(0, 1.47, size=self.rel_sus.size) # Uniform susceptibilities
        # self.rel_sus[:]     = cvu.sample(dist='lognormal', par1=1.0, par2=0.2, size=self.rel_sus.size) # lognormal susceptibilities
        # self.rel_sus[:]     = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=self.rel_sus.size) # normal susceptibilities

        # 0.34   0.67   1   1   1   1   1.24   1.47   1.47   1.47
        # (0.17, 0.5)  (0.5, 0.8) (0.8, 1.1) (1.1, 1.34) (1.34, 1.5)
        constants_sum = np.sum(progs['sus_ORs'][inds])
        def normalize_rel_sus(rel_sus_b):
            coef = float(constants_sum) / (np.sum(rel_sus_b))
            return rel_sus_b * coef

        if pars['rel_sus_type'] == 'constants':
            self.rel_sus[:] = progs['sus_ORs'][inds]  # Default susceptibilities
        elif pars['rel_sus_type'] == 'normal_pos':
            for i in range(1, 10):
                inds_age = np.where((progs['age_cutoffs'][i - 1] <= self.age) * (self.age < progs['age_cutoffs'][i]))[0]
                self.rel_sus[inds_age] = cvu.sample(dist='normal_pos', par1=progs['sus_ORs'][i - 1], par2=0.2,
                                                    size=inds_age.size)
            inds_age = np.where(self.age > progs['age_cutoffs'][9])[0]
            self.rel_sus[inds_age] = cvu.sample(dist='normal_pos', par1=progs['sus_ORs'][9], par2=0.2,
                                                size=inds_age.size)
        elif pars['rel_sus_type'] == 'lognormal':
            for i in range(1, 10):
                inds_age = np.where((progs['age_cutoffs'][i - 1] <= self.age) * (self.age < progs['age_cutoffs'][i]))[0]
                self.rel_sus[inds_age] = cvu.sample(dist='lognormal', par1=progs['sus_ORs'][i - 1], par2=0.2,
                                                    size=inds_age.size)
            inds_age = np.where(self.age > progs['age_cutoffs'][9])[0]
            self.rel_sus[inds_age] = cvu.sample(dist='lognormal', par1=progs['sus_ORs'][9], par2=0.2,
                                                size=inds_age.size)
        elif pars['rel_sus_type'] == 'uniform':
            my_unfiform_intervals = [(0.17, 0.5), (0.5, 0.8), (0.8, 1.1), (0.8, 1.1), (0.8, 1.1), (0.8, 1.1),
                                     (1.1, 1.34), (1.34, 1.5), (1.34, 1.5), (1.34, 1.5)]
            for i in range(1, 10):
                inds_age = np.where((progs['age_cutoffs'][i - 1] <= self.age) * (self.age < progs['age_cutoffs'][i]))[0]
                self.rel_sus[inds_age] = np.random.uniform(
                    my_unfiform_intervals[i - 1][0], my_unfiform_intervals[i - 1][1], size=inds_age.size)
            inds_age = np.where(self.age > progs['age_cutoffs'][9])[0]
            self.rel_sus[inds_age] = np.random.uniform(
                my_unfiform_intervals[9][0], my_unfiform_intervals[9][1], size=inds_age.size)
        elif pars['rel_sus_type'] == 'uniform_all':
            self.rel_sus[:]     = np.random.uniform(0, 1.47, size=self.rel_sus.size) # Uniform susceptibilities
        elif pars['rel_sus_type'] == 'lognormal_lite_all':
            self.rel_sus[:]     = cvu.sample(dist='lognormal', par1=0.65, par2=0.5, size=self.rel_sus.size) # lognormal susceptibilities
            ss = self.rel_sus[self.rel_sus > 1.5].size
            self.rel_sus[self.rel_sus > 1.5] = cvu.sample(dist='lognormal', par1=0.5, par2=0.5, size=ss)
        elif pars['rel_sus_type'] == 'lognormal_hot_all':
            self.rel_sus[:]     = cvu.sample(dist='lognormal', par1=1.0, par2=0.5, size=self.rel_sus.size) # lognormal susceptibilities
            ss = self.rel_sus[self.rel_sus > 1.5].size
            self.rel_sus[self.rel_sus > 1.5] = cvu.sample(dist='lognormal', par1=1.0, par2=0.5, size=ss)        
        elif pars['rel_sus_type'] == 'normal_pos_all':
            self.rel_sus[:]     = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=self.rel_sus.size) # normal susceptibilities
            ss = self.rel_sus[self.rel_sus > 1.5].size
            self.rel_sus[self.rel_sus > 1.5] = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=ss)
        elif pars['rel_sus_type'] == 'beta_1_all':
            self.rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=1.2, par2=6.2, step=1.5, size=self.rel_sus.size)
        elif pars['rel_sus_type'] == 'beta_2_all':
            self.rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=1.8, par2=6.2, step=1.5, size=self.rel_sus.size)
        elif pars['rel_sus_type'] == 'beta_3_all':
            self.rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=2.2, par2=5.6, step=1.5, size=self.rel_sus.size)
        elif pars['rel_sus_type'] == 'neg_binomial_all':
            self.rel_sus[:]     = 0.1 + cvu.sample(dist='neg_binomial', par1=0.21, par2=3, step=0.07, size=self.rel_sus.size)
        #elif pars['rel_sus_type'] == 'binom_2_all':
        #    self.rel_sus[:]     = cvu.sample(dist='beta', par1=2.2, par2=5.6, step=1.5, size=self.rel_sus.size)    
        else:
            raise RuntimeError("Not corrected type of rel_sus")
        
        self.rel_sus[self.rel_sus > 2.5] = 2.5
        self.rel_sus = normalize_rel_sus(self.rel_sus)

        tmp_trans = progs['trans_ORs'][inds] * cvu.sample(**self.pars['beta_dist'], size=len(inds))  # Default transmissibilities, with viral load drawn from a distribution
        if pars['rel_trans_type'] == 'eq_res_type':
            tmp_rel_sus_sort_inds = np.argsort(self.rel_sus)
            self.rel_trans[tmp_rel_sus_sort_inds] = np.sort(tmp_trans)[:]
        else:
            self.rel_trans[:]   = tmp_trans[:]

        return


    def update_states_pre(self, t):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.is_exp = self.true('exposed') # For storing the interim values since used in every subsequent calculation

        # Perform updates

        self.inds_new_infections = []
        self.init_flows()
        self.flows['new_infectious']    += len(self.check_infectious()) # For people who are exposed and not infectious, check if they begin being infectious
        self.flows['new_symptomatic']   += len(self.check_symptomatic())
        self.flows['new_severe']        += len(self.check_severe())
        self.flows['new_critical']      += len(self.check_critical())
        self.flows['new_recoveries']    += len(self.check_recovery())
        self.check_exit_iso()
        _, new_deaths, new_known_deaths    = self.check_death()
        self.flows['new_deaths']        += new_deaths
        self.flows['new_known_deaths']  += new_known_deaths

        if self.pars['use_waning']:
            cvi.check_immunity(self)

        return


    def update_states_post(self):
        ''' Perform post-timestep updates '''
        tmp_check_diagnosed = self.check_diagnosed()
        self.flows['new_diagnoses'] += len(tmp_check_diagnosed[0]) 
        self.flows['new_diagnoses_rpn'] += len(tmp_check_diagnosed[1])
        self.flows['new_quarantined'] += self.check_quar()
        self.flows['new_isolated'] += len(self.check_enter_iso())
        del self.is_exp  # Tidy up

        return


    def update_contacts(self):
        ''' Refresh dynamic contacts, e.g. community '''
        # Figure out if anything needs to be done -- e.g. {'h':False, 'c':True}
        for lkey, is_dynam in self.pars['dynam_layer'].items():
            if is_dynam:
                self.contacts[lkey].update(self)

        return self.contacts


    #%% Methods for updating state

    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet the date criterion '''
        if filter_inds is None:
            not_current = cvu.false(current)
        else:
            not_current = cvu.ifalsei(current, filter_inds)
        has_date = cvu.idefinedi(date, not_current)
        inds     = cvu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_infectious(self):
        ''' Check if they become infectious '''
        inds = self.check_inds(self.infectious, self.date_infectious, filter_inds=self.is_exp)
        self.infectious[inds] = True
        self.infectious_variant[inds] = self.exposed_variant[inds]
        for variant in range(self.pars['n_variants']):
            this_variant_inds = cvu.itrue(self.infectious_variant[inds] == variant, inds)
            n_this_variant_inds = len(this_variant_inds)
            self.flows_variant['new_infectious_by_variant'][variant] += n_this_variant_inds
            self.infectious_by_variant[variant, this_variant_inds] = True
        #print(f"Count tourist infected {np.count_nonzero(inds >= 100000)}")
        return inds


    def check_symptomatic(self):
        ''' Check for new progressions to symptomatic '''
        inds = self.check_inds(self.symptomatic, self.date_symptomatic, filter_inds=self.is_exp)
        self.symptomatic[inds] = True
        return inds


    def check_severe(self):
        ''' Check for new progressions to severe '''
        inds = self.check_inds(self.severe, self.date_severe, filter_inds=self.is_exp)
        self.severe[inds] = True
        return inds


    def check_critical(self):
        ''' Check for new progressions to critical '''
        inds = self.check_inds(self.critical, self.date_critical, filter_inds=self.is_exp)
        self.critical[inds] = True
        return inds


    def check_recovery(self, inds=None, filter_inds='is_exp'):
        '''
        Check for recovery.

        More complex than other functions to allow for recovery to be manually imposed
        for a specified set of indices.
        '''

        # Handle more flexible options for setting indices
        if filter_inds == 'is_exp':
            filter_inds = self.is_exp
        if inds is None:
            inds = self.check_inds(self.recovered, self.date_recovered, filter_inds=filter_inds)

        # Now reset all disease states
        self.exposed[inds]          = False
        self.infectious[inds]       = False
        self.symptomatic[inds]      = False
        self.severe[inds]           = False
        self.critical[inds]         = False
        self.recovered[inds]        = True
        self.recovered_variant[inds] = self.exposed_variant[inds]
        self.infectious_variant[inds] = np.nan
        self.exposed_variant[inds]    = np.nan
        self.exposed_by_variant[:, inds] = False
        self.infectious_by_variant[:, inds] = False


        # Handle immunity aspects
        if self.pars['use_waning']:

            # Reset additional states
            self.susceptible[inds] = True
            self.diagnosed[inds]   = False # Reset their diagnosis state because they might be reinfected

        return inds


    def check_death(self):
        ''' Check whether or not this person died on this timestep  '''
        inds = self.check_inds(self.dead, self.date_dead, filter_inds=self.is_exp)
        self.dead[inds]             = True
        diag_inds = inds[self.diagnosed[inds]] # Check whether the person was diagnosed before dying
        self.known_dead[diag_inds]  = True
        self.susceptible[inds]      = False
        self.exposed[inds]          = False
        self.infectious[inds]       = False
        self.symptomatic[inds]      = False
        self.severe[inds]           = False
        self.critical[inds]         = False
        self.known_contact[inds]    = False
        self.quarantined[inds]      = False
        self.recovered[inds]        = False
        self.infectious_variant[inds] = np.nan
        self.exposed_variant[inds]    = np.nan
        self.recovered_variant[inds]  = np.nan
        return inds, len(inds), len(diag_inds)


    def check_diagnosed(self):
        '''
        Check for new diagnoses. Since most data are reported with diagnoses on
        the date of the test, this function reports counts not for the number of
        people who received a positive test result on a day, but rather, the number
        of people who were tested on that day who are schedule to be diagnosed in
        the future.
        '''
        # Handle people who tested today who will be diagnosed in future
        test_pos_inds = self.check_inds(self.diagnosed, self.date_pos_test, filter_inds=None) # Find people who will be diagnosed in future
        self.date_pos_test[test_pos_inds] = np.nan # Clear date of having will-be-positive test

        # Handle people who were actually diagnosed today
        diag_inds  = self.check_inds(self.diagnosed, self.date_diagnosed, filter_inds=None) # Find who was actually diagnosed on this timestep
        self.date_diagnosed[diag_inds] = np.nan
        self.diagnosed[diag_inds]   = True # Set these people to be diagnosed
        
        #print(test_pos_inds)
        #print(diag_inds)
        #print("-----------")

        return test_pos_inds, diag_inds


    def check_quar(self):
        ''' Update quarantine state '''

        n_quarantined = 0 # Number of people entering quarantine
        for ind,end_day in self._pending_quarantine[self.t]:
            if self.quarantined[ind]:
                self.date_end_quarantine[ind] = max(self.date_end_quarantine[ind], end_day) # Extend quarantine if required
            elif not (self.dead[ind] or self.recovered[ind] or self.diagnosed[ind] or self.isolated[ind]): # Unclear whether recovered should be included here # elif not (self.dead[ind] or self.diagnosed[ind]):
                self.quarantined[ind] = True
                self.date_quarantined[ind] = self.t
                self.date_end_quarantine[ind] = end_day
                n_quarantined += 1

        # If someone has been diagnosed today, end their quarantine
        # By definition, 'quarantine' only applies to people that are not yet diagnosed
        # After diagnosis, they are 'isolating'
        diag_inds  = cvu.true(self.quarantined & (self.date_diagnosed == self.t))
        self.date_end_quarantine[diag_inds] = self.t

        # If someone on quarantine has reached the end of their quarantine, release them
        end_inds = self.check_inds(~self.quarantined, self.date_end_quarantine, filter_inds=None) # Note the double-negative here (~)
        self.quarantined[end_inds] = False # Release from quarantine

        return n_quarantined


    def check_enter_iso(self):
        # Anyone diagnosed today enters isolation for the duration of their infection
        iso_inds  = cvu.true(self.date_diagnosed == self.t)
        self.isolated[iso_inds] = True
        self.date_end_isolation[iso_inds] = self.date_recovered[iso_inds]
        return iso_inds

    def check_exit_iso(self):
        '''
        End isolation for anyone due to exit isolation
        '''
        end_inds = self.check_inds(~self.isolated, self.date_end_isolation, filter_inds=None) # Note the double-negative here (~)
        self.isolated[end_inds] = False # Release from isolation
        return end_inds

    #%% Methods to make events occur (infection and diagnosis)

    def make_naive(self, inds, reset_vx=False):
        '''
        Make a set of people naive. This is used during dynamic resampling.

        Args:
            inds (array): list of people to make naive
            reset_vx (bool): whether to reset vaccine-derived immunity
        '''
        for key in self.meta.states:
            if key in ['susceptible', 'naive']:
                self[key][inds] = True
            else:
                if (key != 'vaccinated') or reset_vx: # Don't necessarily reset vaccination
                    self[key][inds] = False

        # Reset variant states
        for key in self.meta.variant_states:
            self[key][inds] = np.nan
        for key in self.meta.by_variant_states:
            self[key][:, inds] = False

        # Reset immunity and antibody states
        non_vx_inds = inds if reset_vx else inds[~self['vaccinated'][inds]]
        for key in self.meta.imm_states:
            self[key][:, non_vx_inds] = 0
        for key in self.meta.nab_states + self.meta.vacc_states:
            self[key][non_vx_inds] = 0

        # Reset dates
        for key in self.meta.dates + self.meta.durs:
            if (key != 'date_vaccinated') or reset_vx: # Don't necessarily reset vaccination
                self[key][inds] = np.nan

        return


    def make_nonnaive(self, inds, set_recovered=False, date_recovered=0):
        '''
        Make a set of people non-naive.

        This can be done either by setting only susceptible and naive states,
        or else by setting them as if they have been infected and recovered.
        '''
        self.make_naive(inds) # First make them naive and reset all other states

        # Make them non-naive
        for key in ['susceptible', 'naive']:
            self[key][inds] = False

        if set_recovered:
            self.date_recovered[inds] = date_recovered # Reset date recovered
            self.check_recovery(inds=inds, filter_inds=None) # Set recovered

        return



    def infect(self, inds, hosp_max=None, icu_max=None, source=None, layer=None, variant=0):
        '''
        Infect people and determine their eventual outcomes.

            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die
            * If the simulation is being run with waning, this method also sets/updates agents' neutralizing antibody levels

        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds     (array): array of people to infect
            hosp_max (bool):  whether or not there is an acute bed available for this person
            icu_max  (bool):  whether or not there is an ICU bed available for this person
            source   (array): source indices of the people who transmitted this infection (None if an importation or seed infection)
            layer    (str):   contact layer this infection was transmitted on
            variant  (int):   the variant people are being infected by

        Returns:
            count (int): number of people infected
        '''

        # If no infections, short-circuit
        if len(inds) == 0:
            return inds

        # Remove duplicates
        inds, unique = np.unique(inds, return_index=True)
        if source is not None:
            source = source[unique]

        # Keep only susceptibles
        keep = self.susceptible[inds] # Unique indices in inds and source that are also susceptible
        inds = inds[keep]
        if source is not None:
            source = source[keep]

        # Deal with variant parameters
        variant_keys = [
            'rel_symp_prob', 
            'rel_severe_prob',
            'rel_crit_prob', 
            'rel_death_prob',
            'dur_exp2inf',
            'dur_inf2sym',
            'dur_sym2sev',
            'dur_sev2crit',
            'dur_asym2rec',
            'dur_mild2rec',
            'dur_sev2rec',
            'dur_crit2rec',
            'dur_crit2die'
        ]
        infect_pars = {k:self.pars[k] for k in variant_keys}
        variant_label = self.pars['variant_map'][variant]
        if variant:
            for k in variant_keys:
                if 'dur' in k:
                    infect_pars[k] = self.pars['variant_pars'][variant_label][k]
                else:     
                    infect_pars[k] *= self.pars['variant_pars'][variant_label][k]

        # Retrieve those with a breakthrough infection (defined nabs)
        breakthrough_inds = inds[cvu.true(self.peak_nab[inds])]
        if len(breakthrough_inds):
            no_prior_breakthrough = (self.n_breakthroughs[breakthrough_inds] == 0) # We only adjust transmissibility for the first breakthrough
            new_breakthrough_inds = breakthrough_inds[no_prior_breakthrough]
            self.rel_trans[new_breakthrough_inds] *= self.pars['trans_redux']

        # Update states, variant info, and flows
        n_infections = len(inds)
        self.susceptible[inds]    = False
        self.naive[inds]          = False
        self.recovered[inds]      = False
        self.diagnosed[inds]      = False
        self.exposed[inds]        = True
        self.n_infections[inds]  += 1
        self.n_breakthroughs[breakthrough_inds] += 1
        self.exposed_variant[inds] = variant
        self.exposed_by_variant[variant, inds] = True
        
        self.inds_new_infections += list(inds)
        self.flows['new_infections']   += n_infections
        self.flows['new_reinfections'] += len(cvu.defined(self.date_recovered[inds])) # Record reinfections
        self.flows_variant['new_infections_by_variant'][variant] += n_infections
 
        # Record transmissions
        for i, target in enumerate(inds):
            entry = dict(source=source[i] if source is not None else None, target=target, date=self.t, layer=layer, variant=variant_label)
            self.infection_log.append(entry)

        # Calculate how long before this person can infect other people
        self.dur_exp2inf[inds] = cvu.sample(**infect_pars['dur_exp2inf'], size=n_infections)
        self.date_exposed[inds]   = self.t
        self.date_infectious[inds] = self.dur_exp2inf[inds] + self.t

        # Reset all other dates
        for key in ['date_symptomatic', 'date_severe', 'date_critical', 'date_diagnosed', 'date_recovered']:
            self[key][inds] = np.nan

        # Use prognosis probabilities to determine what happens to them
        symp_probs = infect_pars['rel_symp_prob']*self.symp_prob[inds]*(1-self.symp_imm[variant, inds]) # Calculate their actual probability of being symptomatic
        is_symp = cvu.binomial_arr(symp_probs) # Determine if they develop symptoms
        symp_inds = inds[is_symp]
        asymp_inds = inds[~is_symp] # Asymptomatic
        self.flows_variant['new_symptomatic_by_variant'][variant] += len(symp_inds)

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        dur_asym2rec = cvu.sample(**infect_pars['dur_asym2rec'], size=len(asymp_inds))
        self.date_recovered[asymp_inds] = self.date_infectious[asymp_inds] + dur_asym2rec  # Date they recover
        self.dur_disease[asymp_inds] = self.dur_exp2inf[asymp_inds] + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        n_symp_inds = len(symp_inds)
        self.dur_inf2sym[symp_inds] = cvu.sample(**infect_pars['dur_inf2sym'], size=n_symp_inds) # Store how long this person took to develop symptoms
        self.date_symptomatic[symp_inds] = self.date_infectious[symp_inds] + self.dur_inf2sym[symp_inds] # Date they become symptomatic
        sev_probs = infect_pars['rel_severe_prob'] * self.severe_prob[symp_inds]*(1-self.sev_imm[variant, symp_inds]) # Probability of these people being severe
        is_sev = cvu.binomial_arr(sev_probs) # See if they're a severe or mild case
        sev_inds = symp_inds[is_sev]
        mild_inds = symp_inds[~is_sev] # Not severe
        self.flows_variant['new_severe_by_variant'][variant] += len(sev_inds)

        # CASE 2.1: Mild symptoms, no hospitalization required and no probability of death
        dur_mild2rec = cvu.sample(**infect_pars['dur_mild2rec'], size=len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = self.dur_exp2inf[mild_inds] + self.dur_inf2sym[mild_inds] + dur_mild2rec  # Store how long this person had COVID-19

        # CASE 2.2: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = cvu.sample(**infect_pars['dur_sym2sev'], size=len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = self.date_symptomatic[sev_inds] + self.dur_sym2sev[sev_inds]  # Date symptoms become severe
        crit_probs = infect_pars['rel_crit_prob'] * self.crit_prob[sev_inds] * (self.pars['no_hosp_factor'] if hosp_max else 1.) # Probability of these people becoming critical - higher if no beds available
        is_crit = cvu.binomial_arr(crit_probs)  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # CASE 2.2.1 Not critical - they will recover
        dur_sev2rec = cvu.sample(**infect_pars['dur_sev2rec'], size=len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = self.dur_exp2inf[non_crit_inds] + self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec  # Store how long this person had COVID-19

        # CASE 2.2.2: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = cvu.sample(**infect_pars['dur_sev2crit'], size=len(crit_inds))
        self.date_critical[crit_inds] = self.date_severe[crit_inds] + self.dur_sev2crit[crit_inds]  # Date they become critical
        death_probs = infect_pars['rel_death_prob'] * self.death_prob[crit_inds] * (self.pars['no_icu_factor'] if icu_max else 1.)# Probability they'll die
        is_dead = cvu.binomial_arr(death_probs)  # Death outcome
        dead_inds = crit_inds[is_dead]
        alive_inds = crit_inds[~is_dead]

        # CASE 2.2.2.1: Did not die
        dur_crit2rec = cvu.sample(**infect_pars['dur_crit2rec'], size=len(alive_inds))
        self.date_recovered[alive_inds] = self.date_critical[alive_inds] + dur_crit2rec # Date they recover
        self.dur_disease[alive_inds] = self.dur_exp2inf[alive_inds] + self.dur_inf2sym[alive_inds] + self.dur_sym2sev[alive_inds] + self.dur_sev2crit[alive_inds] + dur_crit2rec  # Store how long this person had COVID-19

        # CASE 2.2.2.2: Did die
        dur_crit2die = cvu.sample(**infect_pars['dur_crit2die'], size=len(dead_inds))
        self.date_dead[dead_inds] = self.date_critical[dead_inds] + dur_crit2die # Date of death
        self.dur_disease[dead_inds] = self.dur_exp2inf[dead_inds] + self.dur_inf2sym[dead_inds] + self.dur_sym2sev[dead_inds] + self.dur_sev2crit[dead_inds] + dur_crit2die   # Store how long this person had COVID-19
        self.date_recovered[dead_inds] = np.nan # If they did die, remove them from recovered

        # Handle immunity aspects
        if self.pars['use_waning']:
            symp = dict(asymp=asymp_inds, mild=mild_inds, sev=sev_inds)
            cvi.update_peak_nab(self, inds, nab_pars=self.pars, symp=symp)

        return inds # For incrementing counters


    def test(self, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''

        inds = np.unique(inds)
        self.tested[inds] = True
        self.date_tested[inds] = self.t # Only keep the last time they tested

        is_infectious = cvu.itruei(self.infectious, inds)
        pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.date_diagnosed[is_inf_pos])]
        not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
        final_inds    = not_diagnosed[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[final_inds] = self.t + self._test_delay[final_inds]
        self.date_pos_test[final_inds] = self.t

        return final_inds


    def schedule_quarantine(self, inds, start_date=None, period=None):
        '''
        Schedule a quarantine. Typically not called by the user directly except
        via a custom intervention; see the contact_tracing() intervention instead.

        This function will create a request to quarantine a person on the start_date for
        a period of time. Whether they are on an existing quarantine that gets extended, or
        whether they are no longer eligible for quarantine, will be checked when the start_date
        is reached.

        Args:
            inds (int): indices of who to quarantine, specified by check_quar()
            start_date (int): day to begin quarantine (defaults to the current day, `sim.t`)
            period (int): quarantine duration (defaults to ``pars['quar_period']``)
        '''

        start_date = self.t if start_date is None else int(start_date)
        period = self.pars['quar_period'] if period is None else int(period)
        for ind in inds:
            self._pending_quarantine[start_date].append((ind, start_date + period))
        return


    #%% Analysis methods

    def plot(self, *args, **kwargs):
        '''
        Plot statistics of the population -- age distribution, numbers of contacts,
        and overall weight of contacts (number of contacts multiplied by beta per
        layer).

        Args:
            bins      (arr)   : age bins to use (default, 0-100 in one-year bins)
            width     (float) : bar width
            font_size (float) : size of font
            alpha     (float) : transparency of the plots
            fig_args  (dict)  : passed to pl.figure()
            axis_args (dict)  : passed to pl.subplots_adjust()
            plot_args (dict)  : passed to pl.plot()
            do_show   (bool)  : whether to show the plot
            fig       (fig)   : handle of existing figure to plot into
        '''
        fig = cvplt.plot_people(people=self, *args, **kwargs)
        return fig


    def story(self, uid, *args):
        '''
        Print out a short history of events in the life of the specified individual.

        Args:
            uid (int/list): the person or people whose story is being regaled
            args (list): these people will tell their stories too

        **Example**::

            sim = cv.Sim(pop_type='hybrid', verbose=0)
            sim.run()
            sim.people.story(12)
            sim.people.story(795)
        '''

        def label_lkey(lkey):
            ''' Friendly name for common layer keys '''
            if lkey.lower() == 'a':
                llabel = 'default contact'
            if lkey.lower() == 'h':
                llabel = 'household'
            elif lkey.lower() == 's':
                llabel = 'school'
            elif lkey.lower() == 'w':
                llabel = 'workplace'
            elif lkey.lower() == 'c':
                llabel = 'community'
            else:
                llabel = f'"{lkey}"'
            return llabel

        uids = sc.tolist(uid)
        uids.extend(args)

        for uid in uids:

            p = self[uid]
            sex = 'female' if p.sex == 0 else 'male'

            intro = f'\nThis is the story of {uid}, a {p.age:.0f} year old {sex}'

            if not p.susceptible:
                if np.isnan(p.date_symptomatic):
                    print(f'{intro}, who had asymptomatic COVID.')
                else:
                    print(f'{intro}, who had symptomatic COVID.')
            else:
                print(f'{intro}, who did not contract COVID.')

            total_contacts = 0
            no_contacts = []
            for lkey in p.contacts.keys():
                llabel = label_lkey(lkey)
                n_contacts = len(p.contacts[lkey])
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                nc_string = ', '.join(no_contacts)
                print(f'{uid} has no contacts in the {nc_string} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

            events = []

            dates = {
                'date_critical'       : 'became critically ill and needed ICU care',
                'date_dead'           : 'died ☹',
                'date_diagnosed'      : 'was diagnosed with COVID',
                'date_end_quarantine' : 'ended quarantine',
                'date_infectious'     : 'became infectious',
                'date_known_contact'  : 'was notified they may have been exposed to COVID',
                'date_pos_test'       : 'recieved their positive test result',
                'date_quarantined'    : 'entered quarantine',
                'date_recovered'      : 'recovered',
                'date_severe'         : 'developed severe symptoms and needed hospitalization',
                'date_symptomatic'    : 'became symptomatic',
                'date_tested'         : 'was tested for COVID',
                'date_vaccinated'     : 'was vaccinated against COVID',
            }

            for attribute, message in dates.items():
                date = getattr(p,attribute)
                if not np.isnan(date):
                    events.append((date, message))

            for infection in self.infection_log:
                lkey = infection['layer']
                llabel = label_lkey(lkey)
                if infection['target'] == uid:
                    if lkey:
                        events.append((infection['date'], f'was infected with COVID by {infection["source"]} via the {llabel} layer'))
                    else:
                        events.append((infection['date'], 'was infected with COVID as a seed infection'))

                if infection['source'] == uid:
                    x = len([a for a in self.infection_log if a['source'] == infection['target']])
                    events.append((infection['date'],f'gave COVID to {infection["target"]} via the {llabel} layer ({x} secondary infections)'))

            if len(events):
                for day, event in sorted(events, key=lambda x: x[0]):
                    print(f'On day {day:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return

