import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
from covasim import utils as cvu


class store_seir(cv.Analyzer):

    keys_history = ["date_exposed", "date_infectious", 'date_symptomatic', 'date_severe', 'date_critical', 'date_recovered']
    number_recover_max_history = 3

    def __init__(self, show_all, save_people_history, *args, **kwargs):
        super().__init__(*args, **kwargs) # This is necessary to initialize the class properly
        self.bounds = np.linspace(0, 1.4701, 20)

        self.show_all = show_all
        self.people2contact_count = None
        self.state_statistics = None
        self.rel_sus_trans = None
        sim = None
        self.viral_load_by_layers = None
        self.exposed_by_sus = np.zeros(self.bounds.size, dtype=int)
        self.exposed_by_sus_per_day = np.zeros(self.bounds.size, dtype=int)
        self.exposed_by_sus_list = []
        self.exposed_by_sus_per_day_list = []
        #
        self.exposed_by_sus_by_ages = np.zeros(shape=(4, self.bounds.size), dtype=int)
        self.exposed_by_sus_by_ages_per_day = np.zeros(shape=(4, self.bounds.size), dtype=int)
        self.exposed_by_sus_by_ages_list = []
        self.exposed_by_sus_by_ages_per_day_list = []

        #
        self.ages_bounds_new = np.array([i * 10 for i in range(11)])
        self.naive_by_agegroup = np.zeros(self.ages_bounds_new.size - 1, dtype=int)
        self.naive_by_agegroup_list = []

        #

        self.ages_bounds = [0, 7, 25, 60, 100]
        #
        self.naive_by_sus_per_day = np.zeros(self.bounds.size, dtype=int)
        self.naive_by_sus_agegroup = np.zeros(shape=(self.ages_bounds_new.size - 1, self.bounds.size), dtype=int)
        self.naive_by_sus_list = []
        self.naive_by_sus_agegroup_list = []

        self.new_infections = []

        self.sizes_of_box = None
        self.sizes_of_box_by_ages = None
        self.sizes_ages_groups = None
        self.hist_sus = None

        self.nab_histograms = []
        self.immunity_histograms = []

        self.save_people_history = save_people_history
        self.people2history = None

        return
    
    def _init_sizes_of_box(self, ppl):
        self.rel_sus_trans = (ppl.rel_sus / np.max(ppl.rel_sus)) + (ppl.rel_trans / np.max(ppl.rel_trans))
        self.hist_sus = np.histogram(ppl.rel_sus)
        self.sizes_of_box = np.zeros(shape=self.bounds.shape, dtype=int)
        for i in range(self.bounds.size - 1):
            self.sizes_of_box[i] = np.count_nonzero(
                    (self.bounds[i] <= ppl.rel_sus) * (ppl.rel_sus < self.bounds[i + 1]))
        self.sizes_of_box += 1
        ##
        self.sizes_ages_groups = np.zeros(10, dtype=int)
        for i in range(self.ages_bounds_new.size - 1):
            self.sizes_ages_groups[i] = np.count_nonzero(
                (self.ages_bounds_new[i] <= ppl.age) * (ppl.age < self.ages_bounds_new[i + 1]))

        self.sizes_of_box_by_ages = np.zeros(shape=(4, self.bounds.size), dtype=int)
        for j in range(len(self.ages_bounds) - 1):
            for i in range(self.bounds.size - 1):
                self.sizes_of_box_by_ages[j][i] = np.count_nonzero(
                        (self.bounds[i] <= ppl.rel_sus) * (ppl.rel_sus < self.bounds[i + 1]) * (self.ages_bounds[j] <= ppl.age) * (ppl.age < self.ages_bounds[j + 1]))
            self.sizes_of_box_by_ages[j] += 1
        
        if self.save_people_history:
            self.people2history = np.zeros(shape=(self.number_recover_max_history, len(self.keys_history), len(ppl)))
            self.cur_index_people2history = np.zeros(shape=(len(ppl)), dtype=np.uint8) # update it on recovered
            self.inds_array = np.arange(len(ppl), dtype=int)

    def init_stats(self, sim):
        # never sick
        # exposed (in latent period)
        # asymptomatic (before recovering)
        # psesymptomatic (in asymp period)
        # mild
        # severe
        # critical
        self.state_history = np.zeros(shape=(7, sim.pars['n_days'] + 1))
        self.state_history_recovered = np.zeros(shape=(7, sim.pars['n_days'] + 1))

    def initialize(self, sim=None):
        if sim is None:
            raise RuntimeError("Sim in initialize is None!")
        
        ppl = sim.people
        self.people2contact_count = ppl.get_contact_statistics()
        self.viral_load_by_layers = sim.viral_load_by_layers
        self._init_sizes_of_box(ppl)
        self.init_stats(sim)
        #
        self.initialized = True
        self.finalized = False
        return

    def to_json(self):
        return {}

    def work_nab_histograms(self, sim):
        bins = np.array([0.001, 0.01, 0.05] + list(np.array(np.exp(np.arange(0.2, 8)), dtype=int)))
        def make_hist(nab):
            y, _ = np.histogram(nab, bins=bins)
            return (y, list(map(str, bins[:-1])))
        inds_alive = cvu.false(sim.people.dead)
        nab = sim.people.nab[inds_alive[cvu.true(sim.people.nab[inds_alive])]]
        nab_histogram = make_hist(nab)
        self.nab_histograms.append(nab_histogram)

    def work_immunity_histograms(self, sim):
        max_rel_sus = np.max(sim.people.rel_sus)
        bins = np.arange(0, max_rel_sus, max_rel_sus / 11)
        def make_hist(immunity):
            y, _ = np.histogram(immunity, bins=bins)
            return (y, 0.5 * (bins[:-1] + bins[1:]))
        immunity_histogram = make_hist(sim.people.sus_imm * sim.people.rel_sus)
        self.immunity_histograms.append(immunity_histogram)

    def update_state_history(self, sim):
        self.state_history[0][sim.t] = int(np.count_nonzero(sim.people.never_sick))
        self.state_history[1][sim.t] = np.count_nonzero(sim.people.exposed * (~sim.people.infectious))
        asymp_flag = sim.people.infectious * (~sim.people.symptomatic)
        self.state_history[2][sim.t] = np.count_nonzero(asymp_flag * sim.people.wont_symptomatic)
        self.state_history[3][sim.t] = np.count_nonzero(asymp_flag * sim.people.will_symptomatic)
        self.state_history[4][sim.t] = np.count_nonzero(sim.people.symptomatic * (~sim.people.severe) * (~sim.people.critical))
        self.state_history[5][sim.t] = np.count_nonzero(sim.people.severe * (~sim.people.critical))
        self.state_history[6][sim.t] = np.count_nonzero(sim.people.critical)


    def update_state_history_of_recovered(self, sim):
        inds_rec = sim.people.was_recovered
        self.state_history_recovered[0][sim.t] = np.count_nonzero(sim.people.never_sick[inds_rec])
        self.state_history_recovered[1][sim.t] = np.count_nonzero(sim.people.exposed[inds_rec] * (~sim.people.infectious[inds_rec]))
        asymp_flag = sim.people.infectious[inds_rec] * (~sim.people.symptomatic[inds_rec])
        self.state_history_recovered[2][sim.t] = np.count_nonzero(asymp_flag * sim.people.wont_symptomatic[inds_rec])
        self.state_history_recovered[3][sim.t] = np.count_nonzero(asymp_flag * sim.people.will_symptomatic[inds_rec])
        self.state_history_recovered[4][sim.t] = np.count_nonzero(sim.people.symptomatic[inds_rec] * (~sim.people.severe[inds_rec]) * (~sim.people.critical[inds_rec]))
        self.state_history_recovered[5][sim.t] = np.count_nonzero(sim.people.severe[inds_rec] * (~sim.people.critical[inds_rec]))
        self.state_history_recovered[6][sim.t] = np.count_nonzero(sim.people.critical[inds_rec])

    def update_people2history(self, sim):
        if not self.save_people_history:
            return 
        # work with exposed
        infected_inds = sim.people.inds_new_infections
        self.people2history[self.cur_index_people2history[infected_inds], 0, infected_inds] = sim.t
        self.people2history[self.cur_index_people2history[infected_inds], 1, infected_inds] = sim.people.date_infectious[infected_inds]
        self.people2history[self.cur_index_people2history[infected_inds], 2, infected_inds] = sim.people.date_symptomatic[infected_inds]
        self.people2history[self.cur_index_people2history[infected_inds], 3, infected_inds] = sim.people.date_severe[infected_inds]
        self.people2history[self.cur_index_people2history[infected_inds], 4, infected_inds] = sim.people.date_critical[infected_inds]
        self.people2history[self.cur_index_people2history[infected_inds], 5, infected_inds] = sim.people.date_recovered[infected_inds]
        self.cur_index_people2history[infected_inds] = np.clip(self.cur_index_people2history[infected_inds] + 1, 0, self.number_recover_max_history - 1)
        #print("_______")
        #print(self.people2history[0, :6, :10])
        #print("_______")


    def get_excel_people_history(self):
        import pandas as pd
        # Convert to spreadsheet
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        result_dfs = [pd.DataFrame(
            data={
                "exposed": self.people2history[i, 0],
                "infectious": self.people2history[i, 1],
                "symptomatic": self.people2history[i, 2],
                "severe": self.people2history[i, 3],
                "critical": self.people2history[i, 4],
                "recovered": self.people2history[i, 5],
            }
        ) for i in range(self.number_recover_max_history)]
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            for i in range(self.number_recover_max_history):
                result_dfs[i].to_excel(writer, sheet_name=f'Results time {i}')
        spreadsheet.load()
        return spreadsheet


    def apply(self, sim):
        ppl = sim.people
        self.new_infections.append(len(ppl.inds_new_infections))
        norm_rel_sus = ppl.rel_sus[ppl.inds_new_infections]
        naive_rel_sus = ppl.rel_sus[ppl.naive]
        naive_ages = ppl.age[ppl.naive]
        #
        norm_rel_sus_by_ages = []
        for i in range(len(self.ages_bounds) - 1):
            norm_rel_sus_by_ages.append(norm_rel_sus[(self.ages_bounds[i] <= ppl.age[ppl.inds_new_infections]) * (ppl.age[ppl.inds_new_infections] < self.ages_bounds[i + 1])])

        for i in range(self.ages_bounds_new.size - 1):
            self.naive_by_agegroup[i] = np.count_nonzero((self.ages_bounds_new[i] <= naive_ages) * (naive_ages < self.ages_bounds_new[i + 1]))

        for i in range(self.bounds.size - 1):
            self.exposed_by_sus_per_day[i] += np.count_nonzero((self.bounds[i] <= norm_rel_sus) * (norm_rel_sus < self.bounds[i + 1]))
            #
            self.naive_by_sus_per_day[i] = np.count_nonzero((self.bounds[i] <= naive_rel_sus) * (naive_rel_sus < self.bounds[i + 1]))
            for j in range(self.ages_bounds_new.size - 1):
                self.naive_by_sus_agegroup[j][i] = np.count_nonzero((self.bounds[i] <= naive_rel_sus) * (naive_rel_sus < self.bounds[i + 1]) *
                                                                         (self.ages_bounds_new[j] <= naive_ages) * (naive_ages < self.ages_bounds_new[j + 1]))
            #
            for j in range(len(self.ages_bounds) - 1):
                self.exposed_by_sus_by_ages_per_day[j][i] += np.count_nonzero((self.bounds[i] <= norm_rel_sus_by_ages[j]) * (norm_rel_sus_by_ages[j] < self.bounds[i + 1]))
        self.exposed_by_sus = self.exposed_by_sus + self.exposed_by_sus_per_day
        self.exposed_by_sus_by_ages = self.exposed_by_sus_by_ages + self.exposed_by_sus_by_ages_per_day

        self.exposed_by_sus_list.append(np.array(self.exposed_by_sus, copy=True))
        self.exposed_by_sus_per_day_list.append(np.array(self.exposed_by_sus_per_day, copy=True))
        self.exposed_by_sus_per_day = np.zeros(self.bounds.size, dtype=int)
        #
        self.naive_by_sus_list.append(np.array(self.naive_by_sus_per_day, copy=True))
        self.naive_by_sus_per_day = np.zeros(self.bounds.size, dtype=int)
        #
        self.exposed_by_sus_by_ages_per_day_list.append(np.array(self.exposed_by_sus_by_ages_per_day, copy=True))
        self.exposed_by_sus_by_ages_per_day = np.zeros(shape=(4, self.bounds.size), dtype=int)
        #
        self.naive_by_agegroup_list.append(np.array(self.naive_by_agegroup, copy=True))
        self.naive_by_sus_agegroup_list.append(np.array(self.naive_by_sus_agegroup, copy=True))
        #
        self.exposed_by_sus_by_ages_list.append(np.array(self.exposed_by_sus_by_ages, copy=True))

        # work with nab histograms
        self.work_nab_histograms(sim=sim)

        # work with immunity histograms
        self.work_immunity_histograms(sim=sim)

        # work with health status
        self.update_state_history(sim=sim)
        self.update_state_history_of_recovered(sim=sim)

        # work with storing dates
        self.update_people2history(sim=sim)

        return

    def right_hist(self, arr):
        unique_values, counts = np.unique(arr, return_counts=True)
        tmp_x = np.arange(np.max(unique_values) + 1)
        tmp_y = np.zeros(shape=tmp_x.size)
        tmp_y[unique_values] = counts
        return tmp_x, tmp_y

    def calc_contact_stats(self, sim):
        from covasim.analysis import TransTree
        transTree = TransTree(sim)
        cum_infectious = np.cumsum(self.new_infections)
        self.spread_count_stat = []
        n_days = sim.people.t # transTree.n_days
        for day in range(1, n_days + 1):
            tmp_counts_local = transTree.count_targets(start_day=day-1, end_day=day).astype(int)
            tmp_counts_global = transTree.count_targets(end_day=day).astype(int)
            if tmp_counts_local.size == 0:
                tmp_counts_local = np.array([0])
            if tmp_counts_global.size == 0:
                tmp_counts_global = np.array([0])
            
            self.neo_strike_numbers.append(self.right_hist(tmp_counts_local))
            was_source_number, number_people = self.right_hist(tmp_counts_global)
            self.neo_cum_strike_numbers.append((was_source_number, number_people))

            thr = int(0.8 * cum_infectious[day])
            spreaders_count = 0
            spreaders_infected_count = 0
            for i in range(was_source_number.size - 1, -1, -1):
                if thr == 0:
                    self.spread_count_stat.append(0)
                    break
                if spreaders_infected_count + was_source_number[i] * number_people[i] >= thr:
                    self.spread_count_stat.append((spreaders_count + int((thr - spreaders_infected_count) / was_source_number[i])) / cum_infectious[day])
                    break
                else:
                    spreaders_count += number_people[i]
                    spreaders_infected_count += (was_source_number[i] * number_people[i])
        self.spread_count_stat = np.array(self.spread_count_stat)

    def finalize(self, sim=None):
        import time

        if sim is None:
            raise RuntimeError("Sim in finalize is None!")

        self.neo_strike_numbers = []
        self.neo_cum_strike_numbers = []
        t0 = time.time()
        if self.show_all:
            self.calc_contact_stats(sim)
        t1 = time.time()
        self.state_statistics = sim.state_statistics
        # Calculate rs
        self.rs = []
        self.rs_based_testing = []
        timestep_r = 7
        prev_sum = np.sum(sim.results['new_infections'][0:timestep_r])
        prev_sum_bt = np.sum(sim.results['new_diagnoses_rpn'][0:timestep_r])
        for i in range(timestep_r, len(sim.results['new_infections']), timestep_r):
            cur_sum = np.sum(sim.results['new_infections'][i:(i+timestep_r)])
            cur_sum_bt = np.sum(sim.results['new_diagnoses_rpn'][i:(i+timestep_r)])
            cur_r = cur_sum / (prev_sum + 1)
            cur_r_bt = cur_sum_bt / (prev_sum_bt + 1)
            self.rs.append(cur_r)
            self.rs_based_testing.append(cur_r_bt)
            prev_sum = cur_sum
            prev_sum_bt = cur_sum_bt
        self.rs = np.array(self.rs)
        self.rs_based_testing = np.array(self.rs_based_testing)

        # calculate ars
        self.ars = []
        timestep_ar = 3
        prev_sum = np.sum(self.new_infections[0:timestep_ar])
        for i in range(timestep_ar, len(self.new_infections), timestep_ar):
            cur_sum = np.sum(self.new_infections[i:(i+timestep_ar)])
            susceptible_mean_on_week = np.mean(sim.susceptible_count_per_day[i:(i + timestep_ar)])
            self.ars.append(cur_sum / (susceptible_mean_on_week + 1) * 100.0)
            prev_sum = cur_sum
        self.ars = np.array(self.ars)

        # Calculate SAR
        zz = np.zeros((sim.viral_load_by_layers.shape), dtype=np.float)
        self.sars = np.divide(sim.viral_load_by_layers, sim.dangerous_contacts_count_by_layers,
                              out=zz, where=sim.dangerous_contacts_count_by_layers!=0)[:, :sim.people.t] * 100.0
        
        self.finalized = True
        return


    def plot(self):
        # normal_pos(mean=1.0, sigma=0.5)
        
        #fig = pl.figure(figsize=(40, 40))
        #ax_s = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
        #
        #time_range = np.arange(150)
        #for i in range(4):
        #    #print(np.histogram(sim.viral_load_by_layers[i]))
        #    ax_s[i].plot(time_range, sim.viral_load_by_layers[i])
        
        #pl.legend()
        #pl.xlabel('rel_sus')
        #pl.ylabel('risk(infected)')

        sc.setylim() # Reset y-axis to start at 0
        sc.commaticks() # Use commas in the y-axis labels
        pl.show()
        pl.savefig("sss" + '_viral_load_by_layers.pdf')
        return
