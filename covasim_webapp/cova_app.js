const PlotlyChart = {
    props: ['graph'],
    render(h) {
        return h('div', {
            attrs: {
                id: this.graph.id,
            }
        });
    },

    mounted() {
        this.$nextTick(function () {
            if (this.graph['json']){
                let x = JSON.parse(this.graph.json);
                x.responsive = true;
                Plotly.react(this.graph.id, x);
                window.dispatchEvent(new Event('resize'))
            }
        });
    },
    updated() {
        this.$nextTick(function () {
            if (this.graph['json']){
                let x = JSON.parse(this.graph.json);
                x.responsive = true;
                Plotly.react(this.graph.id, x);
            } else {
                Plotly.purge(this.graph.id)
            }
        });
    }
};

const infectiousTableConfig = {
    random_infectious: {
        formTitle: "Random",
        toolTip: "Parameters per random layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per random layer', value: 0.3}
        ],
        handleSubmit: function(event) {
            const beta_layer = vm.parse_day(event.target.elements.beta_layer.value);
            return {beta_layer};
        }
    },
    home_infectious: {
        formTitle: "Home",
        toolTip: "Parameters per home layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per home layer', value: 3.0}
        ],
        handleSubmit: function(event) {
            const beta_layer = vm.parse_day(event.target.elements.beta_layer.value);
            return {beta_layer};
        }
    },
    work_infectious: {
        formTitle: "Work",
        toolTip: "Parameters per work layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per work layer', value: 0.6}
        ],
        handleSubmit: function(event) {
            const beta_layer = vm.parse_day(event.target.elements.beta_layer.value);
            return {beta_layer};
        }
    },
    school_infectious: {
        formTitle: "School",
        toolTip: "Parameters per school layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per school layer', value: 0.6}
        ],
        handleSubmit: function(event) {
            const beta_layer = vm.parse_day(event.target.elements.beta_layer.value);
            return {beta_layer};
        }
    }
}

const interventionTableConfig = {
    social_distance: {
        formTitle: "Random",
        toolTip: "Physical distancing and social distancing interventions",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Impact of social distancing (examples: 20 = mild, 50 = moderate, 80 = aggressive)', min: 0, max: 100, value: 50}
        ],
        default_intervention_choice: 'Masks',
        default_age_group_choice: 'All',
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            const age_group_choice = event.target.elements.age_group_choice.value;
            return {start, end, level, intervention_choice, age_group_choice};
        }
    },
    school_closures: {
        formTitle: "Schools",
        toolTip: "School and university closures",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Impact of school closures (0 = no schools closed, 100 = all schools closed)', min: 0, max: 100, value: 90}
        ],
        default_intervention_choice: 'Masks',
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            return {start, end, level, intervention_choice};
        }
    },
    work_closures: {
        formTitle: "Works",
        toolTip: "Work",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Impact of work closures (0 = no works closed, 100 = all works closed)', min: 0, max: 100, value: 90}
        ],
        default_intervention_choice: 'Masks',
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            return {start, end, level, intervention_choice};
        }
    },
    home_closures: {
        formTitle: "Homes",
        toolTip: "Homes closures",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Impact of homes closures (0 = no homes closed, 100 = all homes closed)', min: 0, max: 100, value: 90}
        ],
        default_intervention_choice: 'Masks',
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            return {start, end, level, intervention_choice};
        }
    },
    vaccinate_closures: {
        formTitle: "Vaccination",
        toolTip: "Vaccination rates",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Probability', tooltip: 'Probability of being vaccinated (i.e., fraction of the population)', min: 0, max: 100, value: 0.3},
            {key: 'rel_sus_vaccine', type: 'number', label: 'Changing in susceptibility', tooltip: 'Relative change in susceptibility; 0 = perfect, 100 = no effect', min: 0, max: 100, value: 50},
            {key: 'rel_symp_vaccine', type: 'number', label: 'Changing in symptom probability', tooltip: 'Relative change in symptom probability for people who still get infected; 0 = perfect, 100 = no effect', min: 0, max: 100, value: 10},
            {key: 'min_age', type: 'number', label: 'Min age', tooltip: 'Min age bound', value: 0},
            {key: 'max_age', type: 'number', label: 'Max age', tooltip: 'Max age bound (leave blank for no end)', value: null}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            const rel_sus_vaccine = event.target.elements.rel_sus_vaccine.value;
            const rel_symp_vaccine = event.target.elements.rel_symp_vaccine.value;
            const min_age = event.target.elements.min_age.value;
            const max_age = vm.parse_max_age(event.target.elements.max_age.value);
            return {start, end, level, rel_sus_vaccine, rel_symp_vaccine, min_age, max_age};
        }
    },
    symptomatic_testing: {
        formTitle: "Testing",
        toolTip: "Testing rates for people with symptoms",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'alevel', type: 'number', label: 'Asymptomatic', tooltip: 'Probability of testing an asymptomatic (unquarantined) person(0 = no testing, 1 = 1% of asymptomatic people tested per day, 100 = everyone tested every day);', min: 0, max: 100, value: 1},
            {key: 'level', type: 'number', label: 'Symptomatic', tooltip: 'Probability of testing a symptomatic (unquarantined) person (0 = no testing, 10 = 10% of symptomatic people tested per day, 100 = everyone tested every day);', min: 0, max: 100, value: 10}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            const alevel = event.target.elements.alevel.value;
            return {start, end, level, alevel};
        }
    },
    contact_tracing: {
        formTitle: "Tracing",
        toolTip: "Contact tracing of diagnosed cases (requires testing intervention)",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Effectiveness of contact tracing (0 = no tracing, 100 = all contacts traced); assumes 1 day tracing delay. Please note: you must implement a testing intervention as well for tracing to have any effect.', min: 0, max: 100, value: 80}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    }

};

function copyright_year() {
    const release_year = 1999
    const current_year = new Date().getFullYear()
    let range = [release_year]

    if (current_year > release_year){
        range.push(current_year)
    }

    return range.join("-")
}

function generate_upload_file_handler(onsuccess, onerror) {
    return function(e){
            let files = e.target.files;
            if (files.length > 0){
                const data = new FormData();
                data.append('uploadfile', files[0])
                data.append('funcname', 'upload_file')
                data.append('args', undefined)
                data.append('kwargs', undefined)
                fetch("/api/rpcs", {
                  "body": data,
                  "method": "POST",
                  "mode": "cors",
                  "credentials": "include"
                }).then(response => {
                    if(!response.ok){
                        throw new Error(response.json())
                    }
                    return response.text()
                }).then(data => {
                    remote_filepath = data.trim()
                                          .replace(/["]/g, "")
                    onsuccess(remote_filepath)
                })
                .catch(error => {
                    if (onerror){
                        sciris.fail(this, "Could not upload file.", error)
                    } else {
                        onerror(error)
                    }
                })
            } else {
                console.warn("No input file selected.")
            }
        }
}



//var vm = new Vue({
//    el: '#app',
//
//    components: {
//        'plotly-chart': PlotlyChart,
//    },
//
//    data() {
//        return {
//            debug: false,
//            app: {
//                title: "NeoCovasim",
//                version: 'Unable to connect to server!', // This text will display instead of the version
//                copyright_year: copyright_year(),
//                copyright_owner: "Bill & Melinda Gates Foundation",
//                github_url: "https://github.com/IvanKozlov98/covasim",
//                org_url: "https://idmod.org",
//                docs_url: "http://docs.covasim.org",
//                paper_url: "http://paper.covasim.org",
//                publisher_url: "https://gatesfoundation.org",
//                license: 'Loading...',
//                notice: 'Loading...'
//            },
//            panel_open: true,
//            panel_width: null,
//            resizing: false,
//            history: [],
//            historyIdx: 0,
//            sim_length: {},
//            sim_pars: {},
//            epi_pars: {},
//            datafile: {
//                local_path: null,
//                server_path: null
//            },
//            int_pars: {},
//            intervention_figs: {},
//            show_animation: false,
//            result: { // Store currently displayed results
//                graphs: [],
//                summary: {},
//                files: {},
//            },
//            paramError: {},
//            scenarioError: {},
//            interventionTableConfig,
//            infectiousTableConfig,
//            running: false,
//            errs: [],
//            reset_options: ['Default', 'Optimistic', 'Pessimistic'],
//            reset_choice: 'Default',
//            infection_step_options: ['Covasim', 'Cumulative'],
//            infection_step_choice: 'Covasim',
//            rel_sus_options: [
//                'constants', 
//                'normal_pos',
//                'lognormal',
//                'neg_binomial_all',
//                'beta_1_all',
//                'lognormal_lite_all',
//                'beta_3_all',
//                'beta_2_all',
//                'normal_pos_all',
//                'lognormal_hot_all',
//                'uniform',
//                'uniform_all'
//            ],
//            rel_sus_choice: 'constants',
//            rel_trans_options: ['Dependent(sus)', 'Independent(sus)'],
//            rel_trans_choice: 'Independent(sus)',
//            population_volume_options: ['100K', '500K', '1M', '3M'],
//            population_volume_choice: '100K',
//            age_group_options: ['All', '0-7', '7-18', '18-25', '25-60', '60-'],
//            intervention_options: ['Masks', 'Removing contacts'],
//        };
//    },
//
//    created() {
//        this.get_version();
//        this.get_location_options();
//        this.resetPars();
//        this.watchSimLengthParam();
//        this.get_licenses();
//    },
//
//    filters: {
//        to2sf(value) {
//            return Number(value).toFixed(2);
//        }
//    },
//
//    computed: {
//        isRunDisabled: function () {
//            console.log(this.paramError);
//            return this.paramError && Object.keys(this.paramError).length > 0;
//        },
//        is_debug: function () {
//            return this.debug || /debug=true/i.test(window.location.search)
//        }
//    },
//
//    methods: {
//        async addIntervention(scenarioKey, event) {
//            const intervention = this.interventionTableConfig[scenarioKey].handleSubmit(event);
//            const key = scenarioKey;
//            const self = this
//            if (!this.int_pars[key]) {
//                this.$set(this.int_pars, key, []);
//            }
//            // validate intervention
//            const notValid = !intervention.end || intervention.start < 0 || intervention.end <= intervention.start
//            if (notValid) {
//                this.$set(this.scenarioError, scenarioKey, `Please enter a valid day range`);
//                return;
//            }
//
//            let overlaps = false;
//            const int_pars = this.int_pars[key];
//            for (let i = 0; i < int_pars.length; i++) {
//                const { start, end } = int_pars[i];
//                const is_time_overlap = ((start <= intervention.start && end >= intervention.start) ||
//                    (start <= intervention.end && end >= intervention.end) ||
//                    (intervention.start <= start && intervention.end >= end));
//                const is_has_interv_choice = 'intervention_choice' in intervention;
//                const is_has_ag_choice = 'age_group_choice' in intervention;
//                const is_random_interv = is_has_interv_choice && is_has_ag_choice;
//                const is_equal_interv_choice = intervention.intervention_choice == int_pars[i].intervention_choice;
//                const is_equal_ag_choice = intervention.age_group_choice == int_pars[i].age_group_choice || intervention.age_group_choice == "All" || int_pars[i].age_group_choice == "All";
//                
//                const cond_random = is_random_interv && (is_equal_interv_choice && is_equal_ag_choice);
//                const cond_interv_choice = !is_random_interv && is_has_interv_choice && is_equal_interv_choice
//                const cond_else = !is_has_interv_choice
//                if (is_time_overlap && (cond_random || cond_interv_choice || cond_else)) {
//                    if ('min_age' in intervention) {
//                        min_age_tmp = int_pars[i].min_age
//                        max_age_tmp = int_pars[i].max_age
//                        if ((min_age_tmp <= intervention.min_age && max_age_tmp >= intervention.min_age) ||
//                            (min_age_tmp <= intervention.max_age && max_age_tmp >= intervention.max_age) ||
//                            (intervention.min_age <= min_age_tmp && intervention.max_age >= max_age_tmp))
//                            {
//                                overlaps = true;
//                                break
//                            }
//                    } else {
//                        overlaps = true;
//                        break;
//                    }
//                }
//            }
//            if (overlaps){
//                this.$set(this.scenarioError, scenarioKey, `Interventions of the same type cannot have overlapping day ranges.`)
//                return ;
//            }
//
//            const outOfBounds = intervention.start > this.sim_length.best || intervention.end > this.sim_length.best || this.int_pars[key].some(({start, end}) => {
//                return start > self.sim_length.best || end > self.sim_length.best
//            })
//            if (outOfBounds){
//                this.$set(this.scenarioError, scenarioKey, `Intervention cannot start or end after the campaign duration.`)
//                return;
//            }
//            this.$set(this.scenarioError, scenarioKey, '');
//
//            this.int_pars[key].push(intervention);
//            const result = this.int_pars[key].sort((a, b) => a.start - b.start);
//            this.$set(this.int_pars, key, result);
//            const response = await sciris.rpc('get_gantt', undefined, {int_pars: this.int_pars, intervention_config: this.interventionTableConfig, n_days: this.sim_length.best});
//            this.intervention_figs = response.data;
//        },
//        async deleteIntervention(scenarioKey, index) {
//            this.$delete(this.int_pars[scenarioKey], index);
//            const response = await sciris.rpc('get_gantt', undefined, {int_pars: this.int_pars, intervention_config: this.interventionTableConfig});
//            this.intervention_figs = response.data;
//        },
//
//        parse_day(day) {
//            if (day == null || day == '') {
//                const output = this.sim_length.best
//                return output
//            } else {
//                const output = parseInt(day)
//                return output
//            }
//        },
//
//        parse_max_age(age) {
//            if (age == null || age == '') {
//                const output = 100
//                return output
//            } else {
//                const output = parseInt(age)
//                return output
//            }
//        },
//
//        resize_start() {
//            this.resizing = true;
//        },
//        resize_end() {
//            this.resizing = false;
//        },
//        resize_apply(e) {
//            if (this.resizing) {
//                // Prevent highlighting
//                e.stopPropagation();
//                e.preventDefault();
//                this.panel_width = (e.clientX / window.innerWidth) * 100;
//            }
//        },
//
//        dispatch_resize(){
//            window.dispatchEvent(new Event('resize'))
//        },
//        async get_version() {
//            const response = await sciris.rpc('get_version');
//            this.app.version = response.data;
//        },
//
//        async get_location_options() {
//            let response = await sciris.rpc('get_location_options');
//            for (let country of response.data) {
//                this.reset_options.push(country);
//            }
//        },
//
//        async get_licenses() {
//            const response = await sciris.rpc('get_licenses');
//            this.app.license = response.data.license;
//            this.app.notice = response.data.notice;
//        },
//
//        async runSim() {
//            this.running = true;
//            // this.graphs = this.$options.data().graphs; // Uncomment this to clear the graphs on each run
//            this.errs = this.$options.data().errs;
//
//            console.log('status:', this.status);
//
//            // Run a a single sim
//            try {
//                if(this.datafile.local_path === null){
//                    this.reset_datafile()
//                }
//                const kwargs = {
//                    sim_pars: this.sim_pars,
//                    epi_pars: this.epi_pars,
//                    int_pars: this.int_pars,
//                    datafile: this.datafile.server_path,
//                    show_animation: this.show_animation,
//                    n_days: this.sim_length.best,
//                    location: this.reset_choice,
//                    infection_step: this.infection_step_choice,
//                    rel_sus_type: this.rel_sus_choice,
//                    rel_trans_type: this.rel_trans_choice,
//                    population_volume: this.population_volume_choice,
//                    infectiousTableConfig: this.infectiousTableConfig
//                }
//                console.log('run_sim: ', kwargs);
//                const response = await sciris.rpc('run_sim', undefined, kwargs);
//                this.result.graphs = response.data.graphs;
//                this.result.files = response.data.files;
//                this.result.summary = response.data.summary;
//                this.errs = response.data.errs;
//                // this.panel_open = this.errs.length > 0; // Better solution would be to have a pin button
//                this.sim_pars = response.data.sim_pars;
//                this.epi_pars = response.data.epi_pars;
//                this.int_pars = response.data.int_pars;
//                this.history.push(JSON.parse(JSON.stringify({ sim_pars: this.sim_pars, epi_pars: this.epi_pars, reset_choice: this.reset_choice, infection_step_choice: this.infection_step_choice, rel_sus_choice: this.rel_sus_choice, rel_trans_choice: this.rel_trans_choice, population_volume_choice: this.population_volume_choice, int_pars: this.int_pars, result: this.result })));
//                this.historyIdx = this.history.length - 1;
//
//            } catch (e) {
//                this.errs.push({
//                    message: 'Unable to submit model.',
//                    exception: `${e.constructor.name}: ${e.message}`
//                })
//                this.panel_open = true
//            }
//            this.running = false;
//
//        },
//
//        async clearSims() {
//            this.errs = this.$options.data().errs;
//
//            try {
//                await sciris.rpc('clear_sims', undefined);
//                this.result.graphs = [];
//            } catch (e) {
//                this.errs.push({
//                    message: 'Unable to clear graphs.',
//                    exception: `${e.constructor.name}: ${e.message}`
//                })
//                this.panel_open = true
//            }
//        },
//
//        async resetPars() {
//            const response = await sciris.rpc('get_defaults', [this.reset_choice]);
//            this.sim_pars = response.data.sim_pars;
//            this.epi_pars = response.data.epi_pars;
//            this.sim_length = this.sim_pars['n_days'];
//            this.int_pars = {};
//            this.intervention_figs = {};
//            this.setupFormWatcher('sim_pars');
//            this.setupFormWatcher('epi_pars');
//            // this.result.graphs = [];
//            this.reset_datafile()
//        },
//
//        setupFormWatcher(paramKey) {
//            const params = this[paramKey];
//            if (!params) {
//                return;
//            }
//            Object.keys(params).forEach(key => {
//                this.$watch(`${paramKey}.${key}`, this.validateParam(key), { deep: true });
//            });
//        },
//
//        watchSimLengthParam() {
//            this.$watch('sim_length', this.validateParam('sim_length'), { deep: true });
//        },
//
//        validateParam(key) {
//            return (param) => {
//                if (param.best <= param.max && param.best >= param.min) {
//                    this.$delete(this.paramError, key);
//                } else {
//                    this.$set(this.paramError, key, `Please enter a number between ${param.min} and ${param.max}`);
//                }
//            };
//        },
//
//        async downloadPars() {
//            const d = new Date();
//            const datestamp = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}_${d.getHours()}.${d.getMinutes()}.${d.getSeconds()}`;
//            const fileName = `covasim_parameters_${datestamp}.json`;
//
//            // Adapted from https://stackoverflow.com/a/45594892 by Gautham
//            const data = {
//                sim_pars: this.sim_pars,
//                epi_pars: this.epi_pars,
//                int_pars: this.int_pars
//            };
//            const fileToSave = new Blob([JSON.stringify(data, null, 4)], {
//                type: 'application/json',
//                name: fileName
//            });
//            saveAs(fileToSave, fileName);
//        },
//
//        async uploadPars() {
//            try {
//                const response = await sciris.upload('upload_pars');  //, [], {}, '');
//                this.sim_pars = response.data.sim_pars;
//                this.epi_pars = response.data.epi_pars;
//                this.int_pars = response.data.int_pars;
//                this.result.graphs = [];
//                this.intervention_figs = {}
//
//                if (this.int_pars){
//                    const gantt = await sciris.rpc('get_gantt', undefined, {int_pars: this.int_pars, intervention_config: this.interventionTableConfig});
//                    this.intervention_figs = gantt.data;
//                }
//
//            } catch (error) {
//                sciris.fail(this, 'Could not upload parameters', error);
//            }
//        },
//        upload_datafile: generate_upload_file_handler(function(filepath){
//            vm.datafile.server_path = filepath
//        }),
//
//        reset_datafile() {
//            this.datafile = {
//                local_path: null,
//                server_path: null
//            }
//        },
//
//        loadPars() {
//            this.sim_pars = this.history[this.historyIdx].sim_pars;
//            this.epi_pars = this.history[this.historyIdx].epi_pars;
//            this.reset_choice = this.history[this.historyIdx].reset_choice;
//            this.int_pars = this.history[this.historyIdx].int_pars;
//            this.result = this.history[this.historyIdx].result;
//        },
//
//        async downloadExcel() {
//            const res = await fetch(this.result.files.xlsx.content);
//            const blob = await res.blob();
//            saveAs(blob, this.result.files.xlsx.filename);
//        },
//
//        async downloadJson() {
//            const res = await fetch(this.result.files.json.content);
//            const blob = await res.blob();
//            saveAs(blob, this.result.files.json.filename);
//        },
//
//    },
//
//});
//