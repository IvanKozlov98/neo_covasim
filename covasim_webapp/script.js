Vue.component('my-plotly-chart', {
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
            if (this.graph['json']) {
                let x = JSON.parse(this.graph.json);
                x.responsive = true;
                Plotly.react(this.graph.id, x);
            } else {
                Plotly.purge(this.graph.id)
            }
        });
    }
});

const infectiousTableConfig = {
    random_infectious: {
        formTitle: "Random",
        toolTip: "Parameters per random layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per random layer', value: Array(20).fill(0.3)}
        ],
    },
    home_infectious: {
        formTitle: "Home",
        toolTip: "Parameters per home layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per home layer', value: Array(20).fill(3.0)}
        ],
    },
    work_infectious: {
        formTitle: "Work",
        toolTip: "Parameters per work layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per work layer', value: Array(20).fill(0.6)}
        ],
    },
    school_infectious: {
        formTitle: "School",
        toolTip: "Parameters per school layer",
        fields: [
            {key: 'beta_layer', type: 'number', label: 'Beta', tooltip: 'Transmissibility per school layer', value: Array(20).fill(0.6)}
        ],
    }
}

const interventionTableConfig = {
    social_distance: {
        formTitle: "Random",
        toolTip: "Physical distancing and social distancing interventions",
        fields: [
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', step: "0.01", label: 'Effectiveness', tooltip: 'Impact of social distancing (examples: 20 = mild, 50 = moderate, 80 = aggressive)', min: 0, max: 100, value: 50}
        ],
        default_intervention_choice: 'Masks',
        default_age_group_choice: 'All',
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
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
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', step: "0.01", label: 'Effectiveness', tooltip: 'Impact of school closures (0 = no schools closed, 100 = all schools closed)', min: 0, max: 100, value: 90}
        ],
        default_intervention_choice: 'Masks',
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            return {start, end, level, intervention_choice};
        }
    },
    work_closures: {
        formTitle: "Works",
        toolTip: "Work",
        fields: [
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', step: "0.01", label: 'Effectiveness', tooltip: 'Impact of work closures (0 = no works closed, 100 = all works closed)', min: 0, max: 100, value: 90}
        ],
        default_intervention_choice: 'Masks',
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            return {start, end, level, intervention_choice};
        }
    },
    home_closures: {
        formTitle: "Homes",
        toolTip: "Homes closures",
        fields: [
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', step: "0.01", label: 'Effectiveness', tooltip: 'Impact of homes closures (0 = no homes closed, 100 = all homes closed)', min: 0, max: 100, value: 90}
        ],
        default_intervention_choice: 'Masks',
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
            const level = event.target.elements.level.value;
            const intervention_choice = event.target.elements.intervention_choice.value;
            return {start, end, level, intervention_choice};
        }
    },
    vaccinate_closures: {
        formTitle: "Vaccination",
        toolTip: "Vaccination rates",
        fields: [
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', step: "0.01", label: 'Probability', tooltip: 'Probability of being vaccinated (i.e., fraction of the population)', min: 0, max: 100, value: 0.3},
            {key: 'min_age', type: 'number', step: "1", label: 'Min age', tooltip: 'Min age bound', value: 0},
            {key: 'max_age', type: 'number', step: "1", label: 'Max age', tooltip: 'Max age bound (leave blank for no end)', value: null}
        ],
        default_vaccine_choice: 'pfizer',
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
            const level = event.target.elements.level.value;
            const min_age = event.target.elements.min_age.value;
            const max_age = vm.parse_max_age(event.target.elements.max_age.value);
            return {start, end, level, min_age, max_age};
        },
        handleDop: function(prob) {
            this.fields['level'] = prob;
        }
    },
    symptomatic_testing: {
        formTitle: "Testing",
        toolTip: "Testing rates for people with symptoms",
        fields: [
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'alevel', type: 'number', step: "0.01", label: 'Asymptomatic', tooltip: 'Probability of testing an asymptomatic (unquarantined) person(0 = no testing, 1 = 1% of asymptomatic people tested per day, 100 = everyone tested every day);', min: 0, max: 100, value: 1},
            {key: 'level', type: 'number', step: "0.01", label: 'Symptomatic', tooltip: 'Probability of testing a symptomatic (unquarantined) person (0 = no testing, 10 = 10% of symptomatic people tested per day, 100 = everyone tested every day);', min: 0, max: 100, value: 10}
        ],
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
            const level = event.target.elements.level.value;
            const alevel = event.target.elements.alevel.value;
            return {start, end, level, alevel};
        }
    },
    contact_tracing: {
        formTitle: "Tracing",
        toolTip: "Contact tracing of diagnosed cases (requires testing intervention)",
        fields: [
            {key: 'start', type: 'number', step: "1", label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', step: "1", label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', step: "0.01", label: 'Effectiveness', tooltip: 'Effectiveness of contact tracing (0 = no tracing, 100 = all contacts traced); assumes 1 day tracing delay. Please note: you must implement a testing intervention as well for tracing to have any effect.', min: 0, max: 100, value: 80}
        ],
        handleSubmit: function(event, city_ind) {
            const start = vm.parse_day(event.target.elements.start.value, city_ind);
            const end = vm.parse_day(event.target.elements.end.value, city_ind);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    }

};

const interactionTableConfig = {
    field: {key: 'level', type: 'number', step: "0.01", label: 'Effectiveness', tooltip: 'Percentage of agents that move from \'From City\' to \'To City\' ', min: 0, max: 0.005, value: 0.001},

    handleSubmit: function(event) {
        console.log("in handle submit");
        const from_city_choice = event.target.elements.from_city_choice.value;
        const to_city_choice = event.target.elements.to_city_choice.value;
        const level = event.target.elements.level.value;
        return {from_city_choice, to_city_choice, level};
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

function clone(obj) {
    if (obj === null || typeof (obj) !== 'object' || 'isActiveClone' in obj)
        return obj;

    if (obj instanceof Date)
        var temp = new obj.constructor(); //or new Date(obj);
    else
        var temp = obj.constructor();

    for (var key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            obj['isActiveClone'] = null;
            temp[key] = clone(obj[key]);
            delete obj['isActiveClone'];
        }
    }
    return temp;
}

function changeOnlyOne(lst, ind, obj) {
    copy_lst = clone(lst);
    copy_lst[ind] = obj;
    return copy_lst;
}


var vm = new Vue({
    el: '#app',
    //template: '#tab-component-template',

    data() {
        return {
            filtered: Array.from({ length: 20 }, () => ([
                { 
                  "id": "wild",
                  "wild": "1",
                }
            ])),
            fields: Array.from({ length: 20 }, () => ([
                { key: 'id', label: '' },
                { key: 'wild', label: 'wild', editable: true },
            ])),
            currentTime: 0,
            activeTabInd: 0,
            groupHides: {
                'Incidence and outcomes': true,
                'General spread parameters': false,
                'Immunity': false,
                'Spread parameters by layer': false,
                'Spread parameters by age': false,
                'Rest': false
            },
            debug: false,
            tabs: [], 
            city_options: [],
            tabCounter: 0,
            interactionRecords: [],
            app: {
                title: "NeoCovasim",
                version: 'Unable to connect to server!', // This text will display instead of the version
                copyright_year: copyright_year(),
                copyright_owner: "Bill & Melinda Gates Foundation",
                github_url: "https://github.com/IvanKozlov98/covasim",
                org_url: "https://idmod.org",
                docs_url: "http://docs.covasim.org",
                paper_url: "http://paper.covasim.org",
                publisher_url: "https://gatesfoundation.org",
                license: 'Loading...',
                notice: 'Loading...'
            },
            panel_open: true,
            panel_width: null,
            resizing: false,
            history: [],
            historyIdx: 0,
            sim_length: {},
            sim_pars: {},
            epi_pars: {},
            datafile: {
                local_path: null,
                server_path: null
            },
            int_pars: Array(20).fill({ }),
            intervention_figs: Array(20).fill({ }),
            rel_sus_figs: {},
            cur_rel_sus_fig: [],
            multiple_cities: false,
            show_contact_stat: false,
            progresss: 70,
            result: { // Store currently displayed results
                graphs: [],
                summary: {},
                files: {},
            },
            paramError: {},
            scenarioError: {},
            multipleCitiesError: "",
            interventionTableConfig,
            infectiousTableConfig,
            interactionTableConfig,
            running: false,
            errs: [],
            reset_options: ['Default', 'Optimistic', 'Pessimistic'],
            reset_choice: 'Default',
            infection_step_options: ['Covasim', 'Cumulative'],
            infection_step_choice_list: Array.from({ length: 20 }, () => ('Covasim')),
            rel_sus_options: [
                "Constant (Covasim default)", 
                'Normal by age',
                'Lognormal by age',
                //'neg_binomial_all',
                'Beta 1 all',
                'Lognormal lite all',
                'Beta 3 all',
                //'beta_2_all',
                'Normal all',
                'Lognormal hot all',
                'Constant all',
                //'uniform_all'
            ],
            rel_sus_choice_list: Array.from({ length: 20 }, () => ('Constant (Covasim default)')),
            
            default_cross_immunity: null,
            vaccine_options: [
                'pfizer',
                'moderna',
                'az',
                'jj',
                'novavax',
                'sinovac',
                'sinopharm'
            ],
            nab_init_par1_vaccine_list: Array.from({ length: 20 }, () => (-1)),
            nab_init_par2_vaccine_list: Array.from({ length: 20 }, () => (2)),
            nab_boost_vaccine_list: Array.from({ length: 20 }, () => (8)),
            doses_vaccine_list: Array.from({ length: 20 }, () => (2)),
            interval_vaccine_list: Array.from({ length: 20 }, () => (28)),
            vaccine_choice_list: Array.from({ length: 20 }, () => ('pfizer')),
            
            variant_options: [
                'wild',
                'alpha',
                'beta',
                'gamma',
                'delta',
                'custom'
            ],
            variant_choice_list: Array.from({ length: 20 }, () => ('wild')),
            parameters_by_variant: {
                "possible_name": {
                    'description': 'Name of variant',
                    'name': 'Name:',
                    'data': Array.from({ length: 20 }, () => ('grp')),
                },
                "rel_beta": 
                {
                    'description': 'Relative transmissibility varies by variant',
                    'name': 'Transmisibilty:',
                    'data': Array.from({ length: 20 }, () => (1.0)),
                },
                "rel_symp_prob": 
                {
                    'description': 'Scale factor for proportion of symptomatic cases',
                    'name': 'Symptomatic factor:',
                    'data': Array.from({ length: 20 }, () => (1.0))
                },
                "rel_severe_prob": 
                {
                    'description': 'Scale factor for proportion of symptomatic cases that become severe',
                    'name': 'Severe factor:',
                    'data': Array.from({ length: 20 }, () => (1.0))
                },
                "rel_crit_prob": 
                {
                    'description': 'Scale factor for proportion of severe cases that become critical',
                    'name': 'Critical factor',
                    'data': Array.from({ length: 20 }, () => (1.0))
                },
                "rel_death_prob": 
                {
                    'description': 'Scale factor for proportion of critical cases that result in death',
                    'name': 'Death factor',
                    'data': Array.from({ length: 20 }, () => (1.0))
                },
                "dur_exp2inf": 
                {
                    'description': 'Duration from exposed to infectious',
                    'name': 'Latent period(days)',
                    'data': Array.from({length: 20}, () => (4.5))
                },
                "dur_inf2sym": 
                {
                    'description': 'Duration from infectious to symptomatic',
                    'name': 'Asymptomatic period (days):',
                    'data': Array.from({length: 20}, () => (1.1))
                },
                "dur_sym2sev": 
                {
                    'description': 'Duration from symptomatic to severe symptoms',
                    'name': 'From symp to severe (days)',
                    'data': Array.from({length: 20}, () => (6.6))
                },
                "dur_sev2crit": 
                {
                    'description': 'Duration from severe symptoms to requiring ICU; average of 1.9 and 1.0',
                    'name': 'From severe to critical (days)',
                    'data': Array.from({length: 20}, () => (1.5))
                },
                "dur_asym2rec": 
                {
                    'description': 'Duration for asymptomatic people to recover',
                    'name': 'Recovery of asymp (days)',
                    'data': Array.from({length: 20}, () => (8.0))
                },
                "dur_mild2rec": 
                {
                    'description': 'Duration for people with mild symptoms to recover',
                    'name': 'Recovery of mild (days)',
                    'data': Array.from({length: 20}, () => (8.0))
                },
                "dur_sev2rec": 
                {
                    'description': 'Duration for people with severe symptoms to recover',
                    'name': 'Recovery of severe (days)',
                    'data': Array.from({length: 20}, () => (18.1))
                },
                "dur_crit2rec": 
                {
                    'description': 'Duration for people with critical symptoms to recover',
                    'name': 'Recovery of critical (days)',
                    'data': Array.from({length: 20}, () => (18.1))
                },
                "dur_crit2die": 
                {
                    'description': 'Duration from critical symptoms to death',
                    'name': 'Time until death (days):',
                    'data': Array.from({length: 20}, () => (10.7))
                },
                "start_day": 
                {
                    'description': 'Start day of introducing virus',
                    'name': 'Start day:',
                    'data': Array.from({ length: 20 }, () => (0))
                },
                "n_import": 
                {
                    'description': 'The number of imports of the variant to be added',
                    'name': 'Number of imported:',
                    'data': Array.from({ length: 20 }, () => (10))
                },
                "oral_microbiota_percent": {
                    'description': "TODO",
                    "name": "Oral microbiota percent",
                    'data': Array.from({ length: 20 }, () => (0.0))
                },
                "oral_microbiota_factor": {
                    'description': "TODO",
                    "name": "Oral microbiota severe factor",
                    'data': Array.from({ length: 20 }, () => (1.0))
                },
            },
            introduced_variants_list: Array.from({ length: 20 }, () => ([{
                'variant_name': 'wild',
                "rel_beta": 1.0,
                "rel_symp_prob": 1.0,
                "rel_severe_prob": 1.0,
                "rel_crit_prob": 1.0,
                "rel_death_prob": 1.0,
                "dur_exp2inf": 4.5,
                "dur_inf2sym": 1.1,
                "dur_sym2sev": 6.6,
                "dur_sev2crit": 1.5,
                "dur_asym2rec": 8.0,
                "dur_mild2rec": 8.0,
                "dur_sev2rec": 18.1,
                "dur_crit2rec": 18.1,
                "dur_crit2die": 10.7,
                "start_day": 0,
                "n_import": 10,
                "oral_microbiota_percent": 0.0,
                "oral_microbiota_factor": 1.0
            }])),
            variant_figs: Array(20).fill({ }),

            rel_trans_options: ['Dependent(sus)', 'Independent(sus)'],
            rel_trans_choice_list: Array.from({ length: 20 }, () => ('Independent(sus)')),
            population_volume_options: ['100K', '100K(Random)', '500K', '1M', '3M'],
            population_volume_choice_list: Array.from({ length: 20 }, () => ('100K')),
            age_group_options: ['All', '0-7', '7-18', '18-25', '25-60', '60-'],
            intervention_options: ['Masks', 'Removing contacts'],
        };
    },

    created() {
        this.get_version();
        this.get_location_options();
        this.resetPars();
        //this.watchSimLengthParam();
        this.get_licenses();
        if (this.tabs.length === 0) {
            this.newTab();
        }
    },

    filters: {
        to2sf(value) {
            return Number(value).toFixed(2);
        }
    },

    computed: {
        isRunDisabled: function () {
            return this.paramError && Object.keys(this.paramError).length > 0;
        },
        is_debug: function () {
            return this.debug || /debug=true/i.test(window.location.search);
        },
        editableFields() {
            var res = [];
            for(var i = 0; i < 20; i++) {
                res.push(this.fields[i].filter(field => field.editable))
            }
            return res;
        }
    },

    methods: {    
        someFun(kk) {
            this.activeTabInd = kk;
        },

        closeTab(x) {
            if (this.tabs.length === 1)
                return;
            for (let i = 0; i < this.tabs.length; i++) {
              if (this.tabs[i] === x) {
                this.tabs.splice(i, 1);
                this.city_options.splice(i, 1);
              }
            }
          },
        
        updateVariantTable(newInd, oldInd) {
            this.deleteVariantFromTable(newInd, 'wild');
            console.log(this.introduced_variants_list[oldInd]);
            for(var i in this.introduced_variants_list[oldInd]) {
                const variant_name = this.introduced_variants_list[oldInd][parseInt(i)]['variant_name'];
                this.addVariantToTable(newInd, variant_name);
                
            }
            // then copy values
            for(let i = 0; i < this.filtered[newInd].length; i++) {
                for(var k in this.filtered[oldInd][i]) {
                    this.filtered[newInd][i][k] = this.filtered[oldInd][i][k];
                }
            }
            console.log("update variant table finish");
        },

        async copyTab(oldInd, newInd) {
            this.int_pars[newInd] = clone(this.int_pars[oldInd]);
            if (newInd !== oldInd) {
                this.updateVariantTable(newInd, oldInd);                
            }

            this.cur_rel_sus_fig[newInd] = clone(this.cur_rel_sus_fig[oldInd]);
            this.introduced_variants_list[newInd] = clone(this.introduced_variants_list[oldInd]);
            this.variant_figs[newInd] = clone(this.variant_figs[oldInd]); 
            this.infection_step_choice_list[newInd] = clone(this.infection_step_choice_list[oldInd]); 
            this.rel_sus_choice_list[newInd] = clone(this.rel_sus_choice_list[oldInd]); 
            this.vaccine_choice_list[newInd] = clone(this.vaccine_choice_list[oldInd]); 
            this.variant_choice_list[newInd] = clone(this.variant_choice_list[oldInd]); 
            this.rel_trans_choice_list[newInd] = clone(this.rel_trans_choice_list[oldInd]); 
            this.population_volume_choice_list[newInd] = clone(this.population_volume_choice_list[oldInd]);
            // copy sim_pars
            const sim_pars_keys = Object.keys(this.sim_pars);
            for (const key of sim_pars_keys)
              if (key != "n_days")
                this.sim_pars[key]['best'][newInd] = this.sim_pars[key]['best'][oldInd];
            // copy epi_pars
            const epi_pars_keys = Object.keys(this.epi_pars);
            for (const key of epi_pars_keys){
                this.epi_pars[key]['best'][newInd] = this.epi_pars[key]['best'][oldInd];
            }
          },

        async newTab() {
            const city_ind = this.tabCounter;
            this.city_options.push('City ' + city_ind);
            this.copyTab(this.activeTabInd, city_ind);
            this.handleSimpleCase(city_ind);
            this.tabs.push(city_ind);
            const response = await sciris.rpc('get_gantt', undefined, {int_pars_list: this.int_pars, intervention_config: this.interventionTableConfig, n_days: this.sim_length.best, tabs: this.tabs});
            this.intervention_figs = response.data;
            const response_variant = await sciris.rpc('get_gantt_variant', undefined, {introduced_variants_list: this.introduced_variants_list, n_days: this.sim_length.best, tabs: this.tabs});
            this.variant_figs = response_variant.data;
            this.tabCounter = this.tabCounter + 1;
          },

        async handleSimpleCase() {
            const response = await sciris.rpc('get_dist_figs', undefined, {rel_sus_choice_list: this.rel_sus_choice_list, tabs: this.tabs});
            this.cur_rel_sus_fig = response.data;
        },

        async handleChangeVariant(city_ind) {
            const city_ind_int = parseInt(city_ind);
            var variant_name = this.variant_choice_list[parseInt(city_ind)];
            if (variant_name === 'custom') {
                variant_name = 'wild';
            }
            const response = await sciris.rpc('get_variant_pars', undefined, {variant_choice: variant_name});
            for (var key in this.parameters_by_variant) {
                var new_val;
                switch (key) {
                    case "start_day": new_val = 0; break;
                    case "n_import": new_val = 10; break;
                    case "oral_microbiota_percent": new_val = 0.0; break;
                    case "oral_microbiota_factor": new_val = 1.0; break;
                    default: new_val = response.data[key]; break;
                }
                console.log(key);
                console.log(this.parameters_by_variant[key]);
                console.log(this.parameters_by_variant[key]['data']);
                console.log(this.parameters_by_variant[key].data);
                this.parameters_by_variant[key].data = changeOnlyOne(this.parameters_by_variant[key].data, city_ind_int, new_val);
            }        
        },

        async addVariantToTable(city_ind_int, variant_name) {
            // edit table
            var newEntry = {};
            newEntry['id'] = variant_name; 
            for (let i = 1; i < this.fields[city_ind_int].length; i++) {
                const variant_name_op = this.fields[city_ind_int][i]['key'];
                if (variant_name_op in this.default_cross_immunity && variant_name in this.default_cross_immunity)
                    newEntry[variant_name_op] = this.default_cross_immunity[variant_name_op][variant_name];
                else
                    newEntry[variant_name_op] = 0.0;
            }
            newEntry[variant_name] = "1";
            this.filtered[city_ind_int].push(newEntry);
            for (let i = 0; i < this.filtered[city_ind_int].length; i++) {
                if (this.filtered[city_ind_int][i]['id'] !== variant_name){
                    const variant_name_op = this.filtered[city_ind_int][i]['id'];
                    if (variant_name in this.default_cross_immunity && variant_name_op in this.default_cross_immunity) {
                        this.filtered[city_ind_int][i][variant_name] = this.default_cross_immunity[variant_name][variant_name_op];
                    } else {
                        this.filtered[city_ind_int][i][variant_name] = 0.0;
                    }
                }
            }
            this.fields[city_ind_int].push({ key: variant_name, label: variant_name, editable: true });
            //
        },

        async addVariant(city_ind) {
            const city_ind_int = parseInt(city_ind);
            var slice_of_variant_par = {};
            var variant_name = "";
            if (this.variant_choice_list[city_ind_int] === "custom") {
                variant_name = this.parameters_by_variant['possible_name'].data[city_ind_int]
            } else {
                variant_name = this.variant_choice_list[city_ind_int];
            }
            slice_of_variant_par['variant_name'] = variant_name;
            for (var key in this.parameters_by_variant) {
                slice_of_variant_par[key] = this.parameters_by_variant[key].data[city_ind_int];
            }
            if (this.default_cross_immunity === null) {
                console.log("Here");
                default_cross_immunity_resp = await sciris.rpc('get_default_cross_immunity', undefined, {}); 
                this.default_cross_immunity = default_cross_immunity_resp.data; 
                console.log(this.default_cross_immunity);
                console.log("After");      
            }
            this.addVariantToTable(city_ind_int, variant_name);
            this.introduced_variants_list[city_ind_int].push(slice_of_variant_par);
            const response_variant = await sciris.rpc('get_gantt_variant', undefined, {introduced_variants_list: this.introduced_variants_list, n_days: this.sim_length.best, tabs: this.tabs});
            this.variant_figs = response_variant.data;
        },

        async deleteVariantFromTable(city_ind_int, variant_name) {
            // remove variant from table
            var index_removing = 0;
            for (let i = 0; i < this.fields[city_ind_int].length; i++) {
                if (this.fields[city_ind_int][i]['key'] === variant_name) {
                    index_removing = i;
                    console.log("find");
                    break;
                } else {
                    console.log("------");
                    console.log(this.fields[city_ind_int][i]['key']);
                    console.log(variant_name);
                    console.log("------");
                }
            }
            console.log(this.filtered[city_ind_int]);
            this.fields[city_ind_int].splice(index_removing, 1);
            this.filtered[city_ind_int].splice(index_removing - 1, 1);
            for (let i = 0; i < this.filtered[city_ind_int].length; i++) {
                delete this.filtered[city_ind_int][i][variant_name];
            }
            console.log(index_removing);
            console.log(this.filtered);
            //
        },


        async deleteVariant(city_ind, index) {
            const city_ind_int = parseInt(city_ind);
            const variant_name = this.introduced_variants_list[city_ind_int][index]['variant_name'];
            this.deleteVariantFromTable(city_ind_int, variant_name);
            this.introduced_variants_list[city_ind_int].splice(index, 1);
            const response_variant = await sciris.rpc('get_gantt_variant', undefined, {introduced_variants_list: this.introduced_variants_list, n_days: this.sim_length.best, tabs: this.tabs});
            this.variant_figs = response_variant.data;
        },

        async handleChangeVaccine(city_ind) {
            const city_ind_int = parseInt(city_ind);
            const response = await sciris.rpc('get_vaccine_pars', undefined, {vaccine_choice: this.vaccine_choice_list[parseInt(city_ind)]});
            this.nab_init_par1_vaccine_list = changeOnlyOne(this.nab_init_par1_vaccine_list, city_ind_int, response.data['nab_init_par1']);
            this.nab_init_par2_vaccine_list = changeOnlyOne(this.nab_init_par2_vaccine_list, city_ind_int, response.data['nab_init_par2']);
            this.nab_boost_vaccine_list = changeOnlyOne(this.nab_boost_vaccine_list, city_ind_int, response.data['nab_boost']);
            this.doses_vaccine_list = changeOnlyOne(this.doses_vaccine_list, city_ind_int, response.data['doses']);
            this.interval_vaccine_list = changeOnlyOne(this.interval_vaccine_list, city_ind_int, response.data['interval']);
        },

        async addInteraction(event) {
            const interaction = this.interactionTableConfig.handleSubmit(event);
            if (!(interaction.from_city_choice == interaction.to_city_choice))
                this.interactionRecords.push(interaction);
            else 
                multipleCitiesError = "Enter different cities";
        },

        async deleteInteraction(index) {
            this.interactionRecords.splice(index, 1);
        },
        
        async addIntervention(scenarioKey, event, city_ind) {
            const intervention = this.interventionTableConfig[scenarioKey].handleSubmit(event, city_ind);
            const key = scenarioKey;
            const self = this
            if (!this.int_pars[city_ind][key]) {
                this.$set(this.int_pars[city_ind], key, []);
            }
            // validate intervention
            const notValid = !intervention.end || intervention.start < 0 || intervention.end <= intervention.start
            if (notValid) {
                this.$set(this.scenarioError, scenarioKey, `Please enter a valid day range`);
                return;
            }

            let overlaps = false;
            const int_pars = this.int_pars[city_ind][key];
            if (scenarioKey == 'vaccinate_closures') {
                intervention['nab_init_par1'] = this.nab_init_par1_vaccine_list[city_ind];
                intervention['nab_init_par2'] = this.nab_init_par2_vaccine_list[city_ind];
                intervention['nab_boost'] = this.nab_boost_vaccine_list[city_ind];
                intervention['doses'] = this.doses_vaccine_list[city_ind];
                intervention['interval'] = this.interval_vaccine_list[city_ind];
            }
            for (let i = 0; i < int_pars.length; i++) {
                const { start, end } = int_pars[i];
                const is_time_overlap = ((start <= intervention.start && end >= intervention.start) ||
                    (start <= intervention.end && end >= intervention.end) ||
                    (intervention.start <= start && intervention.end >= end));
                const is_has_interv_choice = 'intervention_choice' in intervention;
                const is_has_ag_choice = 'age_group_choice' in intervention;
                const is_random_interv = is_has_interv_choice && is_has_ag_choice;
                const is_equal_interv_choice = intervention.intervention_choice == int_pars[i].intervention_choice;
                const is_equal_ag_choice = intervention.age_group_choice == int_pars[i].age_group_choice || intervention.age_group_choice == "All" || int_pars[i].age_group_choice == "All";
                
                const cond_random = is_random_interv && (is_equal_interv_choice && is_equal_ag_choice);
                const cond_interv_choice = !is_random_interv && is_has_interv_choice && is_equal_interv_choice
                const cond_else = !is_has_interv_choice
                if (is_time_overlap && (cond_random || cond_interv_choice || cond_else)) {
                    if ('min_age' in intervention) {
                        min_age_tmp = int_pars[i].min_age
                        max_age_tmp = int_pars[i].max_age
                        if ((min_age_tmp <= intervention.min_age && max_age_tmp >= intervention.min_age) ||
                            (min_age_tmp <= intervention.max_age && max_age_tmp >= intervention.max_age) ||
                            (intervention.min_age <= min_age_tmp && intervention.max_age >= max_age_tmp))
                            {
                                overlaps = true;
                                break
                            }
                    } else {
                        overlaps = true;
                        break;
                    }
                }
            }
            if (overlaps){
                this.$set(this.scenarioError, scenarioKey, `Interventions of the same type cannot have overlapping day ranges.`)
                return ;
            }

            const outOfBounds = intervention.start > this.sim_length.best || intervention.end > this.sim_length.best || this.int_pars[city_ind][key].some(({start, end}) => {
                return start > self.sim_length.best || end > self.sim_length.best
            })
            if (outOfBounds){
                this.$set(this.scenarioError, scenarioKey, `Intervention cannot start or end after the campaign duration.`)
                return;
            }
            this.$set(this.scenarioError, scenarioKey, '');

            this.int_pars[city_ind][key].push(intervention);
            const result = this.int_pars[city_ind][key].sort((a, b) => a.start - b.start);
            this.$set(this.int_pars[city_ind], key, result);
            const response = await sciris.rpc('get_gantt', undefined, {int_pars_list: this.int_pars, intervention_config: this.interventionTableConfig, n_days: this.sim_length.best, tabs: this.tabs});
            this.intervention_figs = response.data;
        },
        async deleteIntervention(scenarioKey, index, city_ind) {
            this.$delete(this.int_pars[city_ind][scenarioKey], index);
            const response = await sciris.rpc('get_gantt', undefined, {int_pars_list: this.int_pars, intervention_config: this.interventionTableConfig, n_days: this.sim_length.best, tabs: this.tabs});
            this.intervention_figs = response.data;
        },

        parse_day(day, city_ind) {
            if (day == null || day == '') {
                const output = this.sim_length.best
                return output
            } else {
                const output = parseInt(day)
                return output
            }
        },

        parse_max_age(age) {
            if (age == null || age == '') {
                const output = 100
                return output
            } else {
                const output = parseInt(age)
                return output
            }
        },

        resize_start() {
            this.resizing = true;
        },
        resize_end() {
            this.resizing = false;
        },
        resize_apply(e) {
            if (this.resizing) {
                // Prevent highlighting
                e.stopPropagation();
                e.preventDefault();
                this.panel_width = (e.clientX / window.innerWidth) * 100;
            }
        },

        dispatch_resize(){
            window.dispatchEvent(new Event('resize'))
        },
        async get_version() {
            const response = await sciris.rpc('get_version');
            this.app.version = response.data;
        },

        async get_location_options() {
            let response = await sciris.rpc('get_location_options');
            for (let country of response.data) {
                this.reset_options.push(country);
            }
        },

        async get_licenses() {
            const response = await sciris.rpc('get_licenses');
            this.app.license = response.data.license;
            this.app.notice = response.data.notice;
        },

        async observeTime() {
            const response = await sciris.rpc('get_current_time');
            this.currentTime = response.data;
            if (this.currentTime != 100) {
                setTimeout(this.observeTime, 100);
            }
        },

        async runSim() {
            this.running = true;
            // this.graphs = this.$options.data().graphs; // Uncomment this to clear the graphs on each run
            this.errs = this.$options.data().errs;

            console.log('status:', this.status);

            // Run a a single sim
            try {
                if(this.datafile.local_path === null){
                    this.reset_datafile()
                }
                const kwargs = {
                    sim_pars: this.sim_pars,
                    epi_pars: this.epi_pars,
                    int_pars: this.int_pars,
                    datafile: this.datafile.server_path,
                    multiple_cities: this.multiple_cities,
                    show_contact_stat: this.show_contact_stat,
                    n_days: this.sim_length.best,
                    location: this.reset_choice,
                    infection_step_list: this.infection_step_choice_list,
                    rel_sus_type_list: this.rel_sus_choice_list,
                    rel_trans_type_list: this.rel_trans_choice_list,
                    interaction_records: this.interactionRecords,
                    population_volume_list: this.population_volume_choice_list,
                    infectiousTableConfig: this.infectiousTableConfig,
                    introduced_variants_list: this.introduced_variants_list,
                    cross_immunity_data: this.filtered,
                    tabs: this.tabs,
                    timeH: this.timeH
                }
                this.observeTime();
                console.log('run_sim: ', kwargs);
                const response = await sciris.rpc('run_sim', undefined, kwargs);
                this.result.graphs = response.data.graphs;
                this.result.files = response.data.files;
                this.result.summary = response.data.summary;
                this.errs = response.data.errs;
                // this.panel_open = this.errs.length > 0; // Better solution would be to have a pin button
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.int_pars = response.data.int_pars;
                this.history.push(JSON.parse(JSON.stringify({ sim_pars: this.sim_pars, epi_pars: this.epi_pars, reset_choice: this.reset_choice, infection_step_choice_list: this.infection_step_choice_list, rel_sus_choice_list: this.rel_sus_choice_list, rel_trans_choice_list: this.rel_trans_choice_list, population_volume_choice_list: this.population_volume_choice_list, int_pars: this.int_pars, result: this.result })));
                this.historyIdx = this.history.length - 1;

            } catch (e) {
                this.errs.push({
                    message: 'Unable to submit model.',
                    exception: `${e.constructor.name}: ${e.message}`
                })
                this.panel_open = true
            }
            this.running = false;
            this.currentTime = 0;
        },

        async resetPars() {
            const response = await sciris.rpc('get_defaults', [this.reset_choice]);
            this.sim_pars = response.data.sim_pars;
            this.epi_pars = response.data.epi_pars;
            this.sim_length = this.sim_pars['n_days'];
            this.int_pars = Array.from({ length: 20 }, () => ({}));
            this.intervention_figs = Array.from({ length: 20 }, () => ({}));
            this.cur_rel_sus_fig = Array.from({ length: 20 }, () => ({}));
            this.variant_figs = Array.from({ length: 20 }, () => ({}));
            this.infection_step_choice_list = Array.from({ length: 20 }, () => ('Covasim'));
            this.rel_sus_choice_list = Array.from({ length: 20 }, () => ('Constant (Covasim default)')),
            this.vaccine_choice_list = Array.from({ length: 20 }, () => ('pfizer')),
            this.variant_choice_list = Array.from({ length: 20 }, () => ('wild')),
            this.parameters_by_variant['rel_beta'].data = Array.from({ length: 20 }, () => (1.0)),
            this.parameters_by_variant['rel_symp_prob'].data = Array.from({ length: 20 }, () => (1.0)),
            this.parameters_by_variant['rel_severe_prob'].data = Array.from({ length: 20 }, () => (1.0)),
            this.parameters_by_variant['rel_crit_prob'].data = Array.from({ length: 20 }, () => (1.0)),
            this.parameters_by_variant['rel_death_prob'].data = Array.from({ length: 20 }, () => (1.0)),
            this.parameters_by_variant['dur_exp2inf'].data = Array.from({length: 20}, () => (4.5));
            this.parameters_by_variant['dur_inf2sym'].data = Array.from({length: 20}, () => (1.1));
            this.parameters_by_variant['dur_sym2sev'].data = Array.from({length: 20}, () => (6.6));
            this.parameters_by_variant['dur_sev2crit'].data = Array.from({length: 20}, () => (1.5));
            this.parameters_by_variant['dur_asym2rec'].data = Array.from({length: 20}, () => (8.0));
            this.parameters_by_variant['dur_mild2rec'].data = Array.from({length: 20}, () => (8.0));
            this.parameters_by_variant['dur_sev2rec'].data = Array.from({length: 20}, () => (18.1));
            this.parameters_by_variant['dur_crit2rec'].data = Array.from({length: 20}, () => (18.1));
            this.parameters_by_variant['dur_crit2die'].data = Array.from({length: 20}, () => (10.7));
            this.parameters_by_variant['n_import'].data = Array.from({length: 20}, () => (10));
            this.parameters_by_variant['start_day'].data = Array.from({length: 20}, () => (0));
            this.parameters_by_variant['oral_microbiota_percent'].data = Array.from({length: 20}, () => (0.0));
            this.parameters_by_variant['oral_microbiota_factor'].data = Array.from({length: 20}, () => (1.0));
            
            this.rel_trans_choice_list = Array.from({ length: 20 }, () => ('Independent(sus)')),
            this.population_volume_choice_list = Array.from({ length: 20 }, () => ('100K')),

            //this.setupFormWatcher('sim_pars');
            //this.setupFormWatcher('epi_pars');
            this.reset_datafile()
        },

        setupFormWatcher(paramKey) {
            const params = this[paramKey];
            if (!params) {
                return;
            }
            Object.keys(params).forEach(key => {
                this.$watch(`${paramKey}.${key}`, this.validateParam(key), { deep: true });
            });
        },

        watchSimLengthParam() {
            this.$watch('sim_length', this.validateParam('sim_length'), { deep: true });
        },

        validateParam(key) {
            return (param) => {
                if (param.best <= param.max && param.best >= param.min) {
                    this.$delete(this.paramError, key);
                } else {
                    this.$set(this.paramError, key, `Please enter a number between ${param.min} and ${param.max}`);
                }
            };
        },

        async downloadPars() {
            const d = new Date();
            const datestamp = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}_${d.getHours()}.${d.getMinutes()}.${d.getSeconds()}`;
            const fileName = `covasim_parameters_${datestamp}.json`;

            // Adapted from https://stackoverflow.com/a/45594892 by Gautham
            const data = {
                sim_pars: this.sim_pars,
                epi_pars: this.epi_pars,
                int_pars: this.int_pars
            };
            const fileToSave = new Blob([JSON.stringify(data, null, 4)], {
                type: 'application/json',
                name: fileName
            });
            saveAs(fileToSave, fileName);
        },

        async uploadCity(city_ind) {
            try {
                const response = await sciris.upload('upload_city');
                this.population_volume_options.push(response.data);
                this.population_volume_choice_list[parseInt(city_ind)] = response.data
            } catch (error) {
                sciris.fail(this, 'Could not upload city', error);
            }
        },
        async uploadPars() {
            try {
                const response = await sciris.upload('upload_pars');  //, [], {}, '');
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.int_pars = response.data.int_pars;
                this.result.graphs = [];
                this.intervention_figs = [{}, {}, {}, {}, {}];

                if (this.int_pars){
                    const gantt = await sciris.rpc('get_gantt', undefined, {int_pars_list: this.int_pars, intervention_config: this.interventionTableConfig, n_days: this.sim_length.best, tabs: this.tabs});
                    this.intervention_figs = gantt.data;
                }

            } catch (error) {
                sciris.fail(this, 'Could not upload parameters', error);
            }
        },
        upload_datafile: generate_upload_file_handler(function(filepath){
            vm.datafile.server_path = filepath
        }),

        reset_datafile() {
            this.datafile = {
                local_path: null,
                server_path: null
            }
        },

        loadPars() {
            this.sim_pars = this.history[this.historyIdx].sim_pars;
            this.epi_pars = this.history[this.historyIdx].epi_pars;
            this.reset_choice = this.history[this.historyIdx].reset_choice;
            this.int_pars = this.history[this.historyIdx].int_pars;
            this.result = this.history[this.historyIdx].result;
        },

        clickCCC(key) {
            console.log("here");
            console.log(this.groupHides[key]);
            this.groupHides[key] = !this.groupHides[key];
            console.log(this.groupHides[key]);
            console.log("here 2");
        },

        async downloadExcel() {
            const res = await fetch(this.result.files.xlsx.content);
            const blob = await res.blob();
            saveAs(blob, this.result.files.xlsx.filename);
        },

        async downloadJson() {
            const res = await fetch(this.result.files.json.content);
            const blob = await res.blob();
            saveAs(blob, this.result.files.json.filename);
        },

        async downloadPlot(cityIndex, index, ssInd) {
            obj = this.result.graphs[ssInd][cityIndex][parseInt(index)];
            const fileName = obj.description + "_City_" + cityIndex + ".svg";
            const data = await sciris.rpc('get_graph_svg', undefined, {fig: obj});
            // Create a Blob with the SVG content
            const sbb = data.data.substring(2, data.data.length - 1);
            const blob = new Blob([sbb], { type: 'image/svg+xml' })
            saveAs(blob, fileName);
        }

    },

});
  