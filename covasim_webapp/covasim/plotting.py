'''
Core plotting functions for simulations, multisims, and scenarios.

Also includes Plotly-based plotting functions to supplement the Matplotlib based
ones that are of the Sim and Scenarios objects. Intended mostly for use with the
webapp.
'''

import numpy as np
import pylab as pl
import sciris as sc
from . import misc as cvm
from . import defaults as cvd
from . import utils as cvu
from .settings import options as cvo


__all__ = ['plot_sim', 'plot_scens', 'plot_result', 'plot_compare', 'plot_people', 'plotly_sim', 'plotly_people',
           'plotly_hist_sus', 'plotly_hist_number_source_per_day',
           'plotly_not_infected_people_by_sus_norm', 'plotly_hist_number_source_cum', 'plotly_sars',
           'plotly_not_infected_people_by_sus', 'plotly_animate', 'plotly_part_80', 'plotly_rs', 'plotly_ars',
           'plotly_viral_load_per_day', 'plotly_viral_load_cum', 'plotly_risk_infected_by_age_group_per_day',
           'plotly_risk_infected_by_age_group_cum', 'plotly_infected_non_infected_group', 'plotly_contact_to_sus_trans']


#%% Plotting helper functions

def handle_args(fig_args=None, plot_args=None, scatter_args=None, axis_args=None, fill_args=None,
                legend_args=None, date_args=None, show_args=None, style_args=None, do_show=None, **kwargs):
    ''' Handle input arguments -- merge user input with defaults; see sim.plot for documentation '''

    # Set defaults
    defaults = sc.objdict()
    defaults.fig     = sc.objdict(figsize=(10, 8), num=None)
    defaults.plot    = sc.objdict(lw=1.5, alpha= 0.7)
    defaults.scatter = sc.objdict(s=20, marker='s', alpha=0.7, zorder=1.75, datastride=1) # NB: 1.75 is above grid lines but below plots
    defaults.axis    = sc.objdict(left=0.10, bottom=0.08, right=0.95, top=0.95, wspace=0.30, hspace=0.30)
    defaults.fill    = sc.objdict(alpha=0.2)
    defaults.legend  = sc.objdict(loc='best', frameon=False)
    defaults.date    = sc.objdict(as_dates=True, dateformat=None, rotation=None, start=None, end=None)
    defaults.show    = sc.objdict(data=True, ticks=True, interventions=True, legend=True, outer=False, tight=False, maximize=False, annotations=None, do_show=do_show, returnfig=cvo.returnfig)
    defaults.style   = sc.objdict(style=None, dpi=None, font=None, fontsize=None, grid=None, facecolor=None) # Use Covasim global defaults

    # Handle directly supplied kwargs
    for dkey,default in defaults.items():
        keys = list(kwargs.keys())
        for kw in keys:
            if kw in default.keys():
                default[kw] = kwargs.pop(kw)

    # Handle what to show
    if show_args is not None:
        annotations = show_args.get('annotations', None)
        if annotations in [True, False]: # Handle all on or all off
            show_keys = ['data', 'ticks', 'interventions', 'legend']
            for k in show_keys:
                show_args[k] = annotations

    # Merge arguments together
    args = sc.objdict()
    args.fig     = sc.mergedicts(defaults.fig,     fig_args)
    args.plot    = sc.mergedicts(defaults.plot,    plot_args)
    args.scatter = sc.mergedicts(defaults.scatter, scatter_args)
    args.axis    = sc.mergedicts(defaults.axis,    axis_args)
    args.fill    = sc.mergedicts(defaults.fill,    fill_args)
    args.legend  = sc.mergedicts(defaults.legend,  legend_args)
    args.date    = sc.mergedicts(defaults.date,    date_args)
    args.show    = sc.mergedicts(defaults.show,    show_args)
    args.style   = sc.mergedicts(defaults.style,   style_args)

    # Handle potential rcParams keys
    keys = list(kwargs.keys())
    for key in keys:
        if key in pl.rcParams:
            args.style[key] = kwargs.pop(key)

    # If unused keyword arguments remain, parse or raise an error
    if len(kwargs):

        # Everything remaining is not found
        notfound = sc.strjoin(kwargs.keys())
        valid = sc.strjoin(sorted(set([k for d in defaults.values() for k in d.keys()]))) # Remove duplicates and order
        errormsg = f'The following keywords could not be processed:\n{notfound}\n\n'
        errormsg += f'Valid keywords are:\n{valid}\n\n'
        errormsg += 'For more precise plotting control, use fig_args, plot_args, etc.'
        raise sc.KeyNotFoundError(errormsg)

    return args


def make_bold(string):
    return '<b>' + string + '</b>'


def handle_show_return(do_show=None, returnfig=None, fig=None, figs=None):
    ''' Helper function to handle both show and what to return -- a nothing if Jupyter, else a figure '''
    
    if do_show is None:
        do_show = cvo.show
    if returnfig is None:
        returnfig = cvo.returnfig

    figlist = sc.mergelists(fig, figs) # Usually just one figure, but here for completeness
    
    # Decide whether to show the figure or not
    backend = pl.get_backend()
    if backend == 'agg': # Cannot show plots for a non-interactive backend
        do_show = False
    if do_show: # Now check whether to show, and atually do it
        pl.show()

    # Show the figure, or close it
    if cvo.close and not do_show:
        for f in figlist:
            pl.close(f)

    # Return the figure or figures unless we're in Jupyter
    if not returnfig:
        return
    else:
        if figs is not None:
            return figlist
        else:
            return fig


def handle_to_plot(kind, to_plot, n_cols, sim, check_ready=True):
    ''' Handle which quantities to plot '''

    # Allow default kind to be overwritten by to_plot -- used by msim.plot()
    if isinstance(to_plot, tuple):
        kind, to_plot = to_plot # Split the tuple

    # Check that results are ready
    if check_ready and not sim.results_ready:
        errormsg = 'Cannot plot since results are not ready yet -- did you run the sim?'
        raise RuntimeError(errormsg)

    # If it matches a result key, convert to a list
    reskeys = sim.result_keys('main')
    varkeys = sim.result_keys('variant')
    allkeys = reskeys + varkeys
    if to_plot in allkeys:
        to_plot = sc.tolist(to_plot)

    # If not specified or specified as another string, load defaults
    if to_plot is None or isinstance(to_plot, str):
        to_plot = cvd.get_default_plots(to_plot, kind=kind, sim=sim)

    # If a list of keys has been supplied or constructed
    if isinstance(to_plot, list):
        to_plot_list = to_plot # Store separately
        to_plot = sc.odict() # Create the dict
        invalid = sc.autolist()
        for reskey in to_plot_list:
            if reskey in allkeys:
                name = sim.results[reskey].name if reskey in reskeys else sim.results['variant'][reskey].name
                to_plot[name] = [reskey] # Use the result name as the key and the reskey as the value
            else:
                invalid += reskey
        if len(invalid):
            errormsg = f'The following key(s) are invalid:\n{sc.strjoin(invalid)}\n\nValid main keys are:\n{sc.strjoin(reskeys)}\n\nValid variant keys are:\n{sc.strjoin(varkeys)}'
            raise sc.KeyNotFoundError(errormsg)

    to_plot = sc.odict(sc.dcp(to_plot)) # In case it's supplied as a dict

    # Handle rows and columns -- assume 5 is the most rows we would want
    n_plots = len(to_plot)
    if n_cols is None:
        max_rows = 5 # Assumption -- if desired, the user can override this by setting n_cols manually
        n_cols = int((n_plots-1)//max_rows + 1) # This gives 1 column for 1-4, 2 for 5-8, etc.
    n_rows,n_cols = sc.get_rows_cols(n_plots, ncols=n_cols) # Inconsistent naming due to Covasim/Matplotlib conventions

    return to_plot, n_cols, n_rows


def create_figs(args, sep_figs, fig=None, ax=None):
    '''
    Create the figures and set overall figure properties. If a figure is supplied,
    reset the axes labels for automatic use by other plotting functions (i.e. ax1, ax2, etc.)
    '''
    if sep_figs:
        fig = None
        figs = []
    else:
        if fig is None:
            if ax is None:
                fig = pl.figure(**args.fig) # Create the figure if none is supplied
            else:
                fig = ax.figure
        else:
            for i,fax in enumerate(fig.axes):
                fax.set_label(f'ax{i+1}')
        figs = None
    pl.subplots_adjust(**args.axis)
    return fig, figs


def create_subplots(figs, fig, shareax, n_rows, n_cols, pnum, fig_args, sep_figs, log_scale, title):
    ''' Create subplots and set logarithmic scale '''

    # Try to find axes by label, if they've already been defined -- this is to avoid the deprecation warning of reusing axes
    label = f'ax{pnum+1}'
    ax = None
    try:
        for fig_ax in fig.axes:
            if fig_ax.get_label() == label:
                ax = fig_ax
                break
    except:
        pass

    # Handle separate figs
    if sep_figs:
        figs.append(pl.figure(**fig_args))
        if ax is None:
            ax = pl.subplot(111, label=label)
    else:
        if ax is None:
            ax = pl.subplot(n_rows, n_cols, pnum+1, sharex=shareax, label=label)

    # Handle log scale
    if log_scale:
        if isinstance(log_scale, list):
            if title in log_scale:
                ax.set_yscale('log')
        else:
            ax.set_yscale('log')

    return ax


def plot_data(sim, ax, key, scatter_args, color=None):
    ''' Add data to the plot '''
    if sim.data is not None and key in sim.data and len(sim.data[key]):
        if color is None:
            color = sim.results[key].color
        datastride = scatter_args.pop('datastride', 1) # Temporarily pop so other arguments pass correctly to ax.scatter()
        x = np.array(sim.data.index)[::datastride]
        y = np.array(sim.data[key])[::datastride]
        ax.scatter(x, y, c=[color], label='Data', **scatter_args)
        scatter_args['datastride'] = datastride # Restore
    return


def plot_interventions(sim, ax):
    ''' Add interventions to the plot '''
    for intervention in sim['interventions']:
        if hasattr(intervention, 'plot_intervention'): # Don't plot e.g. functions
            intervention.plot_intervention(sim, ax)
    return


def title_grid_legend(ax, title, grid, commaticks, setylim, legend_args, show_args, show_legend=True):
    ''' Plot styling -- set the plot title, add a legend, and optionally add gridlines'''

    # Handle show_legend being in the legend args, since in some cases this is the only way it can get passed
    if 'show_legend' in legend_args:
        show_legend = legend_args.pop('show_legend')
        popped = True
    else:
        popped = False

    # Show the legend
    if show_legend and show_args['legend']: # It's pretty ugly, but there are multiple ways of controlling whether the legend shows

        # Remove duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        unique_inds = np.sort(np.unique(labels, return_index=True)[1])
        handles = [handles[u] for u in unique_inds]
        labels  = [labels[u]  for u in unique_inds]

        # Actually make legend
        ax.legend(handles=handles, labels=labels, **legend_args)

    # If we removed it from the legend_args dict, put it back now
    if popped:
        legend_args['show_legend'] = show_legend

    # Set the title, gridlines, and color
    ax.set_title(title)

    # Set the y axis style
    if setylim and ax.yaxis.get_scale() != 'log':
        ax.set_ylim(bottom=0)
    if commaticks:
        ylims = ax.get_ylim()
        if ylims[1] >= 1000:
            sc.commaticks(ax=ax)

    # Optionally remove x-axis labels except on bottom plots -- don't use ax.label_outer() since we need to keep the y-labels
    if show_args['outer']:
        lastrow = ax.get_subplotspec().is_last_row()
        if not lastrow:
            for label in ax.get_xticklabels(which='both'):
                label.set_visible(False)
            ax.set_xlabel('')

    return


def reset_ticks(ax, sim=None, date_args=None, start_day=None, n_cols=1):
    ''' Set the tick marks, using dates by default '''

    # Handle options
    date_args = sc.objdict(date_args) # Ensure it's not a regular dict
    if start_day is None and sim is not None:
        start_day = sim['start_day']

    # Set xticks as dates
    d_args = {k:date_args.pop(k) for k in ['as_dates', 'dateformat']} # Pop these to handle separately
    if d_args['as_dates']:
        if d_args['dateformat'] is None and n_cols >= 3: # Change default date format if more than 2 columns are shown
            d_args['dateformat'] = 'concise'
        if d_args['dateformat'] in ['covasim', 'sciris', 'auto', 'matplotlib', 'concise', 'brief']: # Handle date formatter rather than date format
            style, dateformat = d_args['dateformat'], None # Swap argument order
            style = style.replace('covasim', 'sciris') # In case any users are confused about what "default" is
        else:
            dateformat, style = d_args['dateformat'], 'sciris' # Otherwise, treat dateformat as a date format
        sc.dateformatter(ax=ax, style=style, dateformat=dateformat, **date_args) # Actually format the axis with dates, rotation, etc.
    else:
        # Handle start and end days
        xmin,xmax = ax.get_xlim()
        if date_args.start:
            xmin = float(sc.day(date_args.start, start_date=start_day)) # Keep original type (float)
        if date_args.end:
            xmax = float(sc.day(date_args.end, start_date=start_day))
        ax.set_xlim([xmin, xmax])

        # Set the x-axis intervals
        if date_args.interval:
            ax.set_xticks(np.arange(xmin, xmax+1, date_args.interval))

    # Restore date args
    date_args.update(d_args)

    return


def tidy_up(fig, figs, sep_figs, do_save, fig_path, args):
    ''' Handle saving, figure showing, and what value to return '''
    
    figlist = sc.mergelists(fig, figs) # Usually just one figure, but here for completeness

    # Optionally maximize -- does not work on all systems
    if args.show.maximize:
        for f in figlist:
            sc.maximize(fig=f)
        pl.pause(0.01) # Force refresh

    # Use tight layout for all figures
    if args.show.tight:
        for f in figlist:
            sc.figlayout(fig=f)

    # Handle saving
    if do_save:
        if isinstance(fig_path, str): # No figpath provided - see whether do_save is a figpath
            fig_path = sc.makefilepath(fig_path) # Ensure it's valid, including creating the folder
        cvm.savefig(fig=figlist, filename=fig_path) # Save the figure

    return handle_show_return(do_show=args.show.do_show, returnfig=args.show.returnfig, fig=fig, figs=figs)


def set_line_options(input_args, reskey, resnum, default):
    '''From the supplied line argument, usually a color or label, decide what to use '''
    if input_args is not None:
        if isinstance(input_args, dict): # If it's a dict, pull out this value
            output = input_args[reskey]
        elif isinstance(input_args, list): # If it's a list, ditto
            output = input_args[resnum]
        else: # Otherwise, assume it's the same value for all
            output = input_args
    else:
        output = default # Default value
    return output



#%% Core plotting functions

def plot_sim(to_plot=None, sim=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, date_args=None,
         show_args=None, style_args=None, n_cols=None, grid=True, commaticks=True,
         setylim=True, log_scale=False, colors=None, labels=None, do_show=None, sep_figs=False,
         fig=None, ax=None, **kwargs):
    ''' Plot the results of a single simulation -- see Sim.plot() for documentation '''

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                       legend_args=legend_args, show_args=show_args, date_args=date_args, style_args=style_args, do_show=do_show, **kwargs)
    to_plot, n_cols, n_rows = handle_to_plot('sim', to_plot, n_cols, sim=sim)

    # Do the plotting
    with cvo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)
        variant_keys = sim.result_keys('variant')
        for pnum,title,keylabels in to_plot.enumitems():
            ax = create_subplots(figs, fig, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
            for resnum,reskey in enumerate(keylabels):
                res_t = sim.results['date']
                if reskey in variant_keys:
                    res = sim.results['variant'][reskey]
                    ns = sim['n_variants']
                    variant_colors = sc.gridcolors(ns)
                    for variant in range(ns):
                        # Colors and labels
                        v_color = variant_colors[variant]
                        v_label = 'wild type' if variant == 0 else sim['variants'][variant-1].label
                        color = set_line_options(colors, reskey, resnum, v_color)  # Choose the color
                        label = set_line_options(labels, reskey, resnum, '')  # Choose the label
                        if label: label += f' - {v_label}'
                        else:     label = v_label
                        # Plotting
                        if res.low is not None and res.high is not None:
                            ax.fill_between(res_t, res.low[variant,:], res.high[variant,:], color=color, **args.fill)  # Create the uncertainty bound
                        ax.plot(res_t, res.values[variant,:], label=label, **args.plot, c=color)  # Actually plot the sim!
                else:
                    res = sim.results[reskey]
                    color = set_line_options(colors, reskey, resnum, res.color)  # Choose the color
                    label = set_line_options(labels, reskey, resnum, res.name)  # Choose the label
                    if res.low is not None and res.high is not None:
                        ax.fill_between(res_t, res.low, res.high, color=color, **args.fill)  # Create the uncertainty bound
                    ax.plot(res_t, res.values, label=label, **args.plot, c=color)  # Actually plot the sim!
                if args.show['data']:
                    plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data
                if args.show['ticks']:
                    reset_ticks(ax, sim, args.date, n_cols=n_cols) # Optionally reset tick marks (useful for e.g. plotting weeks/months)
            if args.show['interventions']:
                plot_interventions(sim, ax) # Plot the interventions
            title_grid_legend(ax, title, grid, commaticks, setylim, args.legend, args.show) # Configure the title, grid, and legend

        output = tidy_up(fig, figs, sep_figs, do_save, fig_path, args)

    return output


def plot_scens(to_plot=None, scens=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, date_args=None,
         show_args=None, style_args=None, n_cols=None, grid=False, commaticks=True, setylim=True,
         log_scale=False, colors=None, labels=None, do_show=None, sep_figs=False, fig=None, ax=None, **kwargs):
    ''' Plot the results of a scenario -- see Scenarios.plot() for documentation '''

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                   legend_args=legend_args, show_args=show_args, date_args=date_args, style_args=style_args, do_show=do_show, **kwargs)
    to_plot, n_cols, n_rows = handle_to_plot('scens', to_plot, n_cols, sim=scens.base_sim, check_ready=False) # Since this sim isn't run

    # Do the plotting
    with cvo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)
        default_colors = sc.gridcolors(ncolors=len(scens.sims))
        for pnum,title,reskeys in to_plot.enumitems():
            ax = create_subplots(figs, fig, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
            reskeys = sc.tolist(reskeys) # In case it's a string
            for reskey in reskeys:
                res_t = scens.datevec
                resdata = scens.results[reskey]
                for snum,scenkey,scendata in resdata.enumitems():
                    sim = scens.sims[scenkey][0] # Pull out the first sim in the list for this scenario
                    variant_keys = sim.result_keys('variant')
                    if reskey in variant_keys:
                        ns = sim['n_variants']
                        variant_colors = sc.gridcolors(ns)
                        for variant in range(ns):
                            res_y = scendata.best[variant,:]
                            color = variant_colors[variant]  # Choose the color
                            label = 'wild type' if variant == 0 else sim['variants'][variant - 1].label
                            ax.fill_between(res_t, scendata.low[variant,:], scendata.high[variant,:], color=color, **args.fill)  # Create the uncertainty bound
                            ax.plot(res_t, res_y, label=label, c=color, **args.plot)  # Plot the actual line
                            if args.show['data']:
                                plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data
                    else:
                        res_y = scendata.best
                        color = set_line_options(colors, scenkey, snum, default_colors[snum])  # Choose the color
                        label = set_line_options(labels, scenkey, snum, scendata.name)  # Choose the label
                        ax.fill_between(res_t, scendata.low, scendata.high, color=color, **args.fill)  # Create the uncertainty bound
                        ax.plot(res_t, res_y, label=label, c=color, **args.plot)  # Plot the actual line
                        if args.show['data']:
                            plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data

                    if args.show.interventions:
                        plot_interventions(sim, ax) # Plot the interventions
                    if args.show['ticks']:
                        reset_ticks(ax, sim, args.date) # Optionally reset tick marks (useful for e.g. plotting weeks/months)
            if args.show.legend:
                title_grid_legend(ax, title, grid, commaticks, setylim, args.legend, args.show, pnum==0) # Configure the title, grid, and legend -- only show legend for first

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, args)


def plot_result(key, sim=None, fig_args=None, plot_args=None, axis_args=None, scatter_args=None,
                date_args=None, style_args=None, grid=False, commaticks=True, setylim=True, color=None, label=None,
                do_show=None, do_save=False, fig_path=None, fig=None, ax=None, **kwargs):
    ''' Plot a single result -- see ``cv.Sim.plot_result()`` for documentation '''

    # Handle inputs
    sep_figs = False # Only one figure
    fig_args  = sc.mergedicts({'figsize':(8,5)}, fig_args)
    axis_args = sc.mergedicts({'top': 0.95}, axis_args)
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args,
                       date_args=date_args, style_args=style_args, do_show=do_show, **kwargs)

    # Gather results
    res = sim.results[key]
    res_t = sim.results['date']
    if color is None:
        color = res.color

    # Do the plotting
    with cvo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)

        # Reuse the figure, if available
        if ax is None: # Otherwise, make a new one
            try:
                ax = fig.axes[0]
            except:
                ax = fig.add_subplot(111, label='ax1')

        if label is None:
            label = res.name
        if res.low is not None and res.high is not None:
            ax.fill_between(res_t, res.low, res.high, color=color, **args.fill) # Create the uncertainty bound

        ax.plot(res_t, res.values, c=color, label=label, **args.plot)
        plot_data(sim, ax, key, args.scatter, color=color) # Plot the data
        plot_interventions(sim, ax) # Plot the interventions
        title_grid_legend(ax, res.name, grid, commaticks, setylim, args.legend, args.show) # Configure the title, grid, and legend
        reset_ticks(ax, sim, args.date) # Optionally reset tick marks (useful for e.g. plotting weeks/months)

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, args)


def plot_compare(df, log_scale=True, fig_args=None, axis_args=None, style_args=None, grid=False,
                 commaticks=True, setylim=True, color=None, label=None, fig=None,
                 do_save=None, do_show=None, fig_path=None, **kwargs):
    ''' Plot a MultiSim comparison -- see MultiSim.plot_compare() for documentation '''

    # Handle inputs
    sep_figs = False
    fig_args  = sc.mergedicts({'figsize':(8,8)}, fig_args)
    axis_args = sc.mergedicts({'left': 0.16, 'bottom': 0.05, 'right': 0.98, 'top': 0.98, 'wspace': 0.50, 'hspace': 0.10}, axis_args)
    args = handle_args(fig_args=fig_args, axis_args=axis_args, style_args=style_args, do_show=do_show, **kwargs)

    # Map from results into different categories
    mapping = {
        'cum': 'Cumulative counts',
        'new': 'New counts',
        'n': 'Number in state',
        'r': 'R_eff',
        }
    category = []
    for v in df.index.values:
        v_type = v.split('_')[0]
        if v_type in mapping:
            category.append(v_type)
        else:
            category.append('other')
    df['category'] = category

    # Plot
    with cvo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs=False, fig=fig)
        for i,m in enumerate(mapping):
            not_r_eff = m != 'r'
            if not_r_eff:
                ax = fig.add_subplot(2, 2, i+1)
            else:
                ax = fig.add_subplot(8, 2, 10)
            dfm = df[df['category'] == m]
            logx = not_r_eff and log_scale
            dfm.plot(ax=ax, kind='barh', logx=logx, legend=False)
            if not(not_r_eff):
                ax.legend(loc='upper left', bbox_to_anchor=(0,-0.3))
            ax.grid(True)

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, args)


def get_window_data(y, window_size=5):
    import pandas as pd
    data = pd.DataFrame({'y': y})
    smoothed_data = data['y'].rolling(window_size).mean()
    for i in range(window_size - 1):
        smoothed_data[i] = np.sum(y[:(i +1 )]) / (i + 1)
    return smoothed_data

def double_window(y):
    return get_window_data(get_window_data(y))

#%% Other plotting functions
def plot_people(people, bins=None, width=1.0, alpha=0.6, fig_args=None, axis_args=None,
                plot_args=None, style_args=None, do_show=None, fig=None):
    ''' Plot statistics of a population -- see People.plot() for documentation '''

    # Handle inputs
    if bins is None:
        bins = np.arange(0,101)

    # Set defaults
    color     = [0.1,0.1,0.1] # Color for the age distribution
    n_rows    = 4 # Number of rows of plots
    offset    = 0.5 # For ensuring the full bars show up
    gridspace = 10 # Spacing of gridlines
    zorder    = 10 # So plots appear on top of gridlines

    # Handle other arguments
    fig_args   = sc.mergedicts(dict(figsize=(18,11)), fig_args)
    axis_args  = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)
    plot_args  = sc.mergedicts(dict(lw=1.5, alpha=0.6, c=color, zorder=10), plot_args)
    style_args = sc.mergedicts(style_args)

    # Compute statistics
    min_age = min(bins)
    max_age = max(bins)
    edges = np.append(bins, np.inf) # Add an extra bin to end to turn them into edges
    age_counts = np.histogram(people.age, edges)[0]

    with cvo.with_style(style_args):

        # Create the figure
        if fig is None:
            fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)

        # Plot age histogram
        pl.subplot(n_rows,2,1)
        pl.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
        pl.xlim([min_age-offset,max_age+offset])
        pl.xticks(np.arange(0, max_age+1, gridspace))
        pl.xlabel('Age')
        pl.ylabel('Number of people')
        pl.title(f'Age distribution ({len(people):n} people total)')

        # Plot cumulative distribution
        pl.subplot(n_rows,2,2)
        age_sorted = sorted(people.age)
        y = np.linspace(0, 100, len(age_sorted)) # Percentage, not hard-coded!
        pl.plot(age_sorted, y, '-', **plot_args)
        pl.xlim([0,max_age])
        pl.ylim([0,100]) # Percentage
        pl.xticks(np.arange(0, max_age+1, gridspace))
        pl.yticks(np.arange(0, 101, gridspace)) # Percentage
        pl.xlabel('Age')
        pl.ylabel('Cumulative proportion (%)')
        pl.title(f'Cumulative age distribution (mean age: {people.age.mean():0.2f} years)')

        # Calculate contacts
        lkeys = people.layer_keys()
        n_layers = len(lkeys)
        contact_counts = sc.objdict()
        for lk in lkeys:
            layer = people.contacts[lk]
            p1ages = people.age[layer['p1']]
            p2ages = people.age[layer['p2']]
            contact_counts[lk] = np.histogram(p1ages, edges)[0] + np.histogram(p2ages, edges)[0]

        # Plot contacts
        layer_colors = sc.gridcolors(n_layers)
        share_ax = None
        for w,w_type in enumerate(['total', 'percapita', 'weighted']): # Plot contacts in different ways
            for i,lk in enumerate(lkeys):
                contacts_lk = people.contacts[lk]
                members_lk = contacts_lk.members
                n_contacts = len(contacts_lk)
                n_members = len(members_lk)
                if w_type == 'total':
                    weight = 1
                    total_contacts = 2*n_contacts # x2 since each contact is undirected
                    ylabel = 'Number of contacts'
                    participation = n_members/len(people) # Proportion of people that have contacts in this layer
                    title = f'Total contacts for layer "{lk}": {total_contacts:n}\n({participation*100:.0f}% participation)'
                elif w_type == 'percapita':
                    age_counts_within_layer = np.histogram(people.age[members_lk], edges)[0]
                    weight = np.divide(1.0, age_counts_within_layer, where=age_counts_within_layer>0)
                    mean_contacts_within_layer = 2*n_contacts/n_members if n_members else 0  # Factor of 2 since edges are bi-directional
                    ylabel = 'Per capita number of contacts'
                    title = f'Mean contacts for layer "{lk}": {mean_contacts_within_layer:0.2f}'
                elif w_type == 'weighted':
                    weight = people.pars['beta_layer'][lk]*people.pars['beta']
                    total_weight = np.round(weight*2*n_contacts)
                    ylabel = 'Weighted number of contacts'
                    title = f'Total weight for layer "{lk}": {total_weight:n}'

                ax = pl.subplot(n_rows, n_layers, n_layers*(w+1)+i+1, sharey=share_ax)
                pl.bar(bins, contact_counts[lk]*weight, color=layer_colors[i], width=width, zorder=zorder, alpha=alpha)
                pl.xlim([min_age-offset,max_age+offset])
                pl.xticks(np.arange(0, max_age+1, gridspace))
                pl.xlabel('Age')
                pl.ylabel(ylabel)
                pl.title(title)
                if w_type == 'weighted':
                    share_ax = ax # Update shared axis



    return handle_show_return(fig=fig, do_show=do_show)


#%% Plotly functions

def import_plotly():
    ''' Try to import Plotly, but fail quietly if not available '''

    # Try to import Plotly normally
    try:
        import plotly.graph_objects as go
        return go

    # If that failed, handle it gracefully
    except Exception as E:

        class PlotlyImportFailed(object):
            ''' Define a micro-class to give a helpful error message if the import failed '''

            def __init__(self, E):
                self.E = E

            def __getattr__(self, attr):
                errormsg = f'Plotly import failed: {str(self.E)}. Plotly plotting is not available. Please install Plotly first.'
                raise ImportError(errormsg)

        go = PlotlyImportFailed(E)
        return go


def get_individual_states(sim): # pragma: no cover
    ''' Helper function to convert people into integers '''

    people = sim.people

    states = [
        {'name': 'Healthy',
         'quantity': None,
         'color': '#a6cee3',
         'value': 0
         },
        {'name': 'Exposed',
         'quantity': 'date_exposed',
         'color': '#ff7f00',
         'value': 2
         },
        {'name': 'Infectious',
         'quantity': 'date_infectious',
         'color': '#e33d3e',
         'value': 3
         },
        {'name': 'Recovered',
         'quantity': 'date_recovered',
         'color': '#3e89bc',
         'value': 4
         },
        {'name': 'Dead',
         'quantity': 'date_dead',
         'color': '#000000',
         'value': 5
         },
    ]

    z = np.zeros((len(people), sim.npts))
    for state in states:
        date = state['quantity']
        if date is not None:
            inds = sim.people.defined(date)
            for ind in inds:
                z[ind, int(people[date][ind]):] = state['value']

    return z, states


# Default settings for the Plotly legend
plotly_legend = dict(legend_orientation='h', legend=dict(x=0.0, y=1.18))


def plotly_interventions(sim, fig, basename="", max_y=1, add_to_legend=False): # pragma: no cover
    ''' Add vertical lines for interventions to the plot '''
    go = import_plotly() # Load Plotly
    freq_count = 300
    if sim['interventions']:
        for interv in sim['interventions']:
            if hasattr(interv, 'days'):
                for interv_day in interv.days:
                    if interv_day and interv_day < sim['n_days']:
                        #interv_date = sim.date(interv_day, as_date=False)
                        #fig.add_shape(dict(type='line', xref='x', yref='paper', x0=interv_day, x1=interv_day, y0=0, y1=1, line=dict(width=0.5, dash='dash')))
                        #if add_to_legend:
                        fig.add_trace(go.Scatter(
                            x=np.full(freq_count, interv_day), y=np.linspace(0, max_y, freq_count),
                            hovertext=f"{basename}: {interv.label} on day {interv_day}",
                            hovertemplate=f'{basename}: {interv.label} on day {interv_day}<extra></extra>', 
                            mode='lines', line=dict(dash='dash', width=0.5), showlegend=False))

    return

def _hsv2rgb(color_hex):
    color_hex = color_hex.lstrip('#')

    red = int(color_hex[0:2], 16)
    green = int(color_hex[2:4], 16)
    blue = int(color_hex[4:6], 16)
    return red, green, blue

def plotly_sim(sims, do_show=False): # pragma: no cover
    ''' Main simulation results -- parallel of sim.plot() '''

    go = import_plotly() # Load Plotly
    plots = []
    to_plot = cvd.get_default_plots()
    
    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]

    title2new_title = {
        'Total counts': 'Total cases',
        'Daily counts': 'Daily cases',
        'Health outcomes': 'Cumulative outcomes',
        'Vaccinating': 'Vaccinating'
    }
    label2new_label = {
        'Cumulative infections': 'Infections',
        'Number of new infections': 'Infections',
        'Cumulative reinfections': 'Reinfections',
        'Number of new reinfections': 'Reinfections',
        'Cumulative severe cases': 'Severe cases',
        'Cumulative critical cases': 'Critical cases',
        'Cumulative deaths': 'Deaths',
        'Number of new diagnoses_rpn': 'Registrations',
        'Cumulative diagnoses_rpn': 'Registrations'
    }
    title2y_axis = {
        'Total counts': 'Total cases',
        'Daily counts': 'Daily cases',
        'Health outcomes': 'Cases',
        'Vaccinating': 'Fraction'
    }

    for p,title,keylabels in to_plot.enumitems():
        fig = go.Figure()
        # plot several sims
        for key in keylabels:
            max_y = 0
            for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
                label = sim.results[key].name
                this_color = sim.results[key].color
                x = np.arange(sim.results['date'][:].size)
                y = sim.results[key][:]
                max_y = np.max(sim.results[key][:]) if np.max(sim.results[key][:]) > max_y else max_y 
                has_testing = any(list(map(lambda x: 'testing' in x.label, sim['interventions'])))

                if label == "Cumulative known deaths" or label == "Number infectious" or ('diagnose' in label and not has_testing):
                    continue
                new_label = label2new_label[label] if label in label2new_label else label
                _r, _g, _b = _hsv2rgb(this_color)
                is_show_legend = (i == sims_count - 1) or title == "Total counts" or title == "Daily counts"
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                    line=dict(
                            color=f'rgba({_r}, {_g}, {_b}, {brightness})',
                            width=2  # Ширина линии
                    ), showlegend=is_show_legend, 
                    name=f"{sim.label}: {new_label}", hovertext=list(map(lambda t: str(t)+ '; ' + sim.label + '; ' + new_label, zip(x, y.astype(int)))), hoverinfo="text"))
                if sim.data is not None and key in sim.data:
                    xdata = sim.data['date']
                    ydata = sim.data[key]
                    fig.add_trace(go.Scatter(x=xdata, y=ydata, mode='markers',
                        line=dict(
                            color=f'rgba({_r}, {_g}, {_b}, {brightness})',
                            width=2  # Ширина линии
                        ), showlegend=is_show_legend,
                        name=f"{sim.label}: {new_label} (data)", hovertemplate=f'{ydata}: {new_label}'))
            for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
                plotly_interventions(sim, fig, basename=sim.label,
                                     max_y=max_y, add_to_legend=(p==0)) # Only add the intervention label to the legend for the first plot
        fig.update_layout(title={'text':title2new_title[title]}, xaxis_title='Day', yaxis_title=title2y_axis[title], autosize=True, **plotly_legend)
        plots.append(fig)

    if do_show:
        for fig in plots:
            fig.show()

    return plots


def plotly_people(sim, do_show=False): # pragma: no cover
    ''' Plot a "cascade" of people moving through different states '''

    go = import_plotly() # Load Plotly
    z, states = get_individual_states(sim)
    fig = go.Figure()

    for state in states[::-1]:  # Reverse order for plotting
        x = np.arange(sim.results['date'][:].size)
        y = (z == state['value']).sum(axis=0)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            stackgroup='one',
            line=dict(width=0.5, color=state['color']),
            fillcolor=state['color'],
            hoverinfo='y+name',
            name=state['name']
        ))

    plotly_interventions(sim, fig, max_y=sim.n)
    fig.update_layout(yaxis_range=(0, sim.n))
    fig.update_layout(title={'text': 'Health status'}, xaxis_title='Day', yaxis_title='Agents', autosize=True, **plotly_legend)

    if do_show:
        fig.show()

    return fig

def plotly_hist_sus(sim, do_show=False):
    go = import_plotly() # Load Plotly
    fig = go.Figure()
    cur_analyzer= sim.get_analyzer('seir')
    fig.add_trace(
        go.Scatter(x=cur_analyzer.hist_sus[1], y=cur_analyzer.hist_sus[0], name="hist sus"))

    fig.update_layout(title={'text': make_bold('Histogram susceptibility')}, 
                      yaxis_title='People count', autosize=True,
                      xaxis_title='susceptibility',  **plotly_legend)
    
    
    if do_show:
        fig.show()

    return fig


#def plotly_dist_sus(distname, age_dist):
#    go = import_plotly() # Load Plotly
#    fig = go.Figure()
#    agent_count = np.sum(list(age_dist.values()))
#    rel_sus = np.zeros(agent_count, dtype=np.float32)
#    progs = dict(
#            age_cutoffs   = np.array([0,       10,      20,      30,      40,      50,      60,      70,      80,      90,]),     # Age cutoffs (lower limits)
#            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from Zhang et al., https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
#    )
#    # generate simple age dist
#    ppl_age = np.zeros(agent_count, dtype=np.int64)
#    tmp_ind = 0
#    for (ind, cnt) in enumerate(age_dist.values()):
#        ppl_age[tmp_ind:tmp_ind+cnt] = ind * 10
#        tmp_ind += cnt
#    inds = np.digitize(ppl_age, progs['age_cutoffs'])-1
#    if distname == 'constants':
#        rel_sus[:] = progs['sus_ORs'][inds]  # Default susceptibilities
#    elif distname == 'normal_pos':
#        for i in range(1, 10):
#            inds_age = np.where((progs['age_cutoffs'][i - 1] <= ppl_age) * (ppl_age < progs['age_cutoffs'][i]))[0]
#            rel_sus[inds_age] = cvu.sample(dist='normal_pos', par1=progs['sus_ORs'][i - 1], par2=0.2,
#                                                size=inds_age.size)
#        inds_age = np.where(ppl_age > progs['age_cutoffs'][9])[0]
#        rel_sus[inds_age] = cvu.sample(dist='normal_pos', par1=progs['sus_ORs'][9], par2=0.2,
#                                            size=inds_age.size)
#    elif distname == 'lognormal':
#        for i in range(1, 10):
#            inds_age = np.where((progs['age_cutoffs'][i - 1] <= ppl_age) * (ppl_age < progs['age_cutoffs'][i]))[0]
#            rel_sus[inds_age] = cvu.sample(dist='lognormal', par1=progs['sus_ORs'][i - 1], par2=0.2,
#                                                size=inds_age.size)
#        inds_age = np.where(ppl_age > progs['age_cutoffs'][9])[0]
#        rel_sus[inds_age] = cvu.sample(dist='lognormal', par1=progs['sus_ORs'][9], par2=0.2,
#                                            size=inds_age.size)
#    elif distname == 'uniform':
#        my_unfiform_intervals = [(0.17, 0.5), (0.5, 0.8), (0.8, 1.1), (0.8, 1.1), (0.8, 1.1), (0.8, 1.1),
#                                    (1.1, 1.34), (1.34, 1.5), (1.34, 1.5), (1.34, 1.5)]
#        for i in range(1, 10):
#            inds_age = np.where((progs['age_cutoffs'][i - 1] <= ppl_age) * (ppl_age < progs['age_cutoffs'][i]))[0]
#            rel_sus[inds_age] = np.random.uniform(
#                my_unfiform_intervals[i - 1][0], my_unfiform_intervals[i - 1][1], size=inds_age.size)
#        inds_age = np.where(ppl_age > progs['age_cutoffs'][9])[0]
#        rel_sus[inds_age] = np.random.uniform(
#            my_unfiform_intervals[9][0], my_unfiform_intervals[9][1], size=inds_age.size)
#    elif distname == 'uniform_all':
#        rel_sus[:]     = np.random.uniform(0, 1.47, size=rel_sus.size) # Uniform susceptibilities
#    elif distname == 'lognormal_lite_all':
#        rel_sus[:]     = cvu.sample(dist='lognormal', par1=0.65, par2=0.5, size=rel_sus.size) # lognormal susceptibilities
#        ss = rel_sus[rel_sus > 1.5].size
#        rel_sus[rel_sus > 1.5] = cvu.sample(dist='lognormal', par1=0.5, par2=0.5, size=ss)
#    elif distname == 'lognormal_hot_all':
#        rel_sus[:]     = cvu.sample(dist='lognormal', par1=1.0, par2=0.5, size=rel_sus.size) # lognormal susceptibilities
#        ss = rel_sus[rel_sus > 1.5].size
#        rel_sus[rel_sus > 1.5] = cvu.sample(dist='lognormal', par1=1.0, par2=0.5, size=ss)        
#    elif distname == 'normal_pos_all':
#        rel_sus[:]     = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=rel_sus.size) # normal susceptibilities
#        ss = rel_sus[rel_sus > 1.5].size
#        rel_sus[rel_sus > 1.5] = cvu.sample(dist='normal_pos', par1=1.0, par2=0.5, size=ss)
#    elif distname == 'beta_1_all':
#        rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=1.2, par2=6.2, step=1.5, size=rel_sus.size)
#    elif distname == 'beta_2_all':
#        rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=1.8, par2=6.2, step=1.5, size=rel_sus.size)
#    elif distname == 'beta_3_all':
#        rel_sus[:]     = 0.1 + cvu.sample(dist='beta', par1=2.2, par2=5.6, step=1.5, size=rel_sus.size)
#    elif distname == 'neg_binomial_all':
#        rel_sus[:]     = 0.1 + cvu.sample(dist='neg_binomial', par1=0.21, par2=3, step=0.07, size=rel_sus.size)
#    #elif pars['rel_sus_type'] == 'binom_2_all':
#    #    rel_sus[:]     = cvu.sample(dist='beta', par1=2.2, par2=5.6, step=1.5, size=rel_sus.size)    
#    else:
#        raise RuntimeError("Not corrected type of rel_sus")
#
#    rel_sus[rel_sus > 2.5] = 2.5
#    
#    hist_rel_sus = np.histogram(rel_sus)
#    summ = np.sum(list(age_dist.values()))
#    y_ttotal = hist_rel_sus[0] / summ
#    fig.add_trace(
#        go.Scatter(x=hist_rel_sus[1], y=y_ttotal, name="hist sus"))
#
#    fig.update_layout(title={'text': make_bold(f'Histogram susceptibility')}, 
#                      yaxis_title='People count', height=400, width=500,
#                      yaxis_range=[-0.03, np.max(y_ttotal) + 0.1],
#                      xaxis_title='Susceptibility',  **plotly_legend)
#    return fig


def plotly_rs(sims, do_show=False):
    go = import_plotly() # Load Plotly
    fig = go.Figure()

    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]
    for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
        cur_analyzer= sim.get_analyzer('seir')
        y = cur_analyzer.rs
        x = np.arange(y.size) * 7 + 7
        fig.add_trace(
            go.Scatter(x=x, y=y, visible=False,
            name=f"{sim.label}: R",
            line=dict(
                color=f'rgba(255, 0, 0, {brightness})',
                width=2  # Ширина линии
            )))
        fig.add_trace(
            go.Scatter(x=x, y=get_window_data(y), 
                name=f"{sim.label}: R",
                line=dict(
                color=f'rgba(255, 0, 0, {brightness})',
                width=2  # Ширина линии
            )))
        # base testing
        y_bt = cur_analyzer.rs_based_testing
        has_testing = np.sum(y_bt) > 0 
        if has_testing:
            x_bt = np.arange(y_bt.size) * 7 + 7
            fig.add_trace(
                go.Scatter(x=x_bt, y=y_bt, visible=False,
                name=f"{sim.label}: R (based testing)",
                line=dict(
                    color=f'rgba(255, 100, 0, {2 * brightness})',
                    width=2  # Ширина линии
                )))
            fig.add_trace(
                go.Scatter(x=x_bt, y=get_window_data(y_bt), 
                    name=f"{sim.label}: R (based testing)",
                    line=dict(
                    color=f'rgba(255, 100, 0, {2 * brightness})',
                    width=2  # Ширина линии
                )))

        y_sim = sim.compute_r_eff()
        x_sim = np.arange(y_sim.size)
        fig.add_trace(
            go.Scatter(x=x_sim, y=y_sim,
                        name=f"{sim.label}: R_eff",
                        line=dict(
                            color=f'rgba(0, 255, 0, {brightness})',
                            width=2  # Ширина линии
                        ),
                        visible=False))
        fig.add_trace(
            go.Scatter(x=x_sim, y=get_window_data(y_sim),
                        name=f"{sim.label}: R_eff",
                        line=dict(
                        color=f'rgba(0, 255, 0, {brightness})',
                        width=2  # Ширина линии
                    )))
    
    n_ft = 3 if has_testing else 2
    tf_arr, ft_arr = ([True, False] * n_ft, [False, True] * n_ft)
    fig.update_layout(title={'text': make_bold('Effective reproductive number (Rt)')}, 
                      yaxis_title='Rt', autosize=True,
                      xaxis_title='Day',
                      updatemenus=[
                        dict(
                            active=1,
                            buttons=list([
                                dict(label="Raw",
                                    method="update",
                                    args=[{"visible": tf_arr * sims_count}]),
                                dict(label="Window",
                                    method="update",
                                    args=[{"visible": ft_arr * sims_count}])
                            ]),
                            x=1,  # Устанавливаем положение кнопки по горизонтали (1 = справа)
                            xanchor="right",  # Устанавливаем якорную точку кнопки справа
                            y=1.15,  # Устанавливаем положение кнопки по вертикали
                            yanchor="top"  # Устанавливаем якорную точку кнопки сверху
                        )
                    ],
                    **plotly_legend)
    
    
    if do_show:
        fig.show()

    return fig


def plotly_ars(sims, do_show=False):
    go = import_plotly() # Load Plotly
    fig = go.Figure()
    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]
    for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
        cur_analyzer= sim.get_analyzer('seir')
        y = cur_analyzer.ars
        x = np.arange(y.size) * 3 + 3
        fig.add_trace(
            go.Scatter(x=x, y=y, visible=False, name=sim.label,
                        line=dict(
                        color=f'rgba(255, 0, 0, {brightness})',
                        width=2  # Ширина линии
                    )))
        fig.add_trace(
            go.Scatter(x=x, y=get_window_data(y), name=sim.label,
                       line=dict(
                        color=f'rgba(255, 0, 0, {brightness})',
                        width=2  # Ширина линии
                    )))
    
    fig.update_layout(title={'text': make_bold('Attack rate')}, 
                      yaxis_title='Attack rate (%)', autosize=True,
                      xaxis_title='Day',
                      updatemenus=[
                        dict(
                            active=1,
                            buttons=list([
                                dict(label="Raw",
                                    method="update",
                                    args=[{"visible": [True, False] * sims_count}]),
                                dict(label="Window",
                                    method="update",
                                    args=[{"visible": [False, True] * sims_count}])
                            ]),
                            x=1,  # Устанавливаем положение кнопки по горизонтали (1 = справа)
                            xanchor="right",  # Устанавливаем якорную точку кнопки справа
                            y=1.15,  # Устанавливаем положение кнопки по вертикали
                            yanchor="top"  # Устанавливаем якорную точку кнопки сверху
                        )
                    ],
                    **plotly_legend)
    
    
    if do_show:
        fig.show()

    return fig


def plotly_part_801(sim, do_show=False):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

  # Создание данных для двух графиков
    x = [1, 2, 3, 4, 5]
    y1 = [2, 4, 1, 5, 3]
    y2 = [1, 3, 5, 2, 4]

    # Создание графиков
    graph1 = go.Scatter(x=x, y=y1, name='Raw', visible=False)
    graph2 = go.Scatter(x=x, y=y2, name='Smooth')

    # Создание объекта Figure с двумя графиками
    fig = go.Figure(data=[graph1, graph2])

    fig.update_layout(
        updatemenus=[
            dict(
                active=1,
                buttons=list([
                    dict(label="Raw",
                        method="update",
                        args=[{"visible": [True, False]}, {"title": "Raw"}]),
                    dict(label="Smooth",
                        method="update",
                        args=[{"visible": [False, True]}, {"title": "Smooth"}])
                ]),
            )
        ])
    return fig


def plotly_part_80(sims, do_show=False):
    go = import_plotly() # Load Plotly
    fig = go.Figure()
    
    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]

    for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
        cur_analyzer= sim.get_analyzer('seir')
        y = cur_analyzer.spread_count_stat
        x = 1 + np.arange(y.size)
        fig.add_trace(
            go.Scatter(x=x, y=y, 
                name=sim.label,
                       line=dict(
                        color=f'rgba(255, 0, 0, {brightness})',
                        width=2),
                visible=False))
        fig.add_trace(
            go.Scatter(x=x, y=get_window_data(y),
                        name=sim.label,
                        line=dict(
                            color=f'rgba(255, 0, 0, {brightness})',
                            width=2)))
        fig.add_trace(
            go.Scatter(x=x, y=double_window(y),
                        name=sim.label,
                        line=dict(
                            color=f'rgba(255, 0, 0, {brightness})',
                            width=2), visible=False))

    fig.update_layout(title={'text': make_bold('Who infected 80% of people')}, 
                      yaxis_title='Percentage of infected people who infected 80% of people', autosize=True,
                      xaxis_title='Day',
                      updatemenus=[
                            dict(
                                active=1,
                                buttons=list([
                                    dict(label="Raw",
                                        method="update",
                                        args=[{"visible": [True, False, False] * sims_count}]),
                                    dict(label="Window",
                                        method="update",
                                        args=[{"visible": [False, True, False] * sims_count}]),
                                    dict(label="Double Window",
                                        method="update",
                                        args=[{"visible": [False, False, True] * sims_count}])
                                ]
                                ),
                                x=1,  # Устанавливаем положение кнопки по горизонтали (1 = справа)
                                xanchor="right",  # Устанавливаем якорную точку кнопки справа
                                y=1.15,  # Устанавливаем положение кнопки по вертикали
                                yanchor="top"  # Устанавливаем якорную точку кнопки сверху
                            )
                        ],
                        **plotly_legend)
    
    
    if do_show:
        fig.show()

    return fig


def plotly_animate_smth(sim, frames, xaxis_title, yaxis_title, title, do_show=False):
    go = import_plotly() # Load Plotly
    cur_analyzer= sim.get_analyzer('seir')

    days = sim.tvec

    fig_dict = {
        'data': [],
        'layout': {},
        'frames': []
    }

    fig_dict['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 200, 'redraw': True},
                                    'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                      'mode': 'immediate',
                                      'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Day: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 200},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data
    fig_dict['data'] = [go.Bar(x=[], y=[])]
    for i, day in enumerate(days):
        slider_step = {'args': [
            [i],
            {'frame': {'duration': 5, 'redraw': True},
             'mode': 'immediate', }
            ],
                'label': i,
                'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    fig_dict['frames'] = frames
    fig_dict['layout']['sliders'] = [sliders_dict]
    
    fig = go.Figure(fig_dict)

    fig.update_layout(
        autosize=True,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )

    fig.update_layout(title={'text': title}, **plotly_legend)

    if do_show:
        fig.show()

    return fig 


def get_df_hist_number_source_per_day(cur_analyzer, days):
    import pandas as pd
    dfs = []
    for day in days:
        was_source_number, number_people = cur_analyzer.neo_strike_numbers[day]
        df = pd.DataFrame({
            'day': [day] * was_source_number.size, 
            'X': was_source_number,
            'Y': number_people})
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def plotly_hist_number_source_per_day(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_hist_number_source_per_day(cur_analyzer, days[:-3])
    # Plot
    fig = px.bar(df, 
        x='X',
        y="Y",
        animation_frame='day',
        labels={"X": "Number of agents infected by a source",
                "day": "Day",
                "Y": "Number of source agents"
                },
        title=make_bold('Susceptible infected by one source')
        )
    degree = int(1 + np.log10(np.max(df['Y'])))
    fig.update_layout(yaxis_range=[0, 10 ** degree])
    fig.update_yaxes(type='log', range=[0, degree])

    if do_show:
        fig.show()
    return fig


def get_df_hist_number_source_cum(cur_analyzer, days):
    import pandas as pd
    dfs = []
    for day in days:
        was_source_number, number_people = cur_analyzer.neo_cum_strike_numbers[day]
        df = pd.DataFrame({
            'day': [day] * was_source_number.size, 
            'X': was_source_number,
            'Y': number_people})
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def plotly_hist_number_source_cum(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_hist_number_source_cum(cur_analyzer, days[:-3])
    # Plot
    fig = px.bar(df, 
        x='X',
        y="Y",
        animation_frame='day',
        labels={"X": "Number of agents infected by a source",
                "day": "Day",
                "Y": "Number of source agents"
                },
        title=make_bold("Cumulative histogram number source")
        )
    degree = int(1 + np.log10(np.max(df['Y'])))
    fig.update_layout(yaxis_range=[0, 10 ** degree])
    fig.update_yaxes(type='log', range=[0, degree])

    if do_show:
        fig.show()
    return fig

def get_df_not_infected_people_by_sus(cur_analyzer, days):
    import pandas as pd
    x = cur_analyzer.bounds
    dfs = []
    for day in days:
        y = [cur_analyzer.naive_by_sus_agegroup_list[day][i] / cur_analyzer.sizes_of_box for i in range(10)]
        for i in range(10):
            df = pd.DataFrame({
                'day': [day] * x.size, 
                'Relative susceptibility': x,
                'Age-group': [f"({i * 10}, {(i + 1) * 10})"] * x.size, 
                'Y': y[i]})
            dfs.append(df)
    return pd.concat(dfs, axis=0)

def plotly_not_infected_people_by_sus(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_not_infected_people_by_sus(cur_analyzer, days)
    # Plot
    fig = px.bar(df, 
        x='Relative susceptibility',
        y="Y",
        color="Age-group",
        barmode='stack',
        animation_frame='day',
        labels={'Relative susceptibility': 'Relative susceptibility',
                "day": "Day",
                "Y": "Fraction of all agents"
                },
        title=make_bold("Uninfected population per susceptibility group")
        )
    if do_show:
        fig.show()
    return fig


def get_df_not_infected_people_by_sus_norm(cur_analyzer, days):
    import pandas as pd
    x = cur_analyzer.bounds
    dfs = []
    for day in days:
        y = np.array([cur_analyzer.naive_by_sus_agegroup_list[day][i] for i in range(10)])
        sums = np.zeros(y.shape[1], dtype=int)
        for age_ind in range(10):
            for rel_sus_ind in range(y[age_ind].size):
                sums[rel_sus_ind] += y[age_ind, rel_sus_ind]
        zz = np.zeros((y.shape[1]), dtype=np.float)
        for i in range(10):
            df = pd.DataFrame({
                'day': [day] * x.size, 
                'Relative susceptibility': x,
                'Age-group': [f"({i * 10}, {(i + 1) * 10})"] * x.size, 
                'Y': np.divide(y[i], sums, out=zz, where=sums!=0)})
            dfs.append(df)
    return pd.concat(dfs, axis=0)

def plotly_not_infected_people_by_sus_norm(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_not_infected_people_by_sus_norm(cur_analyzer, days)
    # Plot
    fig = px.bar(df, 
        x='Relative susceptibility',
        y="Y",
        color="Age-group",
        barmode='stack',
        animation_frame='day',
        labels={'Relative susceptibility': 'Relative susceptibility',
                "day": "Day",
                "Y": "Fraction of uninfected agents"
                },
        title=make_bold("Age of uninfected population per susceptibility group")
        )
    if do_show:
        fig.show()
    return fig


def get_df_infected_non_infected_group(cur_analyzer, days):
    import pandas as pd
    dfs = []
    for day in days:
        not_infected = cur_analyzer.naive_by_agegroup_list[day] / cur_analyzer.sizes_ages_groups
        infected_vals = 1 - not_infected
        # not infected
        df = pd.DataFrame({
                'day': [day] * not_infected.size,
                'State': ['Non-infected'] * not_infected.size,
                'Age-group': [str(age_bound) for age_bound in cur_analyzer.ages_bounds_new[:-1]], 
                'Y': not_infected})
        dfs.append(df)
        # infected
        df = pd.DataFrame({
                'day': [day] * infected_vals.size,
                'State': ['Infected'] * infected_vals.size,
                'Age-group': [str(age_bound) for age_bound in cur_analyzer.ages_bounds_new[:-1]], 
                'Y': infected_vals})
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def plotly_infected_non_infected_group(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_infected_non_infected_group(cur_analyzer, days)
    # Plot
    fig = px.bar(df, 
        x='Age-group',
        y="Y",
        color="State",
        barmode='stack',
        animation_frame='day',
        labels={"Age-group": "Age-group",
                "day": "Day",
                "Y": "Fraction"
                },
        title=make_bold("Fraction of uninfected agents per age group")
        )
    if do_show:
        fig.show()
    return fig


def plotly_viral_load_per_day(sims, do_show=False):
    go = import_plotly() # Load Plotly
    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]
    ind2key = ['home', 'school', 'work', 'random', 'l', 'all']
    ind2color = ["255, 255, 0", "255, 0, 255", "0, 0, 255", "0, 255, 255", "0, 127, 255", '255, 0, 0']
    fig = go.Figure()

    for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
        cur_analyzer= sim.get_analyzer('seir')
        days = sim.tvec
        day_count = len(days)
        x_time = np.arange(day_count)
        base_name = "Pathogen fraction" if sim.pars['is_additive_formula'] else "Agents"
        is_showlegend = (i == sims_count - 1)
        
        choice1 = 15 * [False]
        choice1[:5] = 5 * [True]
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=np.array(cur_analyzer.viral_load_by_layers[j]),
                hovertext=f"{sim.label}: {base_name} {name}",
                hoverinfo="text",
                name=f"{sim.label}: {base_name} {name}",
                    showlegend=is_showlegend,
                    line=dict(
                        color=f'rgba({ind2color[j]}, {brightness})',
                        width=2  # Ширина линии
                    ),
                visible=False))
        fig.add_trace(go.Scatter(x=x_time, y=np.array(cur_analyzer.viral_load_by_layers[1] + cur_analyzer.viral_load_by_layers[2]), visible=False, 
            hovertext=f"{sim.label}: {base_name} school\'&\'work",
            hoverinfo="text",
            name=f"{sim.label}: {base_name} school\'&\'work",
                    showlegend=is_showlegend,
                    line=dict(
                        color=f'rgba({ind2color[4]}, {brightness})',
                        width=2  # Ширина линии
                    )))
        choice2 = 15 * [False]
        choice2[5:10] = 5 * [True]
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=get_window_data(np.array(cur_analyzer.viral_load_by_layers[j])), 
                hovertext=f"{sim.label}: {base_name} {name}",
                hoverinfo="text",
                name=f"{sim.label}: {base_name} {name}",
                showlegend=is_showlegend,
                line=dict(
                    color=f'rgba({ind2color[j]}, {brightness})',
                    width=2  # Ширина линии
                )))
        fig.add_trace(go.Scatter(x=x_time, y=get_window_data(np.array(cur_analyzer.viral_load_by_layers[1] + cur_analyzer.viral_load_by_layers[2])), 
            hovertext=f"{sim.label}: {base_name} school\'&\'work",
            hoverinfo="text",
            name=f"{sim.label}: {base_name} school\'&\'work",
                    showlegend=is_showlegend,
                    line=dict(
                        color=f'rgba({ind2color[4]}, {brightness})',
                        width=2  # Ширина линии
                    )))
        choice3 = 15 * [False]
        choice3[10:15] = 5 * [True]    
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=double_window(np.array(cur_analyzer.viral_load_by_layers[j])), 
                hovertext=f"{sim.label}: {base_name} {name}",
                hoverinfo="text",
                name=f"{sim.label}: {base_name} {name}",
                showlegend=is_showlegend,
                line=dict(
                    color=f'rgba({ind2color[j]}, {brightness})',
                    width=2  # Ширина линии
                ), visible=False))
        fig.add_trace(go.Scatter(x=x_time, y=double_window(np.array(cur_analyzer.viral_load_by_layers[1] + cur_analyzer.viral_load_by_layers[2])), visible=False, 
            hovertext=f"{sim.label}: {base_name} school\'&\'work",
            hoverinfo="text",
            name=f"{sim.label}: {base_name} school\'&\'work",
                    showlegend=is_showlegend,
                    line=dict(
                        color=f'rgba({ind2color[4]}, {brightness})',
                        width=2  # Ширина линии
                    )))
        

    name_title = make_bold("Pathogen fraction per day on each layers" if sim.pars['is_additive_formula'] else "New infections at layers")
    fig.update_layout(title={'text': name_title}, 
                      yaxis_title=base_name, autosize=True,
                      xaxis_title='Day', 
                      updatemenus=[
                            dict(
                                active=1,
                                buttons=list([
                                    dict(label="Raw",
                                        method="update",
                                        args=[{"visible": choice1* sims_count}]),
                                    dict(label="Window",
                                        method="update",
                                        args=[{"visible": choice2 * sims_count}]),
                                    dict(label="Double Window",
                                        method="update",
                                        args=[{"visible": choice3 * sims_count}]),
                        
                                ]),
                                x=1.15,  # Устанавливаем положение кнопки по горизонтали (1 = справа)
                                xanchor="right",  # Устанавливаем якорную точку кнопки справа
                                y=1.15,  # Устанавливаем положение кнопки по вертикали
                                yanchor="top"  # Устанавливаем якорную точку кнопки сверху
                            )
                        ],
                      **plotly_legend) 
    
    if do_show:
        fig.show()

    return fig


def plotly_sars(sims, do_show=False):
    go = import_plotly() # Load Plotly
    ind2key = ['home', 'school', 'work', 'random', 'l', 'all'] 
    ind2color = ["255, 255, 0", "255, 0, 255", "0, 0, 255", "0, 255, 255", "255, 0, 0", "255, 127, 0"]
    fig = go.Figure()
    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]
    for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
        days = sim.tvec
        day_count = len(days)
        x_time = np.arange(day_count)
        cur_analyzer= sim.get_analyzer('seir')
        choice1 = [False] * 12
        choice1[0:4] = 4 * [True]
        is_showlegend = (i == sims_count - 1)
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=np.array(cur_analyzer.sars[j]), 
                hovertext=f"{sim.label}: SAR {name}",
                hoverinfo="text",
                name=f"{sim.label}: SAR {name}",
                showlegend=is_showlegend,
                line=dict(
                    color=f'rgba({ind2color[j]}, {brightness})',
                    width=2  # Ширина линии
                ), visible=False))
        choice2 = [False] * 12
        choice2[4:8] = 4 * [True]
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=get_window_data(np.array(cur_analyzer.sars[j])),
                hovertext=f"{sim.label}: SAR {name}",
                hoverinfo="text",
                name=f"{sim.label}: SAR {name}",
                    showlegend=is_showlegend,
                    line=dict(
                        color=f'rgba({ind2color[j]}, {brightness})',
                        width=2  # Ширина линии
                    )))
        choice3 = [False] * 12
        choice3[8:] = 4 * [True]
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=double_window(np.array(cur_analyzer.sars[j])), 
                        hovertext=f"{sim.label}: SAR {name}",
                        hoverinfo="text",
                        name=f"{sim.label}: SAR {name}",
                        showlegend=is_showlegend,
                        line=dict(
                            color=f'rgba({ind2color[j]}, {brightness})',
                            width=2  # Ширина линии
                        ), visible=False))

    fig.update_layout(title={'text': make_bold('Secondary attack rate at layers')}, 
                      yaxis_title='SAR', autosize=True,
                      xaxis_title='Day', 
                      updatemenus=[
                            dict(
                                active=1,
                                buttons=list([
                                    dict(label="Raw",
                                        method="update",
                                        args=[{"visible": choice1 * sims_count}]),
                                    dict(label="Window",
                                        method="update",
                                        args=[{"visible": choice2 * sims_count}]),
                                    dict(label="Double Window",
                                        method="update",
                                        args=[{"visible": choice3 * sims_count}]),
                        
                                ]),
                                x=1.15,  # Устанавливаем положение кнопки по горизонтали (1 = справа)
                                xanchor="right",  # Устанавливаем якорную точку кнопки справа
                                y=1.15,  # Устанавливаем положение кнопки по вертикали
                                yanchor="top"  # Устанавливаем якорную точку кнопки сверху
                            )
                        ],
                       **plotly_legend) 
    
    if do_show:
        fig.show()

    return fig

def plotly_viral_load_cum(sims, do_show=False):
    go = import_plotly() # Load Plotly
    
    ind2key = ['home', 'school', 'work', 'random', 'l', 'all'] 
    fig = go.Figure()

    sims_count = len(sims)
    brightnesses = np.linspace(0, 1, sims_count + 1)[1:]
    ind2color = ["255, 255, 0", "255, 0, 255", "0, 0, 255", "0, 255, 255", "0, 127, 255", "127, 127, 255"]
    for (i, (sim, brightness)) in enumerate(zip(sims, brightnesses)):
        base_name = "Pathogen fraction" if sim.pars['is_additive_formula'] else "Agents"
        cur_analyzer= sim.get_analyzer('seir')
        days = sim.tvec
        day_count = len(days)
        x_time = np.arange(day_count)
        is_showlegend = (i == sims_count - 1)
        for (j, name) in enumerate(ind2key):
            fig.add_trace(go.Scatter(x=x_time, y=np.cumsum(np.array(cur_analyzer.viral_load_by_layers[j])),
                hovertext=f"{sim.label}: {base_name} {name}",
                hoverinfo="text",
                name=f"{sim.label}: {base_name} {name}",
                showlegend=is_showlegend,
                line=dict(
                    color=f'rgba({ind2color[j]}, {brightness})',
                    width=2  # Ширина линии
                )))
        fig.add_trace(go.Scatter(x=x_time, y=np.cumsum(np.array(cur_analyzer.viral_load_by_layers[1] + cur_analyzer.viral_load_by_layers[2])), 
            showlegend=is_showlegend,
            line=dict(
                    color=f'rgba({ind2color[4]}, {brightness})',
                    width=2  # Ширина линии
                ),
            name=f"{sim.label}: {base_name} school\'&\'work",
            hoverinfo="text",
            hovertext=f"{sim.label}: {base_name} school\'&\'work"))
        
    name_title = make_bold("Cumulative pathogen fraction on each layers" if sim.pars['is_additive_formula'] else "Cumulative infections at layers")
    fig.update_layout(title={'text': name_title}, 
                      yaxis_title=base_name, autosize=True,
                      xaxis_title='Day',  **plotly_legend) 
    
    if do_show:
        fig.show()

    return fig


def get_data_exposed_by_sus_by_ages(cur_analyzer, day):
    x = cur_analyzer.bounds
    y_age = cur_analyzer.exposed_by_sus_list[day] / cur_analyzer.sizes_of_box

    y_age_0 = cur_analyzer.exposed_by_sus_by_ages_list[day][0] / cur_analyzer.sizes_of_box_by_ages[0]
    y_age_1 = cur_analyzer.exposed_by_sus_by_ages_list[day][1] / cur_analyzer.sizes_of_box_by_ages[1]
    y_age_2 = cur_analyzer.exposed_by_sus_by_ages_list[day][2] / cur_analyzer.sizes_of_box_by_ages[2]
    y_age_3 = cur_analyzer.exposed_by_sus_by_ages_list[day][3] / cur_analyzer.sizes_of_box_by_ages[3]

    return {"x": x, "y": [y_age, y_age_0, y_age_1, y_age_2, y_age_3]}


def get_df_exposed_by_sus_by_ages(cur_analyzer, days):
    import pandas as pd
    dfs = []
    ind2age_group = ['all', 'age (0, 7)', 'age (8, 25)', 'age (25, 60)', 'age (60, 100)'] 
    for day in days:
        data = get_data_exposed_by_sus_by_ages(cur_analyzer, day)
        for i in range(5):
            df = pd.DataFrame({
                'day': [day] * data['x'].size,
                'name': [ind2age_group[i]] * data['x'].size,
                'x': data['x'],
                'Y': data['y'][i]})
            dfs.append(df)
    return pd.concat(dfs, axis=0)


def plotly_risk_infected_by_age_group_cum(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_exposed_by_sus_by_ages(cur_analyzer, days)
    # Plot
    fig = px.line(df, 
        x='x',
        y="Y",
        color="name",
        animation_frame='day',
        labels={"x": "Relative susceptibility",
                "day": "Day",
                "Y": "Risk",
                'name': "Age groups"
                },
        title=make_bold("Cumulative infection risk per susceptibility and age")
        )
    fig.update_layout(yaxis_range=[0, 1.1 * np.max(df['Y'])])

    if do_show:
        fig.show()
    return fig


def get_data_exposed_by_sus_by_ages_per_day(cur_analyzer, day):
    x = cur_analyzer.bounds
    y_age = cur_analyzer.exposed_by_sus_per_day_list[day] / cur_analyzer.sizes_of_box

    y_age_0 = cur_analyzer.exposed_by_sus_by_ages_per_day_list[day][0] / cur_analyzer.sizes_of_box_by_ages[0]
    y_age_1 = cur_analyzer.exposed_by_sus_by_ages_per_day_list[day][1] / cur_analyzer.sizes_of_box_by_ages[1]
    y_age_2 = cur_analyzer.exposed_by_sus_by_ages_per_day_list[day][2] / cur_analyzer.sizes_of_box_by_ages[2]
    y_age_3 = cur_analyzer.exposed_by_sus_by_ages_per_day_list[day][3] / cur_analyzer.sizes_of_box_by_ages[3]

    return {"x": x, "y": [y_age, y_age_0, y_age_1, y_age_2, y_age_3]}


def get_df_exposed_by_sus_by_ages_per_day(cur_analyzer, days):
    import pandas as pd
    dfs = []
    ind2age_group = ['all', 'age (0, 7)', 'age (8, 25)', 'age (25, 60)', 'age (60, 100)'] 
    for day in days:
        data = get_data_exposed_by_sus_by_ages_per_day(cur_analyzer, day)
        for i in range(5):
            df = pd.DataFrame({
                'day': [day] * data['x'].size,
                'name': [ind2age_group[i]] * data['x'].size,
                'x': data['x'],
                'Y': data['y'][i]})
            dfs.append(df)
    return pd.concat(dfs, axis=0)


def plotly_risk_infected_by_age_group_per_day(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_exposed_by_sus_by_ages_per_day(cur_analyzer, days)
    # Plot
    fig = px.line(df, 
        x='x',
        y="Y",
        color="name",
        animation_frame='day',
        labels={"x": "Relative susceptibility",
                "day": "Day",
                "Y": "Risk",
                'name': "Age groups"
                },
        title=make_bold("Infection risk per susceptibility and age")
        )
    fig.update_layout(yaxis_range=[0, 1.1 * np.max(df['Y'])])

    if do_show:
        fig.show()
    return fig


def get_df_contact_to_sus_trans(cur_analyzer, days):
    import pandas as pd
    dfs = []
    x = cur_analyzer.rel_sus_trans[:2000]
    y = cur_analyzer.people2contact_count[:2000]
    
    number2state = ['healthy', 'severe', 'recovered', 'exposed', 'dead', 'critical']

    for day in days:
        state_statistics = cur_analyzer.state_statistics[day][:2000]
        for (i, state) in enumerate(number2state):
            total_x = x[state_statistics == i]
            total_y = y[state_statistics == i]
            df = pd.DataFrame({
                'day': [day] * total_x.size,
                'state': [state] * total_x.size,
                'x': total_x,
                'Y': total_y
            })
            dfs.append(df)
    return pd.concat(dfs, axis=0)

def plotly_contact_to_sus_trans(sim, do_show=False):
    import plotly.express as px
    cur_analyzer= sim.get_analyzer('seir')
    days = sim.tvec
    df = get_df_contact_to_sus_trans(cur_analyzer, days)
    df['state'][0] = 'severe'
    df['state'][1] = 'recovered' 
    df['state'][2] = 'exposed' 
    df['state'][3] = 'dead' 
    df['state'][4] = 'critical' 

    # Plot
    fig = px.scatter(df, 
        x='x',
        y="Y",
        animation_frame='day',
        color='state',
        opacity=0.5,
        labels={"x": "Relative susceptibility",
                "day": "Day",
                "Y": "Contacts per day",
                'state': "State"
                },
        title=make_bold("Population composition")
        )

    if do_show:
        fig.show()
    return fig

def plotly_animate(sim, do_show=False): # pragma: no cover
    ''' Plot an animation of each person in the sim '''

    go = import_plotly() # Load Plotly
    z, states = get_individual_states(sim)

    min_color = min(states, key=lambda x: x['value'])['value']
    max_color = max(states, key=lambda x: x['value'])['value']
    colorscale = [[x['value'] / max_color, x['color']] for x in states]

    aspect = 5
    y_size = int(np.ceil((z.shape[0] / aspect) ** 0.5))
    x_size = int(np.ceil(aspect * y_size))

    z = np.pad(z, ((0, x_size * y_size - z.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)

    days = sim.tvec

    fig_dict = {
        'data': [],
        'layout': {},
        'frames': []
    }

    fig_dict['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 200, 'redraw': True},
                                    'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                      'mode': 'immediate',
                                      'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Day: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 200},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data
    fig_dict['data'] = [go.Heatmap(z=np.reshape(z[:, 0], (y_size, x_size)),
                                   zmin=min_color,
                                   zmax=max_color,
                                   colorscale=colorscale,
                                   showscale=False,
                                   )]

    for state in states:
        fig_dict['data'].append(go.Scatter(x=[None], y=[None], mode='markers',
                                           marker=dict(size=10, color=state['color']),
                                           showlegend=True, name=state['name']))

    # make frames
    for i, day in enumerate(days):
        frame = {'data': [go.Heatmap(z=np.reshape(z[:, i], (y_size, x_size)))],
                 'name': i}
        fig_dict['frames'].append(frame)
        slider_step = {'args': [
            [i],
            {'frame': {'duration': 5, 'redraw': True},
             'mode': 'immediate', }
        ],
            'label': i,
            'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    fig_dict['layout']['sliders'] = [sliders_dict]

    fig = go.Figure(fig_dict)

    fig.update_layout(
    autosize=True,
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            automargin=True,
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
    )


    fig.update_layout(title={'text': 'Epidemic over time'}, **plotly_legend)

    if do_show:
        fig.show()

    return fig
