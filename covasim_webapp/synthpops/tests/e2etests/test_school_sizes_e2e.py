"""
Test that school sizes are being generated by school type when with_school_types is turned on and data is available.
"""

import sciris as sc
import synthpops as sp
import matplotlib as mplt
import cmasher as cmr
import cmocean

mplt.rcParams['font.family'] = 'Roboto Condensed'
mplt.rcParams['font.size'] = 7


# parameters to generate a test population
pars = sc.objdict(
    n                               = 160e3,
    rand_seed                       = 123,
    max_contacts                    = None,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    with_school_types               = 1,

    school_mixing_type              = {'pk': 'age_clustered',
                                       'es': 'age_and_class_clustered',
                                       'ms': 'age_and_class_clustered',
                                       'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
)


def test_plot_school_sizes(do_show, do_save, artifact_dir):
    """
    Test that the school size distribution by type plotting method in sp.Pop
    class works for a large population. This will be a longer test that is
    run as part of our end-to-end testing suite.

    Visually show how the school size distribution generated compares to the
    data for the location being simulated.

    Notes:
        The larger the population size, the better the generated school size
        distributions by school type can match the expected data. If generated
        populations are too small, larger schools will be missed and in
        general there won't be enough schools generated to apply statistical
        tests.
    """
    sp.logger.info("Test that the school size distribution by type plotting method in sp.Pop class works. Note: For small population sizes, the expected and generated size distributions may not match very well given that the model is stochastic and demographics are based on much larger populations.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_school_size_distributions_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    if artifact_dir:
        kwargs.figdir = artifact_dir
    kwargs.screen_height_factor = 0.20
    kwargs.hspace = 0.8
    kwargs.bottom = 0.09
    kwargs.keys_to_exclude = ['uv']
    kwargs.cmap = cmr.get_sub_cmap('cmo.curl', 0.08, 1)

    fig, ax = pop.plot_school_sizes(**kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'End-to-end school sizes check failed.'
    print('Check passed. Figure made.')

    return fig, ax, pop


if __name__ == '__main__':

    # run as main to see the code and figures in action!
    sc.tic()

    fig0, ax0, pop0 = test_plot_school_sizes(do_show=True, do_save=True, artifact_dir='artifact')

    sc.toc()
