import covasim as cv
import numpy as np
from covasim.population import make_random_contacts


class TouristLayer(cv.Layer):

    def __init__(self, tourist_count, number_contact_per_tourist):
        t_contacts = make_random_contacts(tourist_count, number_contact_per_tourist)
        super(TouristLayer, self).__init__(t_contacts)

    def update(self, people, frac=1.0):
        '''
        Regenerate contacts on each timestep.

        Args:
            people (People): the Covasim People object, which is usually used to make new contacts
            frac (float): the fraction of contacts to update on each timestep
        '''
        # Choose how many contacts to make

        pop_size   = people.tourist_count # Total number of tourists
        n_contacts = len(self) # Total number of contacts
        n_new = int(np.round(n_contacts*frac)) # Since these get looped over in both directions later
        inds = cv.choose(n_contacts, n_new)

        # Create the contacts, not skipping self-connections
        self['p1'][inds]   = people.pop_size + np.array(cv.choose_r(max_n=pop_size, n=n_new), dtype=cv.default_int) # Choose with replacement
        self['p2'][inds]   = np.array(cv.choose_r(max_n=pop_size, n=n_new), dtype=cv.default_int)
        self['beta'][inds] = np.ones(n_new, dtype=cv.default_float)
        return