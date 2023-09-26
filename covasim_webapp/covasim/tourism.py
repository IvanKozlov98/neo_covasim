import covasim as cv
import numpy as np


class TourismLayer(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self):
        ''' Convert an existing layer to a custom layer and store contact data '''
        self.label = "tourism layer"

    def update(self, people, ratio=1.0):
        ''' Update the contacts '''
        # будем обновлять только контакты туристов
        # pos_tourists = people[thr:]
        # tourists = pos_tourists[pos_tourists.in_city]
        # n_contacts = len(tourists) # Total number of contacts
        # n_new = int(np.round(n_contacts*frac))
        # self['p1'] = random_choose(tourists.uid, n_new)
        # self['p2'] = random_choose(tourists.uid, n_new)
        # self['beta'] = np.ones(n_new, dtype=cvd.default_float)






        # pop_size = len(people)
        # n_new = self.contact_data[people.t] # Pull out today's contacts
        # self['p1']   = np.array(cv.choose_r(max_n=pop_size, n=n_new), dtype=cv.default_int) # Choose with replacement
        # self['p2']   = np.array(cv.choose_r(max_n=pop_size, n=n_new), dtype=cv.default_int) # Paired contact
        # self['beta'] = np.ones(n_new, dtype=cv.default_float) # Per-contact transmission (just 1.0)


        # # Choose how many contacts to make
        # pop_size   = len(people) # Total number of people
        # n_contacts = len(self) # Total number of contacts
        # n_new = int(np.round(n_contacts*frac)) # Since these get looped over in both directions later
        # inds = cvu.choose(n_contacts, n_new)
        #
        # # Create the contacts, not skipping self-connections
        # self['p1'][inds]   = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=cvd.default_int) # Choose with replacement
        # self['p2'][inds]   = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=cvd.default_int)
        # self['beta'][inds] = np.ones(n_new, dtype=cvd.default_float)

        return

