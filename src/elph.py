'''
Created on Sep 4, 2012

@author: vinnie
'''

import numpy as np
from operator import itemgetter
from collections import Counter
from itertools import chain, combinations

def powersetNoEmpty(iterable):
    '''
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def reliableEntropy(pspace):
    '''
    Entropy with an additional false positive to remove the bias towards low
    frequency hypothesis
    Returns the reliable entropy
    '''
    total_frequency = sum(pspace.values()) + 1.0
    h_rel = -((1.0/total_frequency) * np.log2(1.0/total_frequency))
    for frequency in pspace.itervalues():
        tmp = frequency/total_frequency
        h_rel -= tmp * np.log2(tmp)
    
    return h_rel

def prune(hspace, h_thresh=1.00):
    '''
    Prune the hypothesis space using the entropy as a threshold
    Returns a pruned hypothesis space
    '''
    for key in hspace.keys():
        if reliableEntropy((key, hspace[key])) > h_thresh:
            hspace.pop(key)
    return hspace

def predict(hspace, stm):
    '''
    Given a short term memory and hypothesis space, make a prediction.
    Returns the prediction, STM item used to make the prediction, and entropy
    '''
    stm_matches = [hspace[p] for p in powersetNoEmpty(stm) if hspace.has_key(p)]
    if not stm_matches:
        return None, np.inf

    lowest_entropy = min(stm_matches, key=reliableEntropy)
    h = reliableEntropy(lowest_entropy)
    prediction = max(lowest_entropy.items(), key=itemgetter(1))
    return prediction[0], h

def observe(hspace, stm, observation):
    '''
    Observe and learn a new symbol following the STM.
    Returns the updated hypothesis space.
    '''
    hspace_keys = powersetNoEmpty(stm)
    for key in hspace_keys:
        pspace = hspace.setdefault(key, Counter())
        pspace.update(observation)
    return hspace

def observeSequence(hspace, sequence, stm_size=7):
    '''
    Observe an entire sequence (not online), and return the hypothesis space.
    '''
    hits = 0
    for i in xrange(stm_size, len(sequence)):
        stm = sequence[i-stm_size:i]
        prediction, h = predict(hspace, stm)
        observe(hspace, stm, sequence[i])
        prune(hspace, 1.0)
        if sequence[i] == prediction:
            print "HIT"
            hits += 1
        else:
            print "MISS"
    print "Correct: ", float(hits)/(len(sequence)-stm_size)
    
    return hspace
    
class OnlineELPH:
    
    def __init__(self, stm_size=7, entropy_thresh=1.0):
        """
        Create on online ELPH sequence predictor
        Really just a wrapper and state holder for the pure functions above
        """
        self.hspace = {}
        self.stm = []
        self.stm_size = stm_size
        self.entropy_thresh = entropy_thresh
        return
    
    def observe(self, next_mem, observation):
        """
        Observe a symbol.
        Also updates the STM and prunes the hypothesis space
        """
        observe(self.hspace, self.stm, observation)
        self.stm.append(next_mem)
        if len(self.stm) > self.stm_size:
            self.stm = self.stm[-self.stm_size:]
        return
    
    def predict(self):
        """
        Make a prediction
        """
        return predict(self.hspace, self.stm)
    
def test():
    a = 'ABACABACABACABACABACABACABACABACABACABACABACABAC'
    hspace = {}
    observeSequence(hspace, a)
    return

def test_OnlineELPH():
    elph = OnlineELPH(stm_size=3, entropy_thresh=1.5)
    a = 'AABAABAABAABAABAABAACAACAACAACAACAAC'
    hits = 0
    for x in a:
        p, h = elph.predict()
        elph.observe(x, x)
        if p == x:
            hits += 1
    
    print "HITS:        ", hits
    print "Performance: ", hits/float(len(a))
    return

if __name__ == '__main__':
    test_OnlineELPH()