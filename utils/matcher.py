from collections import defaultdict


class Matcher:
    def __init__(self, men, women):
        '''
        Constructs a Matcher instance.
        Takes a pandas.DataFrame `men` and a pandas.DataFrame `women`.
        '''
        self.M = men
        self.W = women
        self.wives = {}
        self.pairs = []

        # we index spousal preferences at initialization
        # to avoid expensive lookups when matching
        self.mrank = defaultdict(dict)  # `mrank[m][w]` is m's ranking of w
        self.wrank = defaultdict(dict)  # `wrank[w][m]` is w's ranking of m

        ncol = self.M.shape[1]
        for _, row in self.M.iterrows():
            for j in range(1, ncol):
                self.mrank[row['id']][row[str(j)]] = j

        ncol = self.W.shape[1]
        for _, row in self.W.iterrows():
            for j in range(1, ncol):
                self.wrank[row['id']][row[str(j)]] = j

    def prefers(self, w, m, h):
        '''Test whether w prefers m over h.'''
        return self.wrank[w][m] < self.wrank[w][h]

    def after(self, m, w):
        '''Return the woman favored by m after w.'''
        i = self.mrank[m][w] + 1  # index of woman following w in list of prefs
        return self.M.loc[self.M['id'] == m][str(i)].values[0]

    def match(self, men=None, next=None, wives=None):
        '''
        Try to match all men with their next preferred spouse.
        '''
        if men is None:
            men = list(self.M['id'].values)  # get the complete list of men
        if next is None:
            # if not defined, map each man to their first preference
            next = dict((row['id'], row['1']) for m, row in self.M.iterrows())
        if wives is None:
            wives = {}  # mapping from women to current spouse
        if not len(men):
            self.pairs = [(h, w) for w, h in wives.items()]
            self.wives = wives
            return wives
        m, men = list(men)[0], list(men)[1:]
        w = next[m]  # next woman for m to propose to
        next[m] = self.after(m, w)  # woman after w in m's list of prefs
        if w in wives:
            h = wives[w]  # current husband
            if self.prefers(w, m, h):
                men.append(h)  # husband becomes available again
                wives[w] = m  # w becomes wife of m
            else:
                men.append(m)  # m remains unmarried
        else:
            wives[w] = m  # w becomes wife of m
        # print('m: {0}, w: {1}, men: {2}'.format(m, w, men))
        return self.match(men, next, wives)

    def is_stable(self, wives=None, verbose=False):
        if wives is None:
            wives = self.wives
        for w, m in wives.items():
            mrank = list(self.M.loc[self.M['id'] == m].values[0][1:])
            i = mrank.index(w)
            preferred = mrank[:i]
            for p in preferred:
                h = wives[p]
                wrank = list(self.W.loc[self.W['id'] == p].values[0][1:])
                if wrank.index(m) < wrank.index(h):
                    msg = "{}'s marriage to {} is unstable: " + \
                          "{} prefers {} over {} and {} prefers " + \
                          "{} over her current husband {}"
                    if verbose:
                        print(msg.format(m, w, m, p, w, p, m, h))
                    return False
        return True


# import pandas as pd
#
# men = pd.read_csv("men.csv")
# women = pd.read_csv("women.csv")
# print(women)
# matcher = Matcher(women, men)
# wives = matcher.match()
# print(wives)
# print(matcher.pairs)
