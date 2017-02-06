import numpy as np
import math
import sys

L = 256 * 256 * 256 # RGB size

class DisjointSet:
    def __init__(self, n):
        self.parent = np.arange(n, dtype='uint32')
        self.size   = np.ones(  n, dtype='uint32')

        self.history = [None]

    def find(self, p):
        while p != self.parent[p]:
            p = self.parent[p]

        return p

    def join(self, p, q, actually_join):
        u = self.find(p)
        v = self.find(q)

        if u == v:
            return p, q, u, v

        if self.size[u] > self.size[v]:
            u, v = v, u
            p, q = q, p

        if actually_join:
            self.history.append((p, q, u, v, self.size[u], self.size[v]))
            self.size[v] += self.size[u]
            self.parent[u] = v

        return p, q, u, v

    def backtrack(self):
        if len(self.history) == 0:
            print "DisjointSet: nothing to backtrack"
            sys.exit(1)
        p, q, u, v, su, sv = self.history.pop()
        self.parent[u] = u
        self.parent[v] = v
        self.size[u] = su
        self.size[v] = sv

        return (p, q)


class VariationOfInformation:

    gt   = None        # a 3d numpy array, ground truth clusterisation
    test = None        # test clusterization

    history = []       # the history of join operations, list of triples (i, j, new_voi), where new_voi is the voi after mergin (i, j)

    def __init__(self, gt, test):
        """
        https://en.wikipedia.org/wiki/Variation_of_information

        gt - a 3d numpy array, ground truth clusterisation
        test - a test clusterisation

        Assumes that all cluster id's are up to L (RGB colors)
        """
        self.gt   = gt
        self.test = test
        self.history = []

        self.gt_label_map = np.zeros(L)
        self.test_label_map = np.zeros(L)

        self.compress_labels(self.gt, self.gt_label_map)
        self.compress_labels(self.test, self.test_label_map)

        self.evaluate_voi()

        self.disjoint_set = DisjointSet(self.M)

    def compress_labels(self, labels, label_map):
        exists = np.zeros(L, dtype='uint8')
        exists[labels] = 1
        label_map[:] = np.cumsum(exists) * exists
        label_map -= 1
        #labels[:] = label_map[labels]

    def evaluate_voi(self):
        self.N = np.max(self.gt_label_map) + 1       # number of segments in gt
        self.M = np.max(self.test_label_map) + 1     # number of segments in original test

        pairs = np.array(self.test_label_map[self.test] * self.N + self.gt_label_map[self.gt], np.dtype('uint32'))

        # r[t][g] is the number of pixels p, where test[p] == t and gt[p] == g
        self.r = np.zeros((self.M, self.N), dtype='uint32')
        tmp = np.bincount(pairs.ravel())
        self.r.flat[:len(tmp)] = tmp

        r = self.r[:self.M, :]

        self.gt_size   = self.r.sum(axis=0)
        self.test_size = self.r.sum(axis=1)

        self.n = self.gt_size.sum()

        a = r * 1.0 / self.test_size[:self.M, np.newaxis]
        a[r == 0] = 1

        b = r * 1.0 / self.gt_size[np.newaxis, :]
        b[r == 0] = 1

        self.merge_voi = -np.sum(r * np.log(a)) / self.n
        self.split_voi = -np.sum(r * np.log(b)) / self.n
        #self.voi = -np.sum(r * (np.log(a) + np.log(b))) / self.n
        self.voi = self.merge_voi + self.split_voi

        self.history.append((-1, -1, -1, -1, np.array([self.merge_voi, self.split_voi])))

    def merge(self, l1, l2, actually_merge=True):
        """
        merge labels l1 and l2, and return the delta voi

        if actually_merge == False, the method will only return the delta, but will not merge
        """
        #print 'test_label', self.test_label_map[l1], self.test_label_map[l2]

        l1, l2, pl1, pl2 = self.disjoint_set.join(self.test_label_map[l1], self.test_label_map[l2], actually_merge)

        if pl1 == pl2:
            #print "pl", pl1, pl2
            return 0.0 # already merged

        delta = np.zeros(2)
        delta += self.change_row(pl1, -1) # remove row pl1 from the sum
        delta += self.change_row(pl2, -1) # remove row pl2 from the sum

        self.r[pl2] += self.r[pl1]
        self.test_size[pl2] += self.test_size[pl1]

        delta += self.change_row(pl2, 1) # add row pl2 (the total one) to the sum

        if actually_merge:
            self.disjoint_set.join(pl1, pl2, actually_merge)
            self.history.append((l1, l2, pl1, pl2, np.array([self.merge_voi, self.split_voi])))
            self.merge_voi += delta[0]
            self.split_voi += delta[1]
            self.voi += delta[0] + delta[1]
        else:
            self.r[pl2] -= self.r[pl1]
            self.test_size[pl2] -= self.test_size[pl1]

        return delta[0] + delta[1]

    def change_row(self, row, coef):
        a = self.r[row] * 1.0 / self.test_size[row]
        a[self.r[row] == 0] = 1

        b = self.r[row] * 1.0 / self.gt_size
        b[self.r[row] == 0] = 1

        return -coef * 1.0 / self.n * np.array([np.sum(self.r[row] * np.log(a)), np.sum(self.r[row] * np.log(b))])

    def backtrack(self):
        """
        cancel last merge
        """
        l1, l2, pl1, pl2, old_voi = self.history.pop()

        delta = 0.0
        delta += self.change_row(pl2, -1)

        self.r[pl2] -= self.r[pl1]
        self.test_size[pl2] -= self.test_size[pl1]

        delta += self.change_row(pl1, 1)
        delta += self.change_row(pl2, 1)

        self.disjoint_set.backtrack()

        self.merge_voi = old_voi[0]
        self.split_voi = old_voi[1]
        self.voi = self.merge_voi + self.split_voi

        return delta, l1, l2

    def backtrack_all(self):
        """
        cancel all changes
        """
        while len(self.history) > 1:
            self.backtrack()

if __name__=='__main__':
    gt1 = np.array([[1, 1, 1, 5, 5],
                   [1, 1, 5, 5, 8],
                   [2, 1, 2, 5, 8],
                   [2, 2, 2, 8, 8]])

    test1 = np.array([[0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 3],
                     [2, 0, 2, 3, 3],
                     [2, 2, 2, 2, 7]])

    a = np.array([[1, 1, 1, 3]])
    b = np.array([[1, 2, 3, 4]])

    v = VariationOfInformation(a, b)
    print v.voi
    print 1, 2, v.merge(1, 2)
    print v.disjoint_set.parent
    print 1, 3, v.merge(1, 3)
    print v.disjoint_set.parent
    print 2, 4, v.merge(2, 4)
    print v.disjoint_set.parent

    print 2, 4, v.backtrack()
    print v.disjoint_set.parent
    print 1, 3, v.backtrack()
    print v.disjoint_set.parent

    print v.voi

