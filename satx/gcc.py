"""
The Global Constraint Catalog implementation for SAT-X.
"""

import satx


def abs_val(x, y):
    """
    Enforce the fact that the first variable is equal to the absolute value of the second variable.
    """
    assert y >= 0
    assert abs(x) == y


def all_differ_from_at_least_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from at least K positions.
    """
    nil1 = satx.integer()
    nil2 = satx.integer()
    for V in lst:
        nil1.is_not_in(V)
        nil2.is_not_in(V)
    assert nil1 != nil2
    T = satx.tensor(dimensions=(len(lst[0]),))
    assert sum(T[[i]](0, 1) for i in range(len(lst[0]))) >= k
    for i1 in range(len(lst) - 1):
        for i2 in range(i1 + 1, len(lst)):
            for j in range(len(lst[0])):
                assert T[[j]](nil1, lst[i1][j]) != T[[j]](nil2, lst[i2][j])


def all_differ_from_at_most_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from at most K positions.
    """
    nil1 = satx.integer()
    nil2 = satx.integer()
    for V in lst:
        nil1.is_not_in(V)
        nil2.is_not_in(V)
    assert nil1 == nil2
    T = satx.tensor(dimensions=(len(lst[0]),))
    assert sum(T[[i]](0, 1) for i in range(len(lst[0]))) >= len(lst[0]) - k
    for i1 in range(len(lst) - 1):
        for i2 in range(i1 + 1, len(lst)):
            for j in range(len(lst[0])):
                assert T[[j]](nil1, lst[i1][j]) == T[[j]](nil2, lst[i2][j])


def all_differ_from_exactly_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from exactly K positions. Enforce K = 0 when |VECTORS| < 2.
    """
    all_differ_from_at_least_k_pos(k, lst)
    all_differ_from_at_most_k_pos(k, lst)


def all_equal(lst):
    """
    Enforce all variables of the collection
    """
    satx.apply_dual(lst, lambda x, y: x == y)


def all_different(lst):
    """
    Enforce all variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 to take distinct values.
    """
    satx.all_different(lst)


def element(idx, lst, val):
    """
    𝚅𝙰𝙻𝚄𝙴 is equal to the 𝙸𝙽𝙳𝙴𝚇-th item of 𝚃𝙰𝙱𝙻𝙴, i.e. 𝚅𝙰𝙻𝚄𝙴 = 𝚃𝙰𝙱𝙻𝙴[𝙸𝙽𝙳𝙴𝚇].
    """
    assert val == satx.index(idx, lst)


def gcd(x, y, z):
    """
    Enforce the fact that 𝚉 is the greatest common divisor of 𝚇 and 𝚈. (assume X <= Y)
    """
    assert 0 < x <= y
    assert z > 0
    assert z == y % x
    assert (x / z) % (y % z) == 0


def sort(lst1, lst2):
    """
    First, the variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 correspond to a permutation of the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 1. Second, the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 are sorted in increasing order.
    """
    _, ys = satx.permutations(lst1, len(lst1))
    satx.apply_single(lst2, lambda i, t: t == ys[i], indexed=True)
    satx.apply_dual(lst2, lambda a, b: a <= b)
