import pytest
from random import Random
from archai.discrete_search.search_spaces.config import (
    ArchConfig, ArchParamTree, ConfigSearchSpace, DiscreteChoice,
    repeat_config
)


@pytest.fixture
def rng():
    return Random(1)


@pytest.fixture
def tree_c1():
    s = DiscreteChoice(['a', 'b', 'c'])

    c = {
        'param1': s,
        'param2': DiscreteChoice(['d', 'e', 'f']),
        'param1_clone': s,

        'sub1': {
            'sub2':{
                'param1_clone': s
            }
        },

        'param_list': repeat_config({
            'param3': DiscreteChoice(['g', 'h', 'i']),
            'param4': repeat_config({
                'constant': 'a',
                'param5': DiscreteChoice(['j', 'k', 'l'])
            }, repeat_times=[3, 4], share_arch=True)
        }, repeat_times=[0, 1, 2], share_arch=False)
    }

    return c


@pytest.fixture
def tree_c2():
    c = {
        'p1': DiscreteChoice(list([False, True])),
        'p2': DiscreteChoice(list([False, True]))
    }

    return c


def test_param_sharing(rng, tree_c1):
    tree = ArchParamTree(tree_c1)

    for _ in range(10):
        config = tree.sample_config(rng)
        p1 = config.pick('param1')
        assert config.get_used_params()['param1']

        p1_c = config.pick('param1_clone')
        assert p1 == p1_c

        p1_c2 = config.pick('sub1').pick('sub2').pick('param1_clone')
        assert p1 == p1_c2


def test_repeat_config_share(rng, tree_c1):
    tree = ArchParamTree(tree_c1)

    for _ in range(10):
        config = tree.sample_config(rng)

        for param_block in config.pick('param_list'):
            par4 = param_block.pick('param4')

            assert len(set(
                p.pick('constant') for p in par4
            )) == 1

            assert len(set(
                p.pick('param5') for p in par4
            )) == 1


def test_default_value():
    a = ArchConfig({
        'p': 2
    })

    assert a.pick('c', default=2) == a.pick('p')


def test_ss(rng, tree_c2, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('test_ss')

    tree = ArchParamTree(tree_c2)

    def use_arch(c):
        if c.pick('p1'):
            return

        if c.pick('p2'):
            return

    cache = []
    for _ in range(2):
        ids = []

        ss = ConfigSearchSpace(use_arch, tree, seed=1)
        m = ss.random_sample()
        ids += [m.archid]
        ss.save_arch(m, tmp_path / 'arch.json')

        m2 = ss.load_arch(tmp_path / 'arch.json')
        assert m.archid == m2.archid

        m3 = ss.mutate(m)

        ids += [m3.archid]
        m4 = ss.crossover([m3, m2])
        ids += [m4.archid]
        cache += [ids]

    # make sure the archid's returned are repeatable so that search jobs can be restartable.
    assert cache[0] == cache[1]


def test_ss_archid(rng, tree_c2):
    tree = ArchParamTree(tree_c2)

    def use_arch(c):
        if c.pick('p1'):
            return

        if c.pick('p2'):
            return

    ss = ConfigSearchSpace(use_arch, tree, seed=1)
    archids = set()

    for _ in range(100):
        config = ss.random_sample()
        archids.add(config.archid)

    assert len(archids) == 3 # Will fail with probability approx 1/2^100
