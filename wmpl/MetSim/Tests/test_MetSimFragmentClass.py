""" Tests of the class of the MetSim fragments, which is a Cython class with typed fields.

The tests check that the fragment has every attribute of the Python class it replaced (the class in
MetSimErosionReference.py), with the same values after init(), and that spawning a child, copying,
pickling, loading a fragment pickled by the Python class and saving it as JSON keep all of them.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimFragmentClass.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import json
import pickle

import wmpl.MetSim.MetSimErosion as MetSimErosion
import wmpl.MetSim.Tests.MetSimErosionReference as MetSimErosionReference
from wmpl.Dynesty.DynestyMetSim import jsonDefault
from wmpl.MetSim.Tests.DefaultFragment import initFragment


# Attributes which are integers, booleans and objects in the Python class, and keep their type
INT_ATTRIBUTES = ['id', 'n_grains']
BOOL_ATTRIBUTES = ['disruption_child', 'erosion_height_change_applied', 'erosion_enabled', \
    'disruption_enabled', 'active', 'erosion_coeff_fixed', 'main', 'grain', 'complex']
OBJECT_ATTRIBUTES = ['const', 'erosion_bin_counts', 'erosion_bin_params', 'complex_id']



def assertSameAttributes(frag, frag_ref, names):
    """ Check that two fragments have the same values of the given attributes, numbers to the last bit.

    Arguments:
        frag: [object] Fragment.
        frag_ref: [object] Reference fragment.
        names: [list] Names of the attributes.

    Return:
        None
    """

    for name in names:

        value, value_ref = getattr(frag, name), getattr(frag_ref, name)

        if isinstance(value_ref, (int, float)) and not isinstance(value_ref, bool):
            assert float(value).hex() == float(value_ref).hex(), name

        else:
            assert value == value_ref, name



def testFragmentHasTheAttributesOfThePythonClass():
    """ Check that a new fragment has every attribute of the Python class, with the same value, and the same
        type for the integers, the booleans and the objects. """

    frag = MetSimErosion.Fragment()
    frag_ref = MetSimErosionReference.Fragment()

    names = list(vars(frag_ref))
    assertSameAttributes(frag, frag_ref, names)

    for name in INT_ATTRIBUTES:
        assert type(getattr(frag, name)) is int, name

    for name in BOOL_ATTRIBUTES:
        assert type(getattr(frag, name)) is bool, name

    for name in OBJECT_ATTRIBUTES:
        assert getattr(frag, name) is None, name



def testInitGivesTheValuesOfThePythonClass():
    """ Check that init() gives the same values as the Python class to the last bit. """

    frag, const = initFragment(MetSimErosion)
    frag_ref = MetSimErosionReference.Fragment()
    frag_ref.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, \
        const.zenith_angle, const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)

    assertSameAttributes(frag, frag_ref, list(vars(frag_ref)))
    assert frag.const is const

    # The shape-density coefficient after a change of the density
    frag.rho = frag_ref.rho = 3500.0
    frag.updateShapeDensityCoeff()
    frag_ref.updateShapeDensityCoeff()
    assert frag.K.hex() == float(frag_ref.K).hex()



def testSpawnChildCopiesTheFragment():
    """ Check that a child has every value of its parent, shares the constants, starts without the mass bins
        of the parent, and keeps attributes added to the parent. """

    frag, const = initFragment(MetSimErosion)
    frag.complex = True
    frag.complex_id = 3
    frag.n_grains = 7
    frag.erosion_bin_counts = [1.5, 2.0]
    frag.erosion_bin_params = (2.0, 1e-11, 1e-9, 1e-5, 10)
    frag.extra_attribute = 'kept'

    child = frag.spawn_child()

    assert type(child) is MetSimErosion.Fragment
    names = [name for name in vars(MetSimErosionReference.Fragment()) \
        if name not in ['erosion_bin_counts', 'erosion_bin_params']] + ['h', 'gamma', 'zenith_angle']
    assertSameAttributes(child, frag, names)
    assert child.const is const
    assert child.erosion_bin_counts is None
    assert child.erosion_bin_params is None
    assert child.extra_attribute == 'kept'

    # The child has its own attributes
    child.extra_attribute = 'changed'
    child.m = 0.0
    assert frag.extra_attribute == 'kept'
    assert frag.m == const.m_init



def testCopyAndPickleKeepTheFragment():
    """ Check that a deep copy and a pickled fragment keep all values, also the added attributes. """

    frag, _ = initFragment(MetSimErosion)
    frag.complex = True
    frag.n_grains = 3
    frag.erosion_bin_counts = [0.5]
    frag.extra_attribute = [1, 2]

    names = list(vars(MetSimErosionReference.Fragment())) + ['h', 'gamma', 'zenith_angle', 'extra_attribute']
    names.remove('const')

    for frag_copy in [copy.deepcopy(frag), pickle.loads(pickle.dumps(frag))]:

        assert type(frag_copy) is MetSimErosion.Fragment
        assertSameAttributes(frag_copy, frag, names)
        assert frag_copy.const.m_init == frag.const.m_init



def testLoadFragmentPickledByThePythonClass():
    """ Check that a fragment pickled when Fragment was a Python class is loaded with all of its
        attributes. """

    frag_ref, _ = initFragment(MetSimErosionReference)
    frag_ref.complex = True
    frag_ref.n_grains = 5
    frag_ref.extra_attribute = 'old'

    # The state of an object of a Python class is its __dict__, which the new class is given on loading
    frag = MetSimErosion.Fragment.__new__(MetSimErosion.Fragment)
    frag.__setstate__(copy.deepcopy(vars(frag_ref)))

    names = list(vars(frag_ref))
    names.remove('const')
    assertSameAttributes(frag, frag_ref, names)
    assert frag.const.m_init == frag_ref.const.m_init



def testFragmentSavedAsJSONHasAllAttributes():
    """ Check that toDict() gives every attribute of the Python class, also the added ones, and that the JSON
        export of Dynesty saves them, although most of them are not in the __dict__ of the fragment. """

    frag, const = initFragment(MetSimErosion)
    frag.complex = True
    frag.n_grains = 4
    frag.extra_attribute = 'kept'

    frag_ref, _ = initFragment(MetSimErosionReference)

    frag_dict = frag.toDict()
    names = list(vars(frag_ref))
    assert list(frag_dict)[:len(names)] == names
    assert frag_dict['const'] is frag.const
    assert frag_dict['complex'] is True
    assert frag_dict['n_grains'] == 4
    assert frag_dict['extra_attribute'] == 'kept'

    saved = json.loads(json.dumps(frag, default=jsonDefault))
    for name in ['m', 'v', 'h', 'K', 'n_grains', 'complex', 'extra_attribute']:
        assert saved[name] == frag_dict[name], name
    assert saved['const']['m_init'] == const.m_init
