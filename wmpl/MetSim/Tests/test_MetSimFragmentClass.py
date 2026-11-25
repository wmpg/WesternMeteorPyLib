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

        # Compare the numbers by their bits, so that they are equal only if they are the same double
        if isinstance(value_ref, (int, float)) and not isinstance(value_ref, bool):
            assert float(value).hex() == float(value_ref).hex(), name

        # Compare the other values directly
        else:
            assert value == value_ref, name



def testFragmentHasTheAttributesOfThePythonClass():
    """ Check that a new fragment has every attribute of the Python class, with the same value, and the same
        type for the integers, the booleans and the objects.
    """

    # Make a new fragment of each class
    frag = MetSimErosion.Fragment()
    frag_ref = MetSimErosionReference.Fragment()

    # Compare the values of all attributes of the Python class
    names = list(vars(frag_ref))
    assertSameAttributes(frag, frag_ref, names)

    # Check that the integers, the booleans and the objects keep their types
    for name in INT_ATTRIBUTES:
        assert type(getattr(frag, name)) is int, name

    for name in BOOL_ATTRIBUTES:
        assert type(getattr(frag, name)) is bool, name

    for name in OBJECT_ATTRIBUTES:
        assert getattr(frag, name) is None, name



def testInitGivesTheValuesOfThePythonClass():
    """ Check that init() gives the same values as the Python class to the last bit. """

    # Initialize a fragment of each class with the same constants
    frag, const = initFragment(MetSimErosion)
    frag_ref = MetSimErosionReference.Fragment()
    frag_ref.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, \
        const.zenith_angle, const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)

    # Compare all attributes, and check that the constants are shared
    assertSameAttributes(frag, frag_ref, list(vars(frag_ref)))
    assert frag.const is const

    # The shape-density coefficient after a change of the density
    frag.rho = frag_ref.rho = 3500.0
    frag.updateShapeDensityCoeff()
    frag_ref.updateShapeDensityCoeff()
    assert frag.K.hex() == float(frag_ref.K).hex()



def testSpawnChildCopiesTheFragment():
    """ Check that a child has every value of its parent, shares the constants, starts without the mass bins
        of the parent, and keeps attributes added to the parent.
    """

    # Make a parent with values which differ from the defaults, and an added attribute
    frag, const = initFragment(MetSimErosion)
    frag.complex = True
    frag.complex_id = 3
    frag.n_grains = 7
    frag.erosion_bin_counts = [1.5, 2.0]
    frag.erosion_bin_params = (2.0, 1e-11, 1e-9, 1e-5, 10)
    frag.extra_attribute = 'kept'

    child = frag.spawn_child()

    # Compare all attributes of the child with the parent, except the mass bins
    assert type(child) is MetSimErosion.Fragment
    names = [name for name in vars(MetSimErosionReference.Fragment()) \
        if name not in ['erosion_bin_counts', 'erosion_bin_params']] + ['h', 'gamma', 'zenith_angle']
    assertSameAttributes(child, frag, names)
    assert child.const is const

    # The child starts without the mass bins of the parent, and keeps the added attribute
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

    # Make a fragment with values which differ from the defaults, and an added attribute
    frag, _ = initFragment(MetSimErosion)
    frag.complex = True
    frag.n_grains = 3
    frag.erosion_bin_counts = [0.5]
    frag.extra_attribute = [1, 2]

    # Compare all attributes but the constants, which are copied too
    names = list(vars(MetSimErosionReference.Fragment())) + ['h', 'gamma', 'zenith_angle', 'extra_attribute']
    names.remove('const')

    # Check a deep copy and a pickled fragment
    for frag_copy in [copy.deepcopy(frag), pickle.loads(pickle.dumps(frag))]:

        assert type(frag_copy) is MetSimErosion.Fragment
        assertSameAttributes(frag_copy, frag, names)
        assert frag_copy.const.m_init == frag.const.m_init



def testLoadFragmentPickledByThePythonClass():
    """ Check that a fragment pickled when Fragment was a Python class is loaded with all of its
        attributes.
    """

    # Make a fragment of the Python class with values which differ from the defaults
    frag_ref, _ = initFragment(MetSimErosionReference)
    frag_ref.complex = True
    frag_ref.n_grains = 5
    frag_ref.extra_attribute = 'old'

    # The state of an object of a Python class is its __dict__, which the new class is given on loading
    frag = MetSimErosion.Fragment.__new__(MetSimErosion.Fragment)
    frag.__setstate__(copy.deepcopy(vars(frag_ref)))

    # Compare all attributes, and the copied constants
    names = list(vars(frag_ref))
    names.remove('const')
    assertSameAttributes(frag, frag_ref, names)
    assert frag.const.m_init == frag_ref.const.m_init



def testFragmentPickledWithoutANewFieldIsLoaded():
    """ Check that a fragment is pickled by the names of its attributes, so that a fragment pickled before a
        field was added is loaded, with the default value of the new field.
    """

    frag, _ = initFragment(MetSimErosion)
    frag.rk4_substep = 0.001

    # The pickled state is the dictionary of the attributes
    cls, args, state = frag.__reduce__()
    assert cls is MetSimErosion.Fragment
    assert state == frag.toDict()

    # Remove the substep from the state, as in a fragment pickled before it was added
    del state['rk4_substep']

    # Load it as pickle does
    frag_loaded = cls(*args)
    frag_loaded.__setstate__(state)

    assert frag_loaded.rk4_substep == 0
    assert frag_loaded.m == frag.m
    assert frag_loaded.n_grains == frag.n_grains



def testFragmentSavedAsJSONHasAllAttributes():
    """ Check that toDict() gives every attribute of the Python class, also the added ones, and that the JSON
        export of Dynesty saves them, although most of them are not in the __dict__ of the fragment.
    """

    # Make a fragment with values which differ from the defaults, and an added attribute
    frag, const = initFragment(MetSimErosion)
    frag.complex = True
    frag.n_grains = 4
    frag.extra_attribute = 'kept'

    frag_ref, _ = initFragment(MetSimErosionReference)

    # The dictionary has the attributes in the order of the Python class, then the added ones
    frag_dict = frag.toDict()
    names = list(vars(frag_ref))
    assert list(frag_dict)[:len(names)] == names
    assert frag_dict['const'] is frag.const
    assert frag_dict['complex'] is True
    assert frag_dict['n_grains'] == 4
    assert frag_dict['extra_attribute'] == 'kept'

    # Save the fragment as JSON as Dynesty does, and compare the saved values
    saved = json.loads(json.dumps(frag, default=jsonDefault))
    for name in ['m', 'v', 'h', 'K', 'n_grains', 'complex', 'extra_attribute']:
        assert saved[name] == frag_dict[name], name
    assert saved['const']['m_init'] == const.m_init
