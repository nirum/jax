"""Save/load pytrees to disk."""

import collections
import h5py
import jax
import numpy as np


def save(filepath, tree):
  """Saves a pytree to an hdf5 file.

  Args:
    filepath: str, Path of the hdf5 file to create.
    tree: pytree, Recursive collection of tuples, lists, dicts,
      namedtuples and numpy arrays to store.
  """
  with h5py.File(filepath, 'w') as f:
    _savetree(jax.device_get(tree), f, 'pytree')


def load(filepath):
  """Loads a pytree from an hdf5 file.

  Args:
    filepath: str, Path of the hdf5 file to load.
  """
  with h5py.File(filepath, 'r') as f:
    return _loadtree(f['pytree'])


def _is_namedtuple(x):
  """Duck typing check if x is a namedtuple."""
  return isinstance(x, tuple) and getattr(x, '_fields', None) is not None


def _savetree(tree, group, name):
  """Recursively save a pytree to an h5 file group."""

  if isinstance(tree, np.ndarray):
    group.create_dataset(name, data=tree)

  else:
    subgroup = group.create_group(name)
    subgroup.attrs['type'] = type(tree).__name__

    if _is_namedtuple(tree):
      for k, subtree in tree._asdict().items():
        _savetree(subtree, subgroup, k)

    elif isinstance(tree, tuple) or isinstance(tree, list):
      for k, subtree in enumerate(tree):
        _savetree(subtree, subgroup, f'arr{k}')

    elif isinstance(tree, dict):
      for k, subtree in tree.items():
        _savetree(subtree, subgroup, k)

    else:
      raise ValueError(f'Unrecognized type {type(tree)}')


def _loadtree(leaf):
  """Recursively load a pytree from an h5 file group."""

  if isinstance(leaf, h5py.Dataset):
    return np.array(leaf)

  else:
    leaf_type = leaf.attrs['type']
    values = map(_loadtree, leaf.values())

    if leaf_type == 'dict':
      return dict(zip(leaf.keys(), values))

    elif leaf_type == 'list':
      return list(values)

    elif leaf_type == 'tuple':
      return tuple(values)

    else:  # namedtuple
      return collections.namedtuple(leaf_type, leaf.keys())(*values)
