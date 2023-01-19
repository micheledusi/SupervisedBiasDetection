# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module offers some utility functions for caching data.

import copy
import json
import os
import time
from typing import Any
import pickle as pkl


def _hash_dict(obj: Any) -> int:
  """
  Makes a hash from a dictionary, list, tuple or set to any level, that contains
  only other hashable types (including any lists, tuples, sets, and
  dictionaries).
  """
  if isinstance(obj, (set, tuple, list)):
    return tuple([_hash_dict(elem) for elem in obj])    

  elif not isinstance(obj, dict):
    return hash(obj)

  new_o = copy.deepcopy(obj)
  for k, v in new_o.items():
    new_o[k] = _hash_dict(v)

  return hash(tuple(frozenset(sorted(new_o.items()))))


class CacheManager:
    """
    This class offers some utility functions for caching data.
    Each data can be saved along with a unique identifier and some metadata.
    Then, the data can be retrieved by providing the unique identifier and the metadata.
    """

    DEFAULT_ROOT_DIR: str = 'cache'
    DEFAULT_GROUP: str = 'generic'

    REGISTER_FILENAME: str = 'register.json'

    def __init__(self, root_dir: str = DEFAULT_ROOT_DIR) -> None:
        self.root: str = root_dir


    def _init_group(self, group: str) -> None:
        """
        This method initializes a new group of data, i.e. the directory where the data will be saved.
        If the group already exists, this method does nothing.
        If the directory where the data will be saved does not exist, this method creates it.
        Plus, this method creates a file containing the register of the group, called 'register.json'.

        :param group: The name of the group.
        :return: None.
        """
        # If the group directory does not exist, create it.
        if not os.path.exists(self.root + '/' + group):
            os.makedirs(self.root + '/' + group)
        # If the register file does not exist, create it.
        if not os.path.exists(self.root + '/' + group + '/' + self.REGISTER_FILENAME):
            with open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'w') as f:
                f.write('[]')


    def _compute_filename(self, identifier: str, group: str, metadata: dict[str, Any]) -> str:
        """
        This method computes the filename of the object with the given identifier, group and metadata.
        The filename is generated by hashing the metadata, the identifier and the group.
        """
        descriptor = metadata.copy()
        descriptor['identifier'] = identifier
        descriptor['group'] = group
        return str(_hash_dict(descriptor))


    def _register(self, identifier: str, filename: str, group: str, metadata: dict[str, Any]) -> None:
        """
        This method register a new object in the correct group.
        The registration writes in the group register file (a JSON file) the identifier and the metadata of the object.
        Default metadata are added to the given metadata, such as:
        - the save timestamp
        - the size of the object
        
        This method does not save the object itself.
        This method does not check if the object already exists.
        This method does not check if the group exists: it is the caller's responsibility to check it and to create the group if necessary.

        :param identifier: The unique identifier of the object.
        :param filename: The filename of the object, generated by the hash of the metadata (see method ``save``)
        :param group: The name of the group
        :param metadata: The metadata of the object, as a dictionary
        :return: None.
        """
        register = json.load(open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'r'))
        assert type(register) == list
        # If a corresponding object already exists, we remove it
        for i in range(len(register)):
            if register[i]['identifier'] == identifier and register[i]['metadata'] == metadata:
                register.pop(i)
                break

        # Adding the new object to the register
        register.append({
            'identifier': identifier,
            'filename': filename,
            'timestamp': time.time(),
            'size': os.path.getsize(self.root + '/' + group + '/' + filename + '.pkl'),
            'metadata': metadata
        })
        # Saving the register
        json.dump(register, open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'w'))


    def save(self, data: Any, identifier: str, group: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """
        This method saves the given data along with the given identifier and metadata.
        The data is saved in a pickle file, and the metadata is saved in the group register file.

        :param data: The data to save.
        :param identifier: The unique identifier of the data.
        :param group: The name of the group (optional; default: None)
        :param metadata: The metadata of the object to save (optional; default: None)
        :return: None.
        """
        # Asserting that the group exists
        if group is None or group == '':
            group = self.DEFAULT_GROUP
        self._init_group(group)

        # Asserting the metadata are a dictionary
        if metadata is None:
            metadata = dict()

        # Computing the filename
        filename = self._compute_filename(identifier, group, metadata)

        # Saving the data
        pkl.dump(data, open(self.root + '/' + group + '/' + filename + '.pkl', 'wb'))

        # Registering the object to the group register
        self._register(identifier, filename, group, metadata)


    def exists(self, identifier: str, group: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
        """
        This method checks if the object with the given identifier, group and metadata exists.
        If the object exists, this method returns True.
        If the object does not exist, this method returns False.

        :param identifier: The unique identifier of the object.
        :param group: The name of the group (optional; default: None)
        :param metadata: The metadata of the object, as a dictionary (optional; default: None)
        :return: True if the object exists, False otherwise.
        """
        if group is None or group == '':
            group = self.DEFAULT_GROUP
        if metadata is None:
            metadata = dict()

        # If the group does not exist, the object does not exist
        if not os.path.exists(self.root + '/' + group):
            return False
        # If the register file does not exist, the object does not exist
        if not os.path.exists(self.root + '/' + group + '/' + self.REGISTER_FILENAME):
            return False
        # If the object is not in the register, the object does not exist
        register = json.load(open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'r'))
        for obj in register:
            if obj['identifier'] == identifier and obj['metadata'] == metadata:
                return True


    def _retrieve_filename(self, identifier: str, group: str, metadata: dict[str, Any]) -> str | None:
        """
        This method returns the filename of the object with the given identifier and metadata.
        If the object does not exist, this method returns None.

        :param identifier: The unique identifier of the object.
        :param group: The name of the group (optional; default: None)
        :param metadata: The metadata of the object, as a dictionary (optional; default: None)
        :return: The filename of the object, or None if the object does not exist.
        """
        # Checking if the object exists
        if not self.exists(identifier, group, metadata):
            return None

        # Searching
        register = json.load(open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'r'))
        for obj in register:
            if obj['identifier'] == identifier and obj['metadata'] == metadata:
                return obj['filename']

        return None
    

    def load(self, identifier: str, group: str | None = None, metadata: dict[str, Any] | None = None) -> Any:
        """
        This method loads the object with the given identifier and metadata.
        If the object does not exist, this method raises an exception.

        :param identifier: The unique identifier of the object.
        :param group: The name of the group (optional; default: None)
        :param metadata: The metadata of the object, as a dictionary (optional; default: None)
        :return: The object.
        """
        # Checking if the object exists
        if group is None or group == '':
            group = self.DEFAULT_GROUP
        if metadata is None:
            metadata = dict()
        if not self.exists(identifier, group, metadata):
            raise Exception('Object does not exist.')

        # Retrieving the filename
        filename = self._retrieve_filename(identifier, group, metadata)

        # Loading the object
        return pkl.load(open(self.root + '/' + group + '/' + filename + '.pkl', 'rb'))


