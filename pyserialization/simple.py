"""
    Known Issues:
        - Dictionary keys are converted to strings, and will not be loaded correctly if they weren't strings.
        - Tuples are converted to lists, if a tuple is important - use a custom wrapper class for the tuple.
"""

import importlib
import inspect
import json
import datetime
from enum import Enum
from functools import reduce
from typing import Dict

import numpy as np
import base64


def qualname(obj):
    """
        __qualname__ was added in python 3.5, and we rely upon it. It is possible to back port this library by providing
         an alternative implementation of this method.
    """
    return obj.__qualname__


_MAGIC_PY_CLASS = '__py_class__'


def encode(obj, allow_implicit_simples=False):
    """
    :param allow_implicit_simples: Allow serialization of objects that don't implement Serializable by using their
        own to_dict / from dict methods if exist, or provide default implementation otherwise
    :param obj: Any of: built-in, NumPy.ndarray, Serializable, to_dict/from_dict, (including decedents)

    :return: String representation of the object
    """
    return json.dumps(
        obj, cls=_JsonEncoderSimple, allow_implicit_simples=allow_implicit_simples, sort_keys=True, indent=4
    )


def decode(str_encoded):
    """
        :param str_encoded: String encoded with 'encode(obj)' function
        :return: Python object
    """
    return json.loads(str_encoded, object_hook=from_dict)


class _PythonClassName:
    """
        Helper class to manipulate fully qualified type names (including module and nested classes)
    """
    _separator = '/'

    def __init__(self, module_name=None, class_name=None):
        self.module_name = module_name
        self.class_name = class_name

    def to_string(self):
        return '{}/{}'.format(self.module_name, self.class_name)

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_string(cls, str_name):
        module_name, class_name = str_name.split(_PythonClassName._separator)
        return cls(module_name, class_name)

    @classmethod
    def from_type(cls, typ: object):
        # noinspection PyTypeChecker
        return cls(typ.__module__, qualname(typ))

    @property
    def type(self):
        """
            Instantiates an object using name of module and class.
            Module is first imported to current execution context.
        """

        module_name = importlib.import_module(self.module_name)

        # Iterate over all parent classes (support nested)
        cls = reduce(getattr, self.class_name.split("."), module_name)
        try:
            return cls
        except TypeError as e:
            raise Exception(
                "Error calling {}.__init__(), Full Exception:{}".format(_PythonClassName.from_type(cls), repr(e))
            )

    def get_instance(self, *args, **kwargs):
        cls = self.type
        return cls(*args, **kwargs)


def to_dict(obj: object, allow_implicit_simples=False) -> Dict:
    """
        Recursively convert object to dictionary primitives, with additional type hint to allow reconstruction
            of an identical object from the dictionary.

        :param obj: Object to be traversed recursively
        :param allow_implicit_simples: Try to serialize not explicitly supported types

        Note: Returned dictionary contains the entire tree structure of the members of the class
    """

    result = dict()
    _add_type_key(result, obj)

    # Convert Serializable objects
    if isinstance(obj, Serializable):
        result.update(obj.to_dict())

    # Non-Serializable - Primitive / Special / Unsupported
    else:
        handler = Handlers.find_for(type(obj))

        # Specially Supported
        if handler != Handlers.Default:
            result.update(handler.to_dict(obj))

        else:
            # Assume primitives have no __dict__
            if not hasattr(obj, '__dict__'):
                result = obj

            # Unsupported
            else:
                if allow_implicit_simples:
                    result.update(handler.to_dict(obj))
                else:
                    raise Exception('Unable to implicitly convert non Serializable objects to dictionary')

    # Update all sub tree of each member
    if type(result) == dict:
        for key, value in result.items():
            result[key] = to_dict(value, allow_implicit_simples)

    return result


def from_dict(dic: Dict) -> object:
    """
        Recursively decode using the following convention, dictionaries without magic string are considered built-ins
        {
            '__py_class__' : '<module name>/<class_name>'
            ... rest of data ...
        }
        :param dic: Dictionary Object
        :return Instance of an object
    """
    # Decode inner objects
    for key, value in dic.items():
        if type(value) == dict:
            dic[key] = from_dict(value)

    # Resolve only dictionaries with magic
    if _MAGIC_PY_CLASS not in dic:
        # Pass through the dictionary as built-in
        return dic
    else:
        # Shortcuts
        cls_str = dic[_MAGIC_PY_CLASS]

        # Determine object type
        cls = _PythonClassName.from_string(cls_str).type

        # Use correct handler for decoding given dictionary to object
        return Handlers.find_for(cls).from_dict(dic)


class Serializable(object):
    def to_dict(self):
        """
            Default implementation for Simple objects (properties are either built-in or simple)
            Generally, the state of simple objects is stored in the __dict__ attribute
            Note: The values of the dict are allowed to be instances of Serializable, NOT ONLY primitives
        """
        dic = dict()

        # Append all variables
        for key, value in vars(self).items():
            # Pass primitives
            if not hasattr(value, '__dict__'):
                dic[key] = value

            # Encode Serializable objects
            elif isinstance(value, Serializable):
                dic[key] = value.to_dict()

                # Store class name to allow later de-serialization using handler
                dic[key][_MAGIC_PY_CLASS] = str(_PythonClassName.from_type(type(value)))

            # Pass objects assuming that they can be serialized implicitly
            else:
                dic[key] = value

        return dic

    @classmethod
    def from_dict(cls, dic):
        """ An new instance of type(self) with updated dictionary (default implementation) """
        instance = cls()
        instance.__dict__.update(dic)
        return instance


class SerializableWrapper(Serializable):
    def __init__(self):
        self._payload = None
        self._decode_type_str = None

    @staticmethod
    def wrap_obj(obj):
        """ Wraps a simple object to allow decoding to the original type """

        ret = SerializableWrapper()

        # Store object dictionary & type
        if hasattr(obj, '__dict__'):
            ret._payload = obj.__dict__
        else:
            ret._payload = obj

        ret._decode_type_str = str(_PythonClassName.from_type(obj.__class__))

        return ret

    @classmethod
    def from_dict(cls, dic):
        # from_dict object as serializable
        # Note: Object is not decoded to type SerializableWrapper but it has all it's properties
        sw = Serializable.from_dict(dic)  # type: SerializableWrapper

        # noinspection PyProtectedMember
        cls_decode = _PythonClassName.from_string(sw._decode_type_str).type
        ret = cls_decode()

        # Decode to desired class or to built-in type
        if hasattr(ret, '__dict__'):
            # noinspection PyProtectedMember
            ret.__dict__.update(sw._payload)
        else:
            # noinspection PyProtectedMember
            return sw._payload

        return ret


class Handlers:
    # Class variable to store all available handlers
    _set_handlers = set()

    # All auto-discoverable handlers must inherit CustomHandler
    # The `type` class variable is used to find supported instances for handler
    class CustomHandler:
        pass

    class Default:
        @staticmethod
        def to_dict(obj):
            dic = SerializableWrapper.wrap_obj(obj).to_dict()

            # Make sure de-serialization happens using SerializableWrapper
            dic[_MAGIC_PY_CLASS] = str(_PythonClassName.from_type(SerializableWrapper))
            return dic

        @staticmethod
        def from_dict(data):
            return Serializable.from_dict(data)

    # noinspection PyClassHasNoInit
    class SerializableHandler(CustomHandler):
        type = Serializable

        @staticmethod
        def to_dict(obj):
            return obj.to_dict()

        @staticmethod
        def from_dict(data):
            # Use magic string to decode using the right class
            cls = _PythonClassName.from_string(data[_MAGIC_PY_CLASS]).type

            # Remove magic string from dictionary
            data.pop(_MAGIC_PY_CLASS)

            return cls.from_dict(data)

    # noinspection PyClassHasNoInit
    class NumPyNdArray(CustomHandler):
        type = np.ndarray

        @staticmethod
        def to_dict(obj):
            if obj.size < 100 or obj.__class__ == np.core.records.recarray:
                return SerializableWrapper.wrap_obj(
                    {
                        'array': obj.tolist(),
                        'dtype': str(obj.dtype)
                    }
                ).to_dict()
            else:
                # use base64 for big arrays
                if not obj.flags.c_contiguous:
                    # b64encode needs contiguous array
                    # (this shouldn't happen for video frames, they should be already contiguous)
                    obj = np.ascontiguousarray(obj)
                return SerializableWrapper.wrap_obj(
                    {
                        'array_base64': base64.b64encode(obj).decode(),
                        'dtype': str(obj.dtype),
                        'shape': list(obj.shape)
                    }
                ).to_dict()

        @staticmethod
        def from_dict(data):
            wrapper = SerializableWrapper.from_dict(data)
            dtype = None

            is_recarray = data['__py_class__'] in ['numpy.core.records/recarray', 'numpy/recarray']
            if is_recarray:
                dtype = np.dtype(eval(wrapper['dtype']))
                array = np.array(wrapper['array'])
                record_size = len(dtype)
                return np.core.records.fromarrays([array[..., x] for x in range(record_size)], dtype=dtype)

            if isinstance(wrapper, dict):
                if wrapper:
                    dtype = np.dtype(wrapper['dtype'])
                if 'array_base64' not in wrapper:
                    array = wrapper['array']
                else:
                    shape = tuple(wrapper['shape'])
                    array_b64 = wrapper['array_base64']
                    buf = base64.decodebytes(array_b64.encode())
                    return np.frombuffer(buf, dtype).reshape(shape)
            else:
                array = wrapper

            return np.array(array, dtype)

    # noinspection PyClassHasNoInit
    class NumPyGeneric(CustomHandler):
        type = np.generic

        @staticmethod
        def to_dict(obj):
            return SerializableWrapper.wrap_obj(obj.item()).to_dict()

        @staticmethod
        def from_dict(data):
            return np.array([SerializableWrapper.from_dict(data)])[0]

    # noinspection PyClassHasNoInit
    class SetSerializable(CustomHandler):
        type = set

        @staticmethod
        def to_dict(obj):
            return SerializableWrapper.wrap_obj(list(obj)).to_dict()

        @staticmethod
        def from_dict(data):
            return set(SerializableWrapper.from_dict(data))

    # noinspection PyClassHasNoInit
    class EnumSerializable(CustomHandler):
        type = Enum

        @staticmethod
        def to_dict(obj):
            if hasattr(obj, 'value'):
                return {
                    '_value': obj.value
                }
            else:
                return {
                    '_type': str(_PythonClassName.from_type(obj))
                }

        @staticmethod
        def from_dict(data):
            if '_value' in data:
                return _PythonClassName.from_string(data[_MAGIC_PY_CLASS]).get_instance(data['_value'])
            elif '_type' in data:
                return _PythonClassName.from_string(data['_type']).type

    # noinspection PyClassHasNoInit
    class DateSerializable(CustomHandler):
        type = datetime.date
        string_format = "%Y-%m-%d"

        @classmethod
        def to_dict(cls, obj: datetime.date):
            return {'value': obj.strftime(cls.string_format)}

        @classmethod
        def from_dict(cls, data):
            return datetime.datetime.strptime(data["value"], cls.string_format).date()

    # noinspection PyClassHasNoInit
    class TimeSerializable(CustomHandler):
        type = datetime.time
        string_format = "%H:%M:%S.%f"

        @classmethod
        def to_dict(cls, obj: datetime.time):
            return {'value': obj.strftime(cls.string_format)}

        @classmethod
        def from_dict(cls, data):
            return datetime.datetime.strptime(data["value"], cls.string_format).time()

    # noinspection PyClassHasNoInit
    class DateTimeSerializable(CustomHandler):
        type = datetime.datetime
        string_format = "%Y-%m-%d %H:%M:%S:%f"

        @classmethod
        def to_dict(cls, obj: datetime.datetime):
            return {'value': obj.strftime(cls.string_format)}

        @classmethod
        def from_dict(cls, data):
            return datetime.datetime.strptime(data["value"], cls.string_format)

    @classmethod
    def find_for(cls, obj_type):
        # Try to find exact type
        for h in cls._set_handlers:
            if obj_type == h.type:
                return h

        # Allow subclass handlers if exact match not available
        for h in cls._set_handlers:
            if issubclass(obj_type, h.type):
                return h

        return cls.Default

    # Static Constructor
    @classmethod
    def init_mapping(cls):
        # Find all handlers
        handlers = []
        for atr in Handlers.__dict__.values():
            if inspect.isclass(atr) and issubclass(atr, Handlers.CustomHandler) and atr != Handlers.CustomHandler:
                handlers.append(atr)

        cls._set_handlers = set(handlers)


# Call static constructor
Handlers.init_mapping()


def is_object_serializable(obj):
    handler = Handlers.find_for(type(obj))

    return handler != Handlers.Default


def _add_type_key(dic, obj):
    dic[_MAGIC_PY_CLASS] = str(_PythonClassName.from_type(type(obj)))


class _JsonEncoderSimple(json.JSONEncoder):
    """
        Encode compound objects - any of: built-in / Serializable / JsonPickle
        Uses obj.to_dict() if possible, to transform to dict for json encoding
    """

    def __init__(self, allow_implicit_simples=False, **kw):
        # noinspection PyArgumentList
        super(_JsonEncoderSimple, self).__init__(**kw)
        self.allow_implicit_simples = allow_implicit_simples

    # Allow method-hidden, because this is the convention of json package
    # pylint: disable=method-hidden
    def default(self, obj):
        handler = Handlers.find_for(type(obj))

        dic = {}
        _add_type_key(dic, obj)

        if handler != Handlers.Default:
            dic.update(handler.to_dict(obj))
        else:
            if self.allow_implicit_simples:
                dic.update(handler.to_dict(obj))
            else:
                raise Exception('Unable to serialize (implicitly) object type: {}'.format(type(obj)))

        return dic
