import datetime
import inspect
import os
import types
import unittest
from collections import Iterable
from enum import Enum
from time import perf_counter

import jsonpickle
import numpy as np
from jsonpickle.ext.numpy import register_handlers
from testfixtures import tempdir

import pyserialization
from pyserialization import Serializable


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Mocks:
    class Stub(Serializable):
        """ Example for Serializable container """

        def __init__(self):
            self.some_string = 'String Constant'
            self.some_int = 666
            self.some_float = 3.14
            self.some_dict = dict(my_str='my string', my_float=3.14)
            self.some_enum = Color.GREEN

    class Simple(Serializable):
        """ Example for Serializable container """

        def __init__(self):
            self.some_string = 'String Constant'
            self.some_int = 666
            self.some_float = 3.14
            self.some_simple_object = Mocks.Stub()

    class StubImplicit:
        """ Example for Serializable container """

        def __init__(self):
            self.some_string = 'String Constant'
            self.some_int = 666
            self.some_float = 3.14
            self.some_dict = dict(my_str='my string', my_float=3.14)
            self.numpy1 = np.array([[0, 0, 0], [100, 0, 0]], dtype='float64')
            self.numpy2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32) / 17

    class SimpleImplicit:
        """ Example for implicit Serializable container (only built-ins, no inheritance) """

        def __init__(self):
            self.some_string = 'String Constant'
            self.some_int = 666
            self.some_float = 3.14
            self.some_simple_object = Mocks.StubImplicit()
            self.some_set = {1, 2, 1}
            self.numpy1 = np.array([[0, 0, 0], [100, 0, 0]], dtype='float64')
            self.numpy2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32) / 17

    class SimpleNumPy(Serializable):
        """ Example of general numpy container object """

        def __init__(self):
            self.numpy1 = np.array([[0, 0, 0], [100, 0, 0]], dtype='float64')
            self.numpy2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32) / 17

    class BigNumPy(Serializable):
        """ Example of general numpy container object """

        def __init__(self):
            x, y = np.meshgrid(range(1000), range(1000))
            self.big_numpy1 = x + y  # type int32
            self.big_numpy2 = x.astype('float64')

    class DateAndTime(Serializable):
        """ Example of datetime objects"""

        def __init__(self):
            self._date_obj = datetime.date(2021, 11, 29)
            self._time_obj = datetime.time(21, 45, 32, 38721)
            self._datetime_obj = datetime.datetime(2001, 7, 16, 19, 23, 11, 986523)


class TestBenchmarkHelper(unittest.TestCase):
    # Prepare dictionary for results in format: 'test function' -> runtime_ms
    _results = {}

    @classmethod
    def setUpClass(cls):
        cls._results = {}

    def benchmark_record_result(self, func, times=1, max_run_time_delta=None):
        """
            Execute a function some times and record timing results
            :param func: Function or Short Lambda
            :param times: How many times to execute
            :param max_run_time_delta: Maximum run-time to assert
            :return: Measured time in ms
        """
        cls = type(self)
        time_start = perf_counter()

        # Execute some times to average
        for _ in range(times):
            func()

        runtime_ms = time_start - perf_counter() * 1000

        time_avg_ms = runtime_ms / times
        cls._results[func] = time_avg_ms

        # Assert runtime if given
        if max_run_time_delta is not None:
            self.assertLessEqual(time_avg_ms, max_run_time_delta.total_seconds() * 1000)

        return time_avg_ms

    @staticmethod
    def _func_code(func):
        """ Return string representation for runtime report """
        from types import LambdaType

        # Return first line of lambda
        if isinstance(func, LambdaType):
            function_code = inspect.getsource(func).strip()
            return function_code.splitlines()[0]

        # Return function title for others
        return func.__name__

    @classmethod
    def tearDownClass(cls):
        """ Print all measurements """
        if len(cls._results) == 0:
            return

        print("Measurement results: ")
        for func, result in cls._results.items():
            print(" - Function: '{}'".format(cls._func_code(func)))

            print(
                '\tAverage Execution Time: {:d} [ms]'.format(int(result))
            )

            print()


class NestedComparator(object):
    def __init__(self, verbose=False):
        self._scanned_values = {}
        self._verbose = verbose

    def compare(self, v1, v2):
        """Compares two complex data structures.

        This handles the case where numpy arrays are leaf nodes.
        Loops are also handled
        """
        return self._compare(v1, v2)

    def _compare(self, v1, v2):
        if self._verbose:
            print("Comparing now {} with {}".format(v1, v2))

        if v1 is None and v2 is None:
            return True

        # both v1 and v2 are not None at the same time
        if v1 is None or v2 is None:
            # only one of them is None
            print("Only one of the values is None: left value {}, right value {}".format(v1, v2))
            return False

        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            if np.allclose(v1, v2):
                return True
            else:  # pragma: no cover
                print("Comparison failed: '{}' is not equal to '{}'".format(v1, v2))
                return False

        # Test if it is a primitive type
        primitive_types = (int, float, bool, str, Enum, datetime.datetime, datetime.date, datetime.time)
        if isinstance(v1, primitive_types) and isinstance(v2, primitive_types):
            if v1 == v2:
                return True
            else:  # pragma: no cover
                if isinstance(v1, float) and isinstance(v2, float) \
                        and np.isclose(v1, v2):
                    return True
                else:
                    print("Comparison failed: '{}' is not equal to '{}'".format(v1, v2))
                    return False

        # Avoid loops
        if id(v1) in self._scanned_values:
            return True
        self._scanned_values[id(v1)] = v1

        # try:
        #    print("{} --- {}".format(v1[0], v2[0]))
        # except:
        #    pass

        if isinstance(v1, dict) and isinstance(v2, dict):
            return self._compare(sorted(v1.items()), sorted(v2.items()))

        if isinstance(v1, Iterable) and isinstance(v2, Iterable):
            # NOTE: len func might not work for all Iterables (not lists)
            return len(v1) == len(v2) and all(self._compare(sub1, sub2) for sub1, sub2 in zip(v1, v2))

        return self._compare(v1.__dict__, v2.__dict__)


class TestSerializer(TestBenchmarkHelper):
    def setUp(self):
        # Consts
        self.consts = types.SimpleNamespace()
        self.consts.num_layer_objects = 100
        self.consts.num_pts_array = 10

        self.consts.times_to_run = 2

    @staticmethod
    def str_dump(path, some_str):
        with open(path, "w") as text_file:
            text_file.write(some_str)

    @staticmethod
    def str_load(path):
        with open(path, "r") as text_file:
            return text_file.read()

    def encode_io_decode_cycle(self, encoder, decoder, path_temp):
        # Encode
        str_enc_1 = encoder(Mocks.SimpleNumPy())

        # Dump to Disk
        self.str_dump(path_temp, str_enc_1)

        # Load from Disk
        str_enc_1 = self.str_load(path_temp)

        # Decode
        _ = decoder(str_enc_1)

    def verify_encode_decode_cycle(self, obj, **kw):
        str_encoded1 = pyserialization.encode(obj, **kw)
        obj_decoded = pyserialization.decode(str_encoded1)

        # Assume no transient fields...
        self.assertTrue(NestedComparator().compare(obj, obj_decoded))

    @tempdir()
    def test_measure_encode(self, tmp_dir):
        """
            Measure the average run-time of object encoding to string using our method vs jsonpickle
        """

        # Support jsonpickle NumPy extension
        register_handlers()

        # Define jsonpickle encoder & decoder
        def encode_jsonpickle(obj):
            return jsonpickle.encode(obj)

        def decode_jsonpickle(str_encoded):
            return jsonpickle.decode(str_encoded)

        # Path for temp saving & loading
        path_temp_file = os.path.join(tmp_dir.path, 'str_encoded.json')

        # Benchmark and save results in dictionary
        self.benchmark_record_result(
            lambda: self.encode_io_decode_cycle(pyserialization.encode, pyserialization.decode, path_temp_file),
            times=self.consts.times_to_run
        )

        self.benchmark_record_result(
            lambda: self.encode_io_decode_cycle(encode_jsonpickle, decode_jsonpickle, path_temp_file),
            times=self.consts.times_to_run
        )

    def test_serializable_simple(self):
        """
            Verify whole cycle of encode -> save -> load -> decode of simple objects - built-ins and structs
        """
        self.verify_encode_decode_cycle(
            {1, 2, 2, 3}
        )

        self.verify_encode_decode_cycle(Mocks.Simple())

        self.verify_encode_decode_cycle(
            [4, 5, dict(a='A', b='B'), Mocks.SimpleNumPy()]
        )

    def test_serializable_big_numpy(self):
        """
            Verify whole cycle in the case of big numpy arrays
        """

        self.verify_encode_decode_cycle(Mocks.BigNumPy())

    def test_serializable_numpy(self):
        """
            Verify whole cycle of encode -> save -> load -> decode of various NumPy objects
        """

        self.verify_encode_decode_cycle(Mocks.SimpleNumPy())

        # Nested Comparator doesn't seem to work on numpy primitives, test manually
        # Note that simple boolean operator is overridden by NumPy and returns a NumPy primitive 'bool_'
        numpy_primitive = np.max(np.array([1, 2, 3])) < 50 or np.max(np.array([1, 2, 3])) > 300
        str_encoded = pyserialization.encode(numpy_primitive)
        obj_decoded = pyserialization.decode(str_encoded)

        self.assertEqual(obj_decoded, numpy_primitive)

    def test_serialization_simple_implicit(self):
        self.verify_encode_decode_cycle(Mocks.SimpleImplicit(), allow_implicit_simples=True)

    def test_serializable_datetime(self):
        self.verify_encode_decode_cycle(Mocks.DateAndTime())


if __name__ == '__main__':
    unittest.main()
