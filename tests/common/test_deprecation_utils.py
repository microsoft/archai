# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings

from archai.common.deprecation_utils import deprecated


def test_deprecated_decorator():
    def my_func():
        pass

    def my_func2():
        pass

    def my_func3():
        pass

    def my_func4():
        pass

    def my_func5():
        pass

    # Assert that it works without arguments
    deprecated_func = deprecated()(my_func)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_func()
        assert len(w) == 1
        assert str(w[0].message) == "`my_func` has been deprecated and will be removed."

    # Assert that it works with message argument
    deprecated_func_message = deprecated(message="Use another function instead.")(my_func2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_func_message()
        assert len(w) == 1
        assert str(w[0].message) == "`my_func2` has been deprecated and will be removed. Use another function instead."

    # Assert that it works with deprecated_version argument
    deprecated_func_version = deprecated(deprecate_version="1.0.0")(my_func3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_func_version()
        assert len(w) == 1
        assert str(w[0].message) == "`my_func3` has been deprecated in v1.0.0 and will be removed."

    # Assert that it works with remove_version argument
    deprecated_func_remove = deprecated(remove_version="2.0.0")(my_func4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_func_remove()
        assert len(w) == 1
        assert str(w[0].message) == "`my_func4` has been deprecated and will be removed in v2.0.0."

    # Assert that it works with both deprecated_version and remove_version arguments
    deprecated_func_both = deprecated(deprecate_version="1.0.0", remove_version="2.0.0")(my_func5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_func_both()
        assert len(w) == 1
        assert str(w[0].message) == "`my_func5` has been deprecated in v1.0.0 and will be removed in v2.0.0."
