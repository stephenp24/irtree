import os

import pytest


def test_read():
    from collections import OrderedDict

    query = "/a0/b0/c1/d0/e2"
    queries = OrderedDict()
    [queries.setdefault(k, v) for k, v in enumerate(filter(None, query.split("/")))]

    print(queries)

    assert True
