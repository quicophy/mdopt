Testing
=======

We use ``pytest`` for tests. From the project root:

.. code-block:: bash

   pytest
   # or, if installed with poetry:
   poetry run pytest

Tips
----

- Run a subset:

  .. code-block:: bash

     pytest tests/contractor -q

- Stop on first failure:

  .. code-block:: bash

     pytest -x

- Show print/log output:

  .. code-block:: bash

     pytest -s
