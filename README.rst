Quantum Memories
=====
This is a repository for solvers for the Maxwell-Bloch equations of
various quantum memories, and calculations using them.

For instance, the memory lifetime of the `ORCA
<https://arxiv.org/abs/1704.00013/>`_ quantum memory can be accurately calculated:

.. image:: https://raw.githubusercontent.com/oscarlazoarjona/quantum_memories/master/examples/hyperfine_orca/doppler_dephasing/doppler_dephasing.png

as well as the memory efficiencies for different control powers

.. image:: https://raw.githubusercontent.com/oscarlazoarjona/quantum_memories/master/examples/orca/control_energies/control_energies.png

Installing
--------
This software requires pip and git to be installed. These can be installed as
"Anaconda Python 3" and "GIT Version Control System" from the department's
self-service application for Windows, or downloaded and manually installed from
https://www.continuum.io/downloads and https://git-scm.com/downloads.

Once Anaconda and git are installed, the Python package quantum_memories can be
installed with

::

    $ pip install git+git://github.com/oscarlazoarjona/quantum_memories

To upgrade to the latest functionality use

::

    $ pip install git+git://github.com/oscarlazoarjona/quantum_memories --upgrade

To uninstall use

::

    $ pip uninstall quantum_memories

Using Quantum Memories
----------------------

The source code can be downloaded from

https://github.com/oscarlazoarjona/quantum_memories

and the Python scripts in the examples folder should run once the package
is installed.

Enjoy!
