# OMM

This project is licensed under the MIT License. See the LICENSE file for more details.

You can find 

### Currently the code is not in an installable form, so some care needs to be taken to run scripts from the right directory in order to avoid import errors. Right now, all the core code is contained in the folders `OMM/`. 

### Suppose we want to run a script `test_script.py` that need some functions or class objects contained in `OMM/`. We need to run `test_script.py` from the same directory in which `OMM/` is located. We can then import and run functions located in `OMM/qOMM.py` like this:

```
from OMM.qOMM import func
```

### This code is the new version of `OMM/` from `Lu-Group-Quantum_Computing-Projects.git` which is now available to work in qiskit==1.1.0. You can find the old version of `OMM/` in:

https://github.com/JoelHBierman/Lu-Group-Quantum_Computing-Projects.git

### The original qOMM paper link: https://arxiv.org/abs/2201.07963

# Installation Instruction

```
pip install ./
cd test
python test_script.py
```

# Alternative Installation

```
pip install git+https://github.com/renlililoli/OMM.git
```

