# OMM

### Currently the code is not in an installable form, so some care needs to be taken to run scripts from the right directory in order to avoid import errors. Right now, all the core code is contained in the folders `OMM/`. 

### Suppose we want to run a script `test_script.py` that need some functions or class objects contained in `OMM/`. We need to run `test_script.py` from the same directory in which `OMM/` is located. We can then import and run functions located in `OMM/qOMM.py` like this:

```
from OMM.qOMM import func
```

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