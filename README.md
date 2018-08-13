`Helium Stark Animate`
===============

Create animations of the Rydberg electron charge distribution in helium as a function of electric field using the Numerov method.

Install
------- 

Install `helium-stark-animate` using `setuptools`,
```bash
git clone https://github.com/axm108/helium-stark-animate
cd helium-stark-animate
python setup.py install
```

Install [`helium-stark-zeeman`](https://github.com/axm108/helium-stark-zeeman) using `setuptools`,
```bash
git clone https://github.com/axm108/helium-stark-zeeman
cd helium-stark-zeeman
python setup.py install
```

Basic usage
-------
Import libraries,
```python
from heliumstarkanimate import HeliumStarkAnimator
from hsz import HamiltonianMatrix
import numpy as np
```

Instantiate `HamiltonianMatrix` object,
```python
n_min = 5
n_max = 6
S = 1
ham = HamiltonianMatrix(n_min=n_min, n_max=n_max, S=S)
```

Instantiate `HeliumStarkAnimator` object,
```python
animator = HeliumStarkAnimator(ham)
```

Calculate charge distributions of a state for different values of the electric field,
```python
state_idx = n_min**2
Efield = np.linspace(0.0, 6.0*10**5, 11) # V/cm
charge_dists = animator.charge_distributions(state_idx, Efield*1e2)
```

Plot the charge distribution for one field,
```python
animator.plot(charge_dists[0])
```

Save the charge distributions to a `.jpg` file for each field,
```python
animator.save(charge_dists)
```

Plot an interactive figure of the charge distribution along with the corresponding Stark map,
```python
animator.plot_interactive(Efield, charge_dists, stark_map, state_idx)
```

Version information
-------------------

| Library  | Version |
| ------------ | ------------ |
| `Python`  | 3.6.1 64bit [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] |
| `IPython` | 5.3.0 |
| `OS` | Darwin 17.4.0 x86_64 i386 64bit |
| `attr` | 17.4.0 |
| `matplotlib` | 2.0.2 |
| `numba` | 0.35.0 |
| `numpy` | 1.14.3 |
| `scipy` | 1.00.0 |
| `sympy` | 1.0 |
| `tqdm` | 4.15.0 |
| `version_information` | 1.0.3 |
