# Notices

This code package is prepared for reproducing the SEA-PINN examples from the
manuscript.

Parts of the PDE definitions, the Heat2D_Multiscale analytical solution, the
Poisson2D_ManyArea and NS2D_BackStep numerical reference data, and PINN utility
code are adapted from the PINNacle benchmark:

- Repository: https://github.com/i207M/PINNacle
- License: MIT
- License text included at `LICENSES/PINNacle-LICENSE.txt`

This release includes a bundled DeepXDE-compatible runtime in `deepxde/`.
DeepXDE upstream is available at:

- Repository: https://github.com/lululxvi/deepxde
- License: LGPL-2.1
- License text included at `LICENSES/DeepXDE-LICENSE.txt`

The optional TSA-PINN baseline is implemented in PyTorch/DeepXDE for this
release. Its architecture was written with reference to the TSA-PINN project:

- Repository: https://github.com/AmirhosseinnnKhademi/TSA-PINN
- License: MIT
- License text included at `LICENSES/TSA-PINN-LICENSE.txt`

The TSA-PINN implementation in this package is not a direct copy of that
TensorFlow codebase; it is an adapted implementation for the present benchmark
and training pipeline.
