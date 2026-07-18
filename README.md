# Altendorf-Jeulin-Model
This python module generates fiber-based microstructures. 
The implementation is based on the original paper by 
Altendorf & Jeulin [1] and the extension to endless fibers
by Easwaran [2]. For the fiber direction distribution, we
implemented generators based on Franke et al. [3].

The module is an early release and still under development. 
If you have questions, implementation requests, or suggestions, 
please contact [keilmann@rptu.de](keilmann@rptu.de) or
[neumann@tu-graz.at](neumann@tu-graz.at).
We are open for contributions and collaboration!

Please cite this work as follows:
A. Keilmann, M. Neumann. An open-source package for simulating and calibrating fiber-based materials with the Altendorf-Jeulin model. In preparation.

## Requirements
Please make sure you have at least Python 3.14 installed. Further requirements are
saved in the requirements.txt.

## Setup (recommended)
Clone the repository (or download it as a zip under the Code-Button).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Usage
Run the example script
```bash
python examples/run_example.py
```
This runs:
- a finite-fiber model (directional distribution: Schladitz), 
- an infinite-fiber model (directional distribution: angular central Gaussian).

Outputs are written to ./outputs/ (adjust if different).

## Parallelized Implementation
For further speed-up, you may want to use our parallelized
implementation. Please note that this is only faster than the 
serialized version if your fiber model is large enough.

If you cloned the repository, you can access the parallelized
version by entering
```bash
git checkout concurrency
pip install -e .
python examples/run_example.py
```
in your terminal. If you downloaded the code as a zip, download
the concurrent code as a zip, switch manually to the concurrency-
branch on the GitHub-page before downloading the zip of the code
again.

## References
[1] H. Altendorf, D. Jeulin, 2011. Random-walk-based stochastic modeling of three-dimensional fiber systems. Physical Review E.

[2] P. Easwaran, 2017. Stochastic geometry models for interacting fibers. Doctoral dissertation, Technische Universität Kaiserslautern.

[3] J. Franke, C. Redenbach, & N. Zhang, 2016. On a Mixture Model for Directional Data on the Sphere. Scandinavian Journal of Statistics, 43(1), 139–155. https://doi.org/10.1111/sjos.12169

## Further Fiber Models
- [SAMSON Microstructure Generator](https://git.uni-due.de/publicsoftwareingmath/samson)
- [High-density Wood Fiber Network](https://github.com/BinChenOPEN/3D-High-Density-Wood-Fiber-Network-Model)
- [White Matter Generator](https://github.com/MaP-science/WhiteMatterGenerator)
- [FibreSimulator](https://github.com/marychrisgo/Fiber-Phantoms)

If your fiber model implementation is missing and you would like us to add it to the list, please let us know.

## License

This project is licensed under the GNU General Public License (GPL). See the `LICENSE.txt` file in this repository for 
the full license text and copyright notices.

If you contact us with a dual-licensing request, we may be able to offer additional licensing terms. Please reach out
via the repository’s contact details.