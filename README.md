# Altendorf-Jeulin-Model
This Python module generates fiber-based microstructures using the Altendorf–Jeulin random-walk model [1], extended to 
endless fibers by Easwaran [2]. It includes refined packing algorithms (AJ++) and a contact model (Contact++) for 
increasing inter-fiber contact [4, 5]. Fiber direction generators are based on Franke et al. [3].

The module is an early release and still under development. If you have questions, implementation requests, or 
suggestions, please contact [alex.keilmann@uni-due.de](mailto:alex.keilmann@uni-due.de) or 
[neumann@tu-graz.at](mailto:neumann@tu-graz.at). We are open for contributions and collaboration!


Please cite this work, depending on the functions you are using, as follows:
- A. Keilmann, M. Neumann. An open-source package for simulating and calibrating fiber-based materials with the Altendorf-Jeulin model. In preparation.
- A. Keilmann, C. Redenbach, F. Willot, 2026. Increasing Inter-Fiber Contact in the Altendorf-Jeulin Model. Computational Materials Science.


## Requirements
Python 3.14 or later. All dependencies are listed in requirements.txt and pyproject.toml.

## Setup (recommended)
Clone the repository (or download it as a zip under the Code-Button). Then, run the following code in your terminal.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cd src
python setup.py build_ext --inplace
```
## Usage

Run the example script from the repository root:
```bash
python examples/run_example.py
```
This demonstrates three workflows:

- Finite fibers (example_AJ_finite): initializes a fiber system, packs it with AJ++ to remove all overlaps, then applies Contact++ to increase inter-fiber contact. Uses a Schladitz direction distribution.
- Endless fibers (example_AJ_endless): same pipeline for domain-spanning fibers with a non-periodic boundary, using an angular central Gaussian distribution.
- Poisson lines (example_poisson_lines): generates a 3D Poisson line process.

Outputs are written to examples/outputs/ (adjust if different). 


## Packing Algorithms

The module provides four packing methods, selectable via the method parameter of run_force_biased:

- AJ – original Altendorf–Jeulin force-biased packing [1].
- AJ++ – refined version that reformulates packing as an optimization problem with a stricter, window-size-independent stop criterion [5]. Recommended for most use cases.
- AJ++fast – same as AJ++, but stops as soon as all overlaps are resolved.
- Contact++ – extends AJ++ with a contact force that draws fibers towards a precomputed shortlist of contact candidates, increasing inter-fiber contact [4, 5].

Intermediate fiber states are saved as .gad files to examples/outputs/ at regular intervals during packing (configurable via output_step_size), so a run can be resumed from the last checkpoint if it crashes.

## Output Formats

### 3D images (fiber and line systems voxelized as label images):

- .tif / .tiff — written via tifffile, axes stored as XYZ metadata
- .mha / .mhd — written via SimpleITK, with physical voxel spacing

### Fiber geometry:

- .txt — plain text, one fiber per line with ball coordinates; readable without any extra tooling
- .gad — GeoDict format (JSON-based); supports read and write, coordinates in metres
- .fft.* — TexMathVisualizer graph format (.fft.nod, .fft.elm, .fft.threads, and three count files); available as full-resolution and simplified (small graph) variants

### Statistics (written during packing):

- .csv — per-step packing statistics (volume fraction, MAE, KL divergence, overlap, contact distance)


## References
[1] H. Altendorf, D. Jeulin, 2011. Random-walk-based stochastic modeling of three-dimensional fiber systems. Physical Review E. [doi: 10.1103/PhysRevE.83.041804](https://doi.org/10.1103/PhysRevE.83.041804)

[2] P. Easwaran, 2017. Stochastic geometry models for interacting fibers. Doctoral dissertation, Technische Universität Kaiserslautern.

[3] J. Franke, C. Redenbach, & N. Zhang, 2016. On a Mixture Model for Directional Data on the Sphere. Scandinavian Journal of Statistics, 43(1), 139–155. [doi: 10.1111/sjos.12169](https://doi.org/10.1111/sjos.12169)

[4] A. Keilmann, C. Redenbach, F. Willot, 2026. Increasing Inter-Fiber Contact in the Altendorf-Jeulin Model. Computational Materials Science. [doi: 10.1016/j.commatsci.2025.114458](https://doi.org/10.1016/j.commatsci.2025.114458)

[5] In preparation.

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