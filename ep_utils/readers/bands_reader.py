class BandStructure:
    def __init__(self, filename, clamp=False, mp_grid=None):
        with open(filename) as bf:
            filedata = iter(bf)
            current_spin = None
            self.kpts = []
            self.kpt_bands = []
            self.kpt_weights = []
            self.clamp = clamp
            self.mp_grid = mp_grid

            for line in filedata:
                if "Number of k-points" in line:
                    self.nkpts = int(line.split()[-1])
                elif "Number of spin components" in line:
                    self.nspins = int(line.split()[-1])
                elif "Number of electrons" in line:
                    self.nelec = [float(ne) for ne in line.split()[-self.nspins:]]
                elif "Number of eigenvalues" in line:
                    self.neigen = [int(ne) for ne in line.split()[-self.nspins:]]
                elif "Fermi energ" in line:
                    self.fermi = [float(ne) for ne in line.split()[-self.nspins:]]
                elif "Unit cell vectors" in line:
                    self.cell_data = [[float(c) for c in next(filedata).split()] for _ in range(3)]
                elif "K-point" in line:
                    # Read kpoint position and weight
                    split_line = line.split()
                    self.kpts.append([float(c) for c in split_line[2:-1]])
                    self.kpt_weights.append(float(split_line[-1]))
                    self.kpt_bands.append([[] for _ in range(self.nspins)])  # Add empty lists for each eigenvalue

                elif "Spin component" in line:
                    current_spin = int(line.split()[-1]) - 1  # zero indexing
                    eigenvalues = [float(next(filedata)) for _ in range(self.neigen[current_spin])]
                    self.kpt_bands[-1][current_spin] = eigenvalues
                else:
                    print("Unrecognised line: {}".format(line))

        # Calculate occupancies
        self.occ = self.calculate_occ()

        # Rationalise if requested
        if clamp:
            if not self.mp_grid:
                self.mp_grid = find_mp_grid(self.kpts)
            self.kpts = [clamp_to_grid(k, self.mp_grid) for k in self.kpts]
            
    def calculate_occ(self):
        occ = []
        for kpt_index, bands in enumerate(self.kpt_bands):
            kpt_occ = []
            for spin_index, spin_bands in enumerate(bands):
                spin_occ = sum(1 for b in spin_bands if b <= self.fermi[spin_index])
                kpt_occ.append(spin_occ)
            occ.append(kpt_occ)
        return occ

    def get_fermi_surface(self, energy_tol):
        """Tolerance gives an energy range(in Hartrees) to count as partially occupied"""
        fs = []
        for k, bands in enumerate(self.kpt_bands):
            if any(abs(b - self.fermi[s]) < energy_tol for s in range(self.nspins) for b in bands[s]):
                fs.append(self.kpts[k])
        return fs


def clamp_to_grid(vals, grid):
    v = [round(2 * x * nx) / (2*nx) for x, nx in zip(vals, grid)]
    if any(abs(x-x1) > 1e-6 for x, x1 in zip(v, vals)):
        print(f"Large discrepancy in kpoint grid detected: {vals} (in bands file) vs {v} (on grid)")
    return v
        

def rationalise(real_number):
    return Fraction(real_number).limit_denominator(200)

from fractions import Fraction
from math import gcd


def lcm(x, y):
    return x * y // gcd(x,y)

def find_mp_grid(kpts):
    lcdx = 1
    lcdy = 1
    lcdz = 1
    
    for k in kpts:
        kfrac = list(map(rationalise, k))
        lcdx = lcm(lcdx, kfrac[0].denominator)
        lcdy = lcm(lcdy, kfrac[1].denominator)
        lcdz = lcm(lcdz, kfrac[2].denominator)

    return lcdx, lcdy, lcdz
