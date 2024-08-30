import numpy as np
import scipy.constants as constants
from scipy.stats import norm
from pint import UnitRegistry

class PhononReader:
    """Reader for a castep .phonon file"""
    def __init__(self, filename, ureg):
        """Read a phonon file and return an object containing the data"""
        self.num_q = 0
        self.num_ions = 0
        self.num_branches = 0
        self.freqs = []
        self.weights = []

        with open(filename) as filedata:
            data = filedata.read().splitlines()
        d1 = iter(data)
        for d in d1:
            if "Number of ions" in d:
                self.num_ions = int(d.split()[-1])
                self.num_branches = self.num_ions * 3
            elif "Number of wavevectors" in d:
                self.num_q = int(d.split()[-1])
            elif "Frequencies in" in d:
                unit_name = d.split()[-1]
                if unit_name == "cm-1":
                    self.freq_unit = constants.physical_constants["atomic unit of length"][0]
                    self.freq_unit *= 100.0 * 137.0 * 2.0 * constants.pi
                else:
                    print("Unknown unit????")
            elif "q-pt=" in d:
                ld = d.split()
                weight = float(ld[-1])
                self.weights.append(weight)
                qfreq = []
                for l in range(self.num_branches):
                    td = float(next(d1).split()[1]) * self.freq_unit * ureg.atomic_unit_of_time**-1
                    qfreq.append(td)
                self.freqs.append(qfreq)

    def plot(self, width=0.001, num_samples=500):
        """Generate a DOS plot from the phonon points"""
        max_x = max(map(max, self.freqs)) * 1.5
        width *= max_x
        xpoints = np.linspace(width / 100, max_x, num_samples)
        gaussian = norm(0, scale=width)
        freqs_mag = []
        for freq in self.freqs:
            freqs_mag.append([f.magnitude for f in freq])
        xpoints_mag = np.asarray([x.magnitude for x in xpoints])
        y_points = np.sum(weight * gaussian.pdf(xpoints_mag - f) for weight, freqs in zip(self.weights, freqs_mag) for f in freqs)
        return [xpoints, y_points, xpoints_mag]