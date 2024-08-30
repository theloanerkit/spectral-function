import numpy as np
import matplotlib.pyplot as mp
from scipy import interpolate
from pint import UnitRegistry

class Epme:
    '''Class to hold electron-phonon data from .epme file'''
    def __init__(self,ph_idx, band_i, band_f, ph_freq, E_i, E_f, v_i, v_f, matrix_element):
        self.ph_idx = ph_idx
        self.band_i = band_i
        self.band_f = band_f
        self.ph_freq = ph_freq
        self.E_i = E_i
        self.E_f = E_f
        self.v_i = v_i
        self.v_f = v_f
        self.matrix_element = matrix_element
        self.mag = abs(self.matrix_element)

    @staticmethod
    def from_line(line_string,ureg):
        split_line = line_string.split()
        #print(float(split_line[12]))
        return Epme(
            ph_idx=int(split_line[0]),
            band_i=int(split_line[1]),
            band_f=int(split_line[2]),
            ph_freq=float(split_line[3])*ureg.atomic_unit_of_time**-1,
            E_i=float(split_line[4])*ureg.hartree_energy,
            E_f=float(split_line[5])*ureg.hartree_energy,
            v_i=list(map(float,split_line[6:9]))*(ureg.bohr/ureg.atomic_unit_of_time),
            v_f=list(map(float,split_line[9:12]))*(ureg.bohr/ureg.atomic_unit_of_time),
            matrix_element=float(split_line[12])
        )

    def __repr__(self):
        return "Epme<{0.ph_freq},{0.band_i},{0.band_f}:{0.mag}>".format(self)
    
class EpmeReader:
    """Reader for a .empe electron-phonon coupling file"""
    def __init__(self, filename, ureg):
        self.epme_data = []
        self.version_no = None
        self.fermi_energy = None
        self.cell_volume = None
        self.num_ph = None
        self.num_bands = None
        self.ureg = ureg

        with open(filename) as fd:
            for line in fd:
                if "Electron-phonon coupling file" in line:
                    pass
                elif "Version" in line:
                    self.version_no = line.split()[-1]
                elif "All units are atomic units" in line:
                    pass
                elif "Unit cell volume" in line:
                    self.cell_volume = float(line.split()[-1]) * self.ureg.bohr**3
                elif "Fermi Energy" in line:
                    self.fermi_energy = float(line.split()[-1]) * self.ureg.hartree_energy
                elif "Number of bands:" in line:
                    self.num_bands = int(line.split()[-1])
                elif "Number of phonon branches:" in line:
                    self.num_ph = int(line.split()[-1])
                elif "Electron-Phonon coupling matrix elements between kpoints" in line:
                    pass
                elif "Ph index  Band_i  Band_f      ph_freq         E_i         E_f" in line:
                    pass
                else:
                    try:
                        self.epme_data.append(Epme.from_line(line,self.ureg))
                    except ValueError:
                        print("Unrecognised line:", line)

        assert self.version_no == "1"

    def get_fs_epme(self, energy_tol):
        """Returns only epme elements with initial and final states within energy_tol of fermi energy"""
        fs_epme = []

        for epme in self.epme_data:
            if close(epme.E_i, self.fermi_energy, energy_tol) and close(epme.E_f, self.fermi_energy, energy_tol):
                fs_epme.append((epme.ph_freq, epme.mag))
        return fs_epme
    
    def get_fs_epme_tr(self, energy_tol):
        """Returns only empe elements and velocities with initial and final states within energy_tol of fermi energy"""
        fs_epme = []

        for epme in self.epme_data:
            if close(epme.E_i, self.fermi_energy, energy_tol) and close(epme.E_f, self.fermi_energy, energy_tol):
                fs_epme.append((epme.ph_freq, epme.mag, epme.v_i, epme.v_f))
        return fs_epme
    
    def get_interpolator(self, bin_size, energy_tol):

        mp.figure()
        samples = sorted(self.get_fs_epme(energy_tol))
        xpoints, weights = zip(*samples)
        xpoints_nounit = np.array([x.magnitude for x in xpoints])
        ebins,left_over = gen_bins(xpoints_nounit,bin_size)
        norm_weights = np.divide(weights,float(bin_size))
        if left_over > 0:
            norm_weights[-left_over:] *= float(bin_size) / float(left_over)

        hist,bins,_ = mp.hist(xpoints_nounit,weights=norm_weights,bins=np.array(ebins))
        
        f = GetInterpolator(hist,bins)

        mp.close()
        return f
    
    def get_interpolator_tr(self, bin_size, energy_tol):

        mp.figure()

        samples = sorted(self.get_fs_epme_tr(energy_tol))
        xpoints, weights, v_i, v_f = zip(*samples)

        xpoints_nounit = np.array([x.magnitude for x in xpoints])
        effs, v = self.efficiency(v_i, v_f)
        
        ebins,left_over = gen_bins(xpoints_nounit,bin_size)
        norm_weights = np.divide(np.multiply(weights,effs),float(bin_size))
        
        if left_over > 0:
            norm_weights[-left_over:] *= float(bin_size) / float(left_over)
        
        hist,bins,_ = mp.hist(xpoints_nounit,weights=norm_weights,bins=ebins)
        
        f = GetInterpolator(hist,bins)
        
        mp.close()
        
        return f, v

    def efficiency(self,v_i,v_f):
        effs = []
        v = []
        for i in range(len(v_i)):
            v_initial = np.asarray(v_i[i].magnitude)
            v_final = np.asarray(v_f[i].magnitude)
            v.append(np.dot(v_initial, v_initial))
            v.append(np.dot(v_final, v_final))
            a = np.dot(v_initial, v_final)
            b = np.dot(v_initial, v_initial)
            eff = 1 - (a/b)
            effs.append(eff)
        v *= self.ureg.bohr**2/self.ureg.atomic_unit_of_time**2
        return effs, v

def GetInterpolator(hist,bins):
    xp = [(x1+x0)/2.0 for x0,x1 in zip(bins,bins[1:])]
    xp = [0] + xp + [10]
    hist = [hist[0]] + list(hist) + [hist[-1]]
    return interpolate.PchipInterpolator(xp,hist) # GOOD ONE

def gen_bins(xpoints,per_bin):
    xp = sorted(xpoints)
    return ([0.0] + xp[per_bin::per_bin] + [xp[-1]],len(xp)%per_bin)

def close(a, b, tol):
    return abs(a-b) < tol