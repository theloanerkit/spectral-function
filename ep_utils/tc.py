#!/usr/bin/env python3
"""
Script to generate electron-phonon mediated superconducting transition
temperature from Castep electron-phonon data.
"""
from math import exp, log, sinh
import numpy as np
import argparse
import matplotlib.pyplot as mp
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import scipy.constants as sc
from pint import UnitRegistry
from readers.phonon_reader import PhononReader
from readers.epme_reader import EpmeReader

ureg = UnitRegistry()
mp.rcParams.update({'font.size': 16})
mp.rcParams.update({'figure.autolayout': True})

def dividew(xpoints, ypoints):
    '''Divide the y-points by the x-points'''
    return [y / x for x, y in zip(xpoints, ypoints)]

def logw(xpoints, ypoints):
    '''Calculate the logarithmically averaged y-points'''
    return [log(x) * y / x for x, y in zip(xpoints, ypoints)]

def integratew(xpoints, ypoints):
    '''Integrate reciprocally weighted points'''
    xpoints = [x.magnitude for x in xpoints]
    ypoints = [y.magnitude for y in ypoints]
    return integrate.simpson(dividew(xpoints, ypoints), x=xpoints)

def integratewlog(xpoints, ypoints):
    '''Integrate logarithmically weighted points'''
    xpoints = [x.magnitude for x in xpoints]
    ypoints = [y.magnitude for y in ypoints]
    return integrate.simpson(logw(xpoints, ypoints), x=xpoints)

def remove_dimension(arr):
    return [a.magnitude for a in arr]

def plot_func(filename, xpts, ypts, xlabel, ylabel, fmt="r-"):
    '''Plot the function given to a file'''
    if type(xpts)==ureg.Quantity:
        xpts = remove_dimension(xpts)
    if type(ypts)==ureg.Quantity:
        ypts = remove_dimension(ypts)
    mp.figure()
    mp.xlabel(xlabel)
    mp.ylabel(ylabel)
    mp.plot(xpts,ypts,fmt)
    mp.savefig(filename)

def plot_many_funcs(filename, xpts, ypts, xlabel, ylabel, labels, fmts):
    '''Plot the functions given to a file'''
    mp.figure()
    mp.xlabel(xlabel)
    mp.ylabel(ylabel)
    for i in range(len(xpts)):
        if type(xpts[i])==ureg.Quantity:
            xpts[i] = remove_dimension(xpts[i])
        if type(ypts[i])==ureg.Quantity:
            ypts[i] = remove_dimension(ypts[i])
        if labels:
            mp.plot(xpts[i], ypts[i], fmts[i], label=labels[i])
        else:
            mp.plot(xpts[i], ypts[i], fmts[i])
    if labels:
        mp.legend()
    mp.savefig(filename)

def generate_a2f(name, ep_filename, ph_filename, fermi_dos, bin_size=50):
    '''Reads EPME and phonon data and generates the eliashberg spectral function a2f(w)'''

    # set units for fermi dos
    fermi_dos = fermi_dos * ureg.hartree_energy**-1

    # read phonon and ep data from respective files
    epdata = EpmeReader(ep_filename, ureg)
    phdata = PhononReader(ph_filename, ureg)

    # generate weighted phonon dos (ie F)
    xpts, ph_y, xpts_mag = phdata.plot(width=0.01)

    # normalise phonon data to n(Ef)
    ph_norm = integrate.simpson(ph_y, x=xpts_mag)
    ph_y *= (fermi_dos / epdata.cell_volume) / ph_norm
    plot_func(name+"-f.png", xpts.to("terahertz")/(2*sc.pi), ph_y, "Frequency (THz)", r"$F(\omega)$")

    # generate an average epme as a function of frequency
    epf = epdata.get_interpolator(bin_size=bin_size, energy_tol=0.01*ureg.hartree_energy)

    # generate a^2 on those same points
    ep_y = epf(xpts_mag)*ureg.bohr**3/ureg.electron_volt
    plot_func(name+"-a2.png", xpts.to("terahertz")/(2*sc.pi), ep_y, "Frequency (THz)", r"$\alpha^2(\omega)$")

    # full a2f is the product of these contributions
    tot_points = ep_y * ph_y

    # plot a2f
    plot_func(name + "-a2f.png", xpts.to("terahertz")/(2*sc.pi), tot_points, "Frequency (THz)",
              r"Eliashberg Spectral Function, $\alpha^2F(\omega)$")
    
    # make an interpolator for the spectral function
    spectral_function = interp1d(xpts_mag, remove_dimension(tot_points), fill_value=0, bounds_error=False)

    # integrate to get the ep coupling constant (lambda)
    ep_coupling_constant = integratew(xpts, tot_points)

    # calculate logarithmically averaged frequency
    wlog = exp(integratewlog(xpts, tot_points) / ep_coupling_constant)

    # convert to kelvin (from energy)
    Ha = sc.value('Hartree energy')
    Kb = sc.value('Boltzmann constant')
    wlog_si = wlog * Ha / Kb

    return spectral_function, ep_coupling_constant, wlog_si, xpts, tot_points

def generate_a2trf(name, ep_filename, ph_filename, fermi_dos, bin_size=50):
    
    # read phonon and ep data from respective files
    epdata = EpmeReader(ep_filename, ureg)
    phdata = PhononReader(ph_filename, ureg)
    # generate weighted phonon dos (ie F)
    
    xpts, ph_y, xpts_mag = phdata.plot(width=0.01)

    # normalise phonon data to n(Ef)
    ph_norm = integrate.simpson(ph_y, x=xpts_mag)
    ph_y *= (fermi_dos / epdata.cell_volume) / ph_norm

    # generate an average epme as a function of frequency
    epf, v = epdata.get_interpolator_tr(bin_size=bin_size, energy_tol=0.01*ureg.hartree_energy)

    # generate a^2 on those same points
    ep_y = epf(xpts_mag)*ureg.bohr**3/ureg.electron_volt
    
    # plot a2trf
    plot_func(name+"-a2tr.png", xpts.to("terahertz")/(2*sc.pi), ep_y, "Frequency (THz)", r"$\alpha^2(\omega)$")

    # full a2trf is the product of these contributions
    tot_points = ep_y * ph_y

    temp = epdata.get_interpolator(bin_size=bin_size, energy_tol=0.01*ureg.hartree_energy)
    tempy = temp(xpts_mag)
    temp_points = tempy * ph_y

    plot_func(name + "-a2trf.png", xpts.to("terahertz")/(2*sc.pi), tot_points, "Frequency (THz)",
              r"Transport Spectral Function, $\alpha^2F(\omega)$")
    
    plot_many_funcs(name+"-both", [xpts.to("terahertz")/(2*sc.pi), xpts.to("terahertz")/(2*sc.pi)], [tot_points, temp_points], "Frequency (THz)", " ", [r"$\alpha^2_{tr}F(\omega)$", r"$\alpha^2F(\omega)$"], ["-r", "-b"])

    # make an interpolator for the spectral function
    transport_spectral_function = interp1d(xpts_mag, remove_dimension(tot_points), fill_value=0, bounds_error=False)

    return transport_spectral_function, xpts, tot_points, epdata.cell_volume, v

def transition_temp(ep_constant, mu, wlog):
    '''Calculate superconducting transition temperature from McMillan formula'''
    t1 = 1.04 * (1.0 + ep_constant)
    t2 = ep_constant - mu * (1.0 + 0.62 * ep_constant)
    if t1 < 0 or t2 < 0:
        print("mu too big or lambda too small!")
    expterm = exp(-t1 / t2)
    t_factor = wlog / 1.20
    return t_factor * expterm

def resistivity_constant(Omega, dos, avg_v2):
    """calculate constant for thermal and electrical resistivity"""
    n = 3 * sc.pi * Omega
    d = (sc.e*ureg.coulomb)**2 * dos * avg_v2
    return n/d

def calculate_rho_el(w, a2trf,T):
    """calculate the electrical resistivity at temperature T"""
    c = (sc.hbar*ureg.joule*ureg.second)/(2*(sc.k*ureg.joule/ureg.kelvin)*T)
    n = list(map(lambda x: c*x, w))
    d = list(map(lambda x: (sinh(c*x))**2, w))
    y = np.multiply(np.divide(n,d),a2trf)
    return integrate.simpson(remove_dimension(y), x=remove_dimension(w))*ureg.hertz

def calculate_rho_th(w,a2trf,a2f,T):
    """calculate the thermal resistivity at temperature T"""
    c = (sc.hbar*ureg.joule*ureg.second)/(2*(sc.k*ureg.joule/ureg.kelvin)*T)
    n = list(map(lambda x: c*x, w))
    d = list(map(lambda x: (sinh(c*x))**2, w))
    one = list(map(lambda x: 1-(2/sc.pi**2)*(c*x)**2,w))
    two = list(map(lambda x: (6/sc.pi**2)*(c*x)**2,w))
    y1 = np.multiply(np.multiply(np.divide(n,d),one),a2trf)
    y2 = np.multiply(np.multiply(np.divide(n,d),two),a2f)
    return integrate.simpson(remove_dimension(y1), x=remove_dimension(w))*ureg.hertz + integrate.simpson(remove_dimension(y2), x=remove_dimension(w))*ureg.hertz

def plot_electrical_resistivity(element, con, a2trf_x, a2trf_y, temp=500):
    """plot the electrical resistivity"""
    xpts = []
    yptsel = []
    for t in range(1,temp):
        num = calculate_rho_el(a2trf_x.to(ureg.hertz)/(2*sc.pi), a2trf_y, t*ureg.kelvin)
        xpts.append(t)
        yptsel.append(num*con)
    yptsel = [y.to_base_units() for y in yptsel]
    yptsel = [y.to(ureg.microohm*ureg.centimeter) for y in yptsel]
    yptsel = [y.magnitude for y in yptsel]
    plot_func(element+"-rho_el", xpts, yptsel, "Temperature (K)", r"$\rho_{el}$ ($\mu\Omega$ cm)")
    return yptsel

def plot_thermal_resistivity(element, con, a2trf_x, a2trf_y, a2f_y, temp=500):
    """plot the thermal resistivity"""
    xpts = []
    yptsth = []
    l0 = (1/3) * ((sc.pi*(sc.k*ureg.joule/ureg.kelvin))/(sc.e*ureg.coulomb))**2
    for t in range(1,temp):
        num = calculate_rho_th(a2trf_x.to(ureg.hertz)/(2*sc.pi),a2trf_y, a2f_y, t*ureg.kelvin)
        xpts.append(t)
        yptsth.append((num*con)/(l0*(t*ureg.kelvin)))
    yptsth = [y.to_base_units() for y in yptsth]
    yptsth = [y.to(ureg.centimeter*ureg.kelvin/ureg.watt) for y in yptsth]
    yptsth = [y.magnitude for y in yptsth]
    plot_func(element+"-rho_th", xpts, yptsth, "Temperature (K)", r"$\rho_{th}$ (cm K/W)")
    return xpts, yptsth


def calculate_tc(element,fermi_dos_per_ev, mu, ep_filename, ph_filename):
    '''Generate the spectral function and calculate Tc for a material'''
    print(f"Element: {element}")

    eV = sc.value('Hartree energy in eV')

    fermi_dos = fermi_dos_per_ev*eV*ureg.hartree_energy**-1

    a2f, ep_lambda, wlog, a2f_x, a2f_y = generate_a2f(element, ep_filename, ph_filename, fermi_dos)
    a2trf, a2trf_x, a2trf_y, Omega, v = generate_a2trf(element, ep_filename, ph_filename, fermi_dos)
    print(f"EP Coupling Constant: {ep_lambda}")
    print(f"Logarithmically averaged frequency: {wlog} K")
    tc = transition_temp(ep_lambda, mu, wlog)
    print(f"T_c = {tc} K with mu*={mu}")

    con = resistivity_constant(Omega, fermi_dos, np.average(v))
    con = con.to_base_units()

    yptsel = plot_electrical_resistivity(element, con, a2trf_x, a2trf_y, 500)

    xpts, yptsth = plot_thermal_resistivity(element, con, a2trf_x, a2trf_y, a2f_y, 500)
    y1 = np.divide(np.divide(yptsel,np.multiply(yptsth,10**6)),xpts)
    y2 = np.full(len(xpts),2.44E-8)
    plot_many_funcs(element+"-lorenz", [xpts, xpts], [y1,y2], "Temperature (K)", r"L $V^2\;K^{-2}$", labels=None, fmts=["r-", "k--"])

def main():
    # Parse arguments and calculate
    parser = argparse.ArgumentParser(description="Reads castep .epme file and .phonon file"
                                     " to calculate superconducting transition temperature")
    parser.add_argument("seedname", help="seedname to read files from -- both seedname.epme"
                        " and seedname-ph.phonon should be in the current folder")


    parser.add_argument("nef", help="DOS at the fermi energy (in eV^1)", type=float)
    parser.add_argument("mu", default=0.11, type=float, help=
                        "Screen coulomb interaction constant for the material")

    parser.add_argument("-ep", "--epme_file", type=str, default=None,
                        help="Electron-phonon matrix elements file to read (.empe)")
    parser.add_argument("-ph", "--phonon_file", type=str, default=None,
                        help="Phonon mode file to read (.phonon)")


    args = parser.parse_args()

    if args.phonon_file is None:
        args.phonon_file = args.seedname + ".phonon"

    if args.epme_file is None:
        args.epme_file = args.seedname + ".epme"

    calculate_tc(args.seedname, args.nef, args.mu, args.epme_file, args.phonon_file)


if __name__ == "__main__":
    main()