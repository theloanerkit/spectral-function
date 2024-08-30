#!/usr/bin/env python3
import shutil
from itertools import combinations, islice
from enum import Enum
import argparse
from readers.bands_reader import BandStructure


def generate_qpts(fermi_surface, target_number=100):
    """Generate list of qpts linking points on the fermi surface"""
    from operator import sub
    nkpts = len(fermi_surface)
    nqpts = nkpts * (nkpts - 1) // 2
    step = max(nqpts // target_number, 1)

    for k1, k2 in islice(combinations(fermi_surface, 2), 0, step * target_number, step):
        yield k1, k2


class CellMode(Enum):
    """Enum for cell writing mode (ep or phonon)"""
    EP_KPOINT_PAIRS = 1
    PHONON_FINE_KPOINT_LIST = 2


def generate_ep_cell(fs, seed_name, num_qpts, mode):
    """Write a cell file containing some number of qpts linking points on the fermi surface"""
    if mode == CellMode.EP_KPOINT_PAIRS:
        target_cell = seed_name + "-ep.cell"
        qpt_header = "ep_kpoint_pairs"
    elif mode == CellMode.PHONON_FINE_KPOINT_LIST:
        target_cell = seed_name + "-ph.cell"
        qpt_header = "phonon_fine_kpoint_list"
    else:
        raise Exception(f"Invalid mode!: {mode}")

    # Copy cell file
    print(f"Writing {mode.name} to {target_cell}: {num_qpts} pairs")
    cell_file = seed_name + ".cell"
    shutil.copyfile(cell_file, target_cell)

    # Write additional ep_kpoint_pairs
    with open(target_cell, "a") as ecf:
        ecf.write("\n")
        ecf.write(f"%block {qpt_header}\n")

        # Generate k, k' pairs
        for k1, k2 in generate_qpts(fs, num_qpts):
            if mode == CellMode.EP_KPOINT_PAIRS:  # write k and k'
                ecf.write(" ".join(f"{x:16.12f}" for x in k1))
                ecf.write(" ".join(f"{x:16.12f}" for x in k2))
            elif mode == CellMode.PHONON_FINE_KPOINT_LIST:  # write k'-k
                ecf.write(" ".join(f"{y - x:16.12f}" for x, y in zip(k1, k2)))
            ecf.write("\n")
        ecf.write(f"%endblock {qpt_header}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finds the fermi surface from a DOS calculation and generates electron-phonon input files')
    parser.add_argument("seedname", help="Seedname of .bands file. eg Al for Al.bands")
    parser.add_argument("-ph", "--num_ph", help="Number of phonon interpolation points to generate", default=10000,
                        type=int)
    parser.add_argument("-ep", "--num_ep", help="Number of electron-phonon pairs to generate", default=100, type=int)
    parser.add_argument("-mp", "--mp_grid", help="The original kpoint grid used to generate the DOS", type=int, nargs=3)
    parser.add_argument("-c", "--clamp", help="Clamp the kpoints read onto an MP grid", action='store_true')
    args = parser.parse_args()

    # Read .bands file
    b = BandStructure(args.seedname + ".bands", args.clamp, args.mp_grid)
        
    # Generate fermi surface of points within 0.01 Ha of the fermi energy
    fs = b.get_fermi_surface(0.01)
    print(f"Number of fermi surface points: {len(fs)}")
    generate_ep_cell(fs, args.seedname, args.num_ep, mode=CellMode.EP_KPOINT_PAIRS)
    generate_ep_cell(fs, args.seedname, args.num_ph, mode=CellMode.PHONON_FINE_KPOINT_LIST)
