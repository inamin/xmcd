#!/bin/sh
""""
exec python "$0" ${1+"$@"}
"""
__doc__ = """The above defines the script's __doc__ string. \
             You can fix it by like this."""
#coding:utf-8
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

import argparse
import configparser
import os
import sys
import numpy as np
import pandas as pd
import csv
from matplotlib import pylab as plt
from scipy import signal, interpolate, integrate
from scipy.optimize import curve_fit
from fractions import Fraction


class Xas:
    """ X-ray absorption spectrum """
    def __init__(self, filename):
        step = 0.1
        param = np.array([])  # type: float[param]
        cov = np.array([])  # type: float[cov]
        fit_preedge = np.array([])  # type: float[fit_preedge]

        self.ene, self.i0, self.i1 = self._loaddat_pf(filename)
        self.mu = self.i1 / self.i0 # normalize by I0
        start = round(self.ene[0])
        stop = round(self.ene[-1])
        num = (stop - start) / step + 1
        self.energy = np.linspace(start, stop, num, endpoint=True)
        self.mui = self._interpolate()
        self.e0 = self._find_e0()

        print(self.e0)
        
#        self.fit_preedge = self._preedgefitting(elem.get_preedgemask())

    def getelem(self):
        return self.elem

    def setelem(self, elem):
        self.elem = elem

    def _loaddat_pf(self, filename):
        """
        load PF BL16A data
        """
        return np.loadtxt(filename, delimiter=" ", usecols=(0, 1, 2), unpack=True)

    def _interpolate(self):
        f = interpolate.interp1d(self.ene, self.mu,
                                 kind="quadratic", fill_value="extrapolate")
        return f(self.energy)

    def fitting_preedge(self):
        """ pre-edge fitting with linear function """
        mask = self.elem.get_preedgemask()
        self.fit_preedge = np.poly1d(np.polyfit(self.energy[mask], self.mui[mask], 2))(self.energy)

    def _find_e0(self):
        """
        return E0
        """
        muidiff = np.diff(self.mui)
        e0index = np.argmax(muidiff)
        return self.energy[e0index]

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.ene, self.mu, "o")
        plt.plot(self.energy, self.mui, "r", label="quadratic")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read Xas data.')
    parser.add_argument('filename', metavar='filename',
                        help='a filename of XAS datafile')
    parser.add_argument('-c', dest='opt_c', action='store',
                        help='a config file')
    args = parser.parse_args()
    xas = Xas(args.filename)
    xas.plot()
