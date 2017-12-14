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

from xas import Xas

class Xmcd:
    """ handle XMCD data """

    def __init__(self, cfg, filenames, elem):
        """ read and analyze XAS and XMCD """
        # XXX DO CHANGE, like following
        # self.mcd{"mcd": mcd, "bg": mcdbg, "int": mcdbgitg}
        #
        self.elem = Element(cfg, elem)
        self.xas = [Xas(filename) for filename in filenames]
        for x in self.xas:
            x.setelem(self.elem)
            x.fitting_preedge()
        self.ratio = cfg.get_ratio(elem)


        # XXX not python like XXX
        self.energyL3 = cfg.get_energyL3(elem)
        self.energyL2 = cfg.get_energyL2(elem)
        rangeL3 = cfg.get_rangeL3(elem)
        self.energyL3st = rangeL3["start"]
        self.energyL3en = rangeL3["end"]
        rangeL2 = cfg.get_rangeL2(elem)
        self.energyL2st = rangeL2["start"]
        self.energyL2en = rangeL2["end"]


    def calc(self):
        """ calc XMCD spectrum """
        mask = self.elem.get_preedgemask()
        p = [xas.fit_preedge[mask] for xas in self.xas]
        y = p[0] / p[1] # XXX
        pmave = np.average(np.array(y))
        self.xascnst = self.xas[0].mui / pmave
        self.sumxas = self.xascnst + self.xas[1].mui
        self.mcd = self.xascnst - self.xas[1].mui


    def calc_moment(self):
        self._integrate_mcd()
        self._integrate_xas()
        moments = self._calc_sumrules()
        print(moments)


    def _integrate_mcd(self):
        self.mcdfit = self._fit_mcd()
        self.mcdbg = self.mcd - self.mcdfit
        self.mcdbgitg = integrate.cumtrapz(self.mcdbg, self.elem.x, initial=0)


    def _integrate_xas(self):
        self.sumfit = self._fit_sum()
        self.sumbg = self.sumxas - self.sumfit

        # find L3 & L2 peaks
        self.maxId = signal.argrelmax(self.sumxas)
        self.energyL3, self.energyL2  = self.elem.x[self.maxId[0]]
    
        coeff = [self.energyL3, self.energyL2, 0.2, 0, self.ratio, 0]
        mask = [(self.elem.x <= (self.energyL3 + self.energyL3st)) | (self.elem.x >= (self.energyL2 + self.energyL2en))]
        xs = self.elem.x[mask]
        ys = self.sumbg[mask]
        self.ys_fit = self._sum_back(coeff, xs, ys)
        self.sumbgat = self.sumbg - self.ys_fit
        self.sumbgatitg = integrate.cumtrapz(self.sumbgat, self.elem.x, initial=0)


    def _fit_mcd(self):
        """ fit MCD background """
        mask = self.elem.get_mask_mcd()
        self.x_for_fit = self.elem.x[mask]
        self.mcd_for_fit = self.mcd[mask]
        z = np.polyfit(self.x_for_fit, self.mcd_for_fit, 5)
        z5 = np.poly1d(z)
        return z5(self.elem.x)


    def _fit_sum(self):
        """ fit Sum background """
        mask = self.elem.get_mask_sum()
        xs = self.elem.x[mask]
        ys = self.sumxas[mask]
        return np.poly1d(np.polyfit(xs, ys, 1))(self.elem.x)


    def _sum_back(self, coeff, xs, ys):
        """ arctan type background fitting """
        xl3, xl2, hwhm, al3, l23ratio, const = coeff
        def func(x, xl3, xl2, hwhm, al3, l23ratio, const):
            return const + al3*np.arctan((x - xl3)/hwhm) + al3 * l23ratio * np.arctan((x - xl2)/hwhm)

        popt, pcov = curve_fit(lambda x, a, b: func(x, xl3, xl2, hwhm, a, l23ratio, b), xs, ys)
        al3, const = popt
        return func(self.elem.x, xl3, xl2, hwhm, al3, l23ratio, const)


    def _calc_sumrules(self):
        """
        calc moments by the sumrule
        """

        pc = 1.0 # XXX
        holes, offset_p, id_q, id_r, offset_r0 = self.elem.get_rulesparam()

        # XXX self.sumbgatitg.get_nearest_value_index(value)

#        p = self.mcdbgitg[get_nearest_value_index(self.elem.x, (self.energyL2 + offset_p))]
        p = self.mcdbgitg[get_nearest_value_index(self.elem.x, (offset_p))]
        q = self.mcdbgitg[id_q]
        r = self.sumbgatitg[id_r]
#        r0 = self.sumbgatitg[get_nearest_value_index(self.elem.x, (self.energyL3 + offset_r0))]
        r0 = self.sumbgatitg[get_nearest_value_index(self.elem.x, (offset_r0))]
        r -= r0

        print("{0:.5f}".format(p))
        print("{0:.5f}".format(q))
        print("{0:.5f}".format(r))
        print("{0:.5f}".format(r0))

        c = holes / pc
        ms = (-(6 * p - 4 * q) / r) * c
        mo = ((-4 * q) / (3 * r)) * c
        ratio = 2 * q / (9 * p - 6 *q)

        return [ms, mo, ratio, p, q, r, r0]
 

    def plot(self):
        """ plot XAS and XMCD spectra """
        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        ax4 = fig.add_subplot(2,3,4)
        ax5 = fig.add_subplot(2,3,5)
        ax6 = fig.add_subplot(2,3,6)
        ax1.plot(self.xas[0].ene, self.xas[0].mu, "o", label="raw")
        ax1.plot(self.xas[0].energy, self.xas[0].mui, "r", label="xas0")
        ax1.plot(self.xas[1].energy, self.xas[1].mui, "b", label="xas1")
        ax1.plot(self.xas[0].energy, self.xas[0].fit_preedge, "r--", label="xas0.y")
        ax1.plot(self.xas[0].energy, self.xas[1].fit_preedge, "b--", label="xas1.y")
        ax1.tick_params(axis='both', which='both', direction='in')
        ax1.legend()
        ax4.plot(self.elem.x, self.xascnst, "g", label="xascnst")
        ax4.plot(self.elem.x, self.xas[1].mui, "b", label="xas1")
        ax4.plot(self.elem.x, self.sumxas, "g", label="sumxas")
        ax4.plot(self.elem.x, self.sumfit, "r--", label="sumfit")
        ax4.plot(self.elem.x, self.sumbg, "r", label="sum_bg")
        ax4.tick_params(axis='both', which='both', direction='in')
        ax4.legend()
        ax2.plot(self.elem.x, self.mcd, "g", label="mcd")
        ax2.plot(self.elem.x, self.mcdfit, "r", label="mcd_fit")
        ax2.plot(self.x_for_fit, self.mcd_for_fit, 'b+', label="fit")
        ax2.plot(self.elem.x, self.mcdbg, "m", label="mcd_bg")
        ax2.tick_params(axis='both', which='both', direction='in')
        ax2.legend()
        ax5.plot(self.elem.x, self.mcdbgitg, "y", label="mcd_bg_int")
        ax5.tick_params(axis='both', which='both', direction='in')
        ax5.legend()
        ax3.plot(self.xas[0].energy, self.sumxas, "g", label="sumxas")
        ax3.plot(self.elem.x[self.maxId], self.sumxas[self.maxId], "go", label="peak2")
        ax3.plot(self.elem.x, self.ys_fit, "r", label="arctan")
        ax3.plot(self.elem.x, self.sumbgat, "b", label="arctanbg")
        ax3.tick_params(axis='both', which='both', direction='in')
        ax3.legend()
        ax6.plot(self.elem.x, self.sumbgatitg, "g", label="arctanbgint")
        ax6.tick_params(axis='both', which='both', direction='in')
        ax6.legend()
        plt.show()

        filename = 'figure1'
        ext = '.pdf'
        fig.savefig(filename + ext)
        ext = '.png'
        fig.savefig(filename + ext)


class Element:
    """ element parameter """
    def __init__(self, cfg, elem):
        self.elem = elem

        st, en, step = cfg.get_energyrange(elem)
        self.x = np.linspace(st, en, int((en - st)/step + 1))

        st, en = cfg.get_preedgerange(elem)
        self.preedgemask = [(self.x >= st) & (self.x <= en)]

        pre, inter, post = cfg.get_maskrange(elem)
        self.mcdmask = [(self.x <= pre["end"]) |
                       ((self.x >= inter["start"]) & (self.x <= inter["end"])) |
                        (self.x >= post["start"])]
        self.summask = self.x >= post["start"]

        self.energyL3 = cfg.get_energyL3(elem)
        self.energyL2 = cfg.get_energyL2(elem)

        self.rulesparam = cfg.get_rulesparam(elem)
        self.rulesparam[1] += self.energyL2
        self.rulesparam[2] = -1 # XXX
        self.rulesparam[3] = -1 # XXX
        self.rulesparam[4] += self.energyL3


    def get_energy(self):
        """ return interpolation range """
        return self.x


    def get_preedgemask(self):
        """ return preedge mask for background fitting """
        return self.preedgemask


    def get_mask_mcd(self):
        """ return mcd mask """
        return self.mcdmask

    def get_mask_sum(self):
        """ return sum mask """
        return self.summask

    def get_rulesparam(self):
        return self.rulesparam


class Config:
    """ read config file """
    def __init__(self, filename):
        """ load an inifile """
        self.cfg = configparser.SafeConfigParser()
        if os.path.exists(filename):
            self.cfg.read(filename, encoding='utf8')
        else:
            sys.stderr.write(filename + " not found")
            sys.exit(2)

    def get_sections(self):
        """ print sections """
        for section in self.cfg.sections():
            for key in self.cfg.options(section):
                print(self.cfg.get(section, key))

    def get_energyrange(self, elem):
        """ get energy range """
        section = self.cfg[elem]
        st = section.getfloat('startenergy')
        en = section.getfloat('endenergy')
        step = section.getfloat('stepenergy')
        return [st, en, step]

    def get_preedgerange(self, elem):
        section = self.cfg[elem]
        st = section.getfloat('preedgestart')
        width = section.getfloat('preedgewidth')
        en = st + width
        return [st, en]

    def get_maskrange(self, elem):
        section = self.cfg[elem]
        energyL3 = section.getfloat('energyL3')
        energyL2 = section.getfloat('energyL2')
        st = section.getfloat('startenergy')
        en = energyL3 + section.getfloat('pre.en.offset')
        pre = {"start":st, "end":en}

        st = energyL3 + section.getfloat('inter.st.offset')
        en = energyL3 + section.getfloat('inter.en.offset')
        inter = {"start":st, "end":en}

        st = energyL2 + section.getfloat('post.st.offset')
        en = section.getfloat('endenergy')
        post = {"start":st, "end":en}

        return [pre, inter, post]

    def get_rangeL3(self, elem):
        section = self.cfg[elem]
        st = section.getfloat('L3st')
        en = section.getfloat('L3en')
        return {"start": st, "end": en}

    def get_rangeL2(self, elem):
        section = self.cfg[elem]
        st = section.getfloat('L2st')
        en = section.getfloat('L2en')
        return {"start": st, "end": en}

    def get_ratio(self, elem):
        section = self.cfg[elem]
        r = Fraction(section.get('ratio'))
        return float(r)

    def get_holes(self, elem):
        return self.cfg.getfloat(elem, 'holes')

    def get_rulesparam(self, elem):
        section = self.cfg[elem]
        return [section.getfloat(s) for s in ['holes', 'offset.p', 'offset.q', 'offset.r', 'offset.r0']]


    def get_energyL3(self, elem):
        return self.cfg.getfloat(elem, 'energyL3')


    def get_energyL2(self, elem):
        return self.cfg.getfloat(elem, 'energyL2')


def get_nearest_value(data, num):
    """
    Abstract: return the nearest value in the list
    @param data: data list
    @param num: target value
    @return the nearest value index
    """
    return data[get_nearest_value_index]


def get_nearest_value_index(data, num):
    """
    Abstract: return the nearest value in the list
    @param data: data list
    @param num: target value
    @return the nearest value
    """
    return np.abs(np.asarray(data) - num).argmin()


def main(filenames, configfile):
    """ main routine for reading and analyzing XAS data """

    cfg = Config(configfile)

    for filename in filenames:
        with open(filename) as f:
            elem = f.readline().rstrip('\r\n')
            datafile = [line.rstrip('\r\n') for line in f]

            xmcd = Xmcd(cfg, datafile, elem)
            xmcd.calc()
            xmcd.calc_moment()
            xmcd.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read Xas data.')
    parser.add_argument('filename', metavar='filename', nargs='+',
                        help='a filename of XAS datafile')
    parser.add_argument('-c', dest='opt_c', action='store',
                        help='a config file')
    args = parser.parse_args()
    main(args.filename, args.opt_c)
