#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probability generating function of a branching process on metapopulation network for early epidemic forecast.
This module contains a class derived from BasePGF.

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

from .base_pgf import *

class SIRPGF(BasePGF):
    """SIRPGF. Basic PGF encoding the state of an agent-based system under SIR compartment structure."""

    def __init__(self, umat, mmat, infectious_period, infection='poisson',
                 nb_microsteps=1, cumulant=False, umap=None, mmap=None, **kwargs):
        """__init__.

        If umat (mmat) is a list of ndarrays or matrices and umap (mmap) is not None, then
        it is assumed that the contact matrix (mobility matrix) is time-varying.

        Parameters
        ----------
        umat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Contact matrix.
        mmat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Mobility matrix.
        infectious_period : float
            Mean time spent in the infectious state [days].
        infection :
            Type of offspring distribution for infectious agents.
        cumulant : bool
            If true, turn the generating function into the cumulant generating function.
        umap : function
            Function mapping time to index for umat
        mmap : function
            Function mapping time to index for mmat
        """
        super().__init__(umat, mmat, infection, nb_microsteps, cumulant, umap, mmap,**kwargs)
        #rescale the periods on the scale of microsteps
        self.infectious_period = infectious_period*nb_microsteps

    def get_initial_state_vars(self, value):
        vec = np.zeros(self.nb_types, dtype=complex)+value
        state_vars = {'infectious': vec.copy(),
                      'cumulative infectious': vec.copy(),
                      'cumulative importation': vec.copy()}

        return state_vars

    def reaction_phase(self, state_vars):
        cumulative_infectious = state_vars['cumulative infectious']
        infectious = state_vars['infectious']

        #new infections from infectious ; infectious advance state
        state_vars['infectious'] =  self.G(infectious*cumulative_infectious)*self.B(infectious,1,1./self.infectious_period)


    def mobility_phase(self, state_vars):
        cumulative_importation = state_vars['cumulative importation']
        infectious = state_vars['infectious']

        #infectious move
        state_vars['infectious'] = self.mmat @ (cumulative_importation*infectious) + self.mvec*infectious


    def add_initial_conditions(self, idx, weight=None, nb_infectious=0, label=None, **kwargs):
        """add_initial_condition.

        Parameters
        ----------
        idx : int or array_like
            indices for the categories, e.g., if we want to distribute the probability across age and
            locations.
        weight : array_like
            probability associated to each category if idx is array_like. Uniform by default. Does nothing if
            idx is an integer.
        nb_infectious : int or array_like
            nb_infectious for each category
        """
        if label is None:
            label = len(self.Psi0)
        if isinstance(idx, (np.ndarray, list)):
            if len(weight) != len(idx) or not np.isclose(1.,sum(weight)):
                raise ValueError("weight ill-defined")
            else:
                self.Psi0[label] = lambda state_vars:\
                        np.sum(weight*state_vars["infectious"][idx]*state_vars["cumulative infectious"][idx])**nb_infectious
        else:
            self.Psi0[label] = lambda state_vars: \
                    (state_vars["infectious"][idx]*state_vars["cumulative infectious"][idx])**nb_infectious
