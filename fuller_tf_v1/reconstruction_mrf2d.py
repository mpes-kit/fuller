#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# Reconstruction object
class ReconstructionMRF2d(object):
    def __init__(self, k, E, I=None, E0=None, sigma=0.1):
        """
        Initialize object
        :param k: Momentum as numpy vector
        :param E: Energy as numpy vector
        :param I: Measured intensity wrt momentum (rows) and energy (columns), generated if None
        :param E0: Initial guess for band structure energy values, if None mean of E is taken
        :param sigma: Standard deviation of neighboring energies
        """
        
        self.k = k.copy()
        self.E = E.copy()
        self.kk, self.EE = np.meshgrid(self.k, self.E)
        self.I = I
        self.sigma = sigma
        self.sigmaGenerate = 0.1
        
        # Generate I if needed
        if I is None:
            self.generateI()
            
        # Initialize band structure
        if E0 is None:
            self.indEb = np.ones_like(k, np.int) * int(E.size / 2)
        else:
            EE, EE0 = np.meshgrid(E, E0)
            self.indEb = np.argmin(np.abs(EE - EE0), 1)
            
        # Initialize change of log likelihood
        self.deltaLogP = np.array([0.])
        
        
    def generateI(self):
        """
        Generate intensity with made up band structure
        """
        
        self.I = (0.95 * self.kk ** 2 + 0.05) / (1 + ((self.EE - self.bandStructGen(self.kk)) / 0.25) ** 2) + np.random.normal(0, self.sigmaGenerate, size=self.kk.shape)
        self.I = np.maximum(self.I, 0.1)
    
    
    def bandStructGen(self, k):
        """
        Function of band if bands are generated
        :param k: Vector or matrix of momenta
        :return: Energy values for each momentum
        """
        
        return k ** 3
        
        
    def iter(self, num=1):
        """
        Iterate band structure reconstruction process
        :param num: Number of iterations
        """
        
        # Do iterations
        deltaLogP = np.zeros(num)
        indList = np.random.choice(self.k.size, num)
        for i, ind in enumerate(indList):
            if ind == 0:
                Ebm = self.E[self.indEb[1]]
                Ebp = Ebm
            elif ind == (self.k.size - 1):
                Ebm = self.E[self.indEb[ind - 1]]
                Ebp = Ebm
            else:
                Ebm = self.E[self.indEb[ind - 1]]
                Ebp = self.E[self.indEb[ind + 1]]
            logP = np.log(self.I[:, ind]) - ((self.E - Ebm) ** 2 + (self.E - Ebp) ** 2) / 2 / self.sigma
            indMax = np.argmax(logP)
            deltaLogP[i] = logP[self.indEb[ind]] - logP[indMax]
            self.indEb[ind] = indMax
        
        # Update delta log likelihood
        self.deltaLogP = np.append(self.deltaLogP, np.cumsum(deltaLogP) + self.deltaLogP[-1])
        
        
    def getEb(self):
        """
        Get energy values of the electronic band
        :return: energy values of the electronic band
        """
        
        return self.E[self.indEb]


    def plotI(self):
        """
        Plot the intensity against k and E
        """
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(self.kk, self.EE, self.I)
    
    
    def plotBands(self, groundTruth=True):
        """
        Plot reconstructed electronic band structure
        :param groundTruth: Flag whether to plot true band from which data got generated
        """
        
        plt.figure()
        plt.plot(self.k, self.getEb(), marker='.')
        plt.plot(self.k, self.bandStructGen(self.k))
        plt.show()
    
    
    def plotLogP(self):
        """
        Plot the change of the log likelihood
        """
        
        plt.figure()
        plt.plot(self.deltaLogP)