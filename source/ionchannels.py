"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

import numpy as np
import sympy as sp
import scipy.optimize

import math
import copy

def vtrap(x, y):
    if np.abs(x/y) < 1e-6:
        trap = y*(1. - x/(y*2.))
    else:
        trap = x/(np.exp(x/y) - 1.)
    return trap


## Expansion coefficients ##############################################
def leastsquaresfit(fun, gridV, gridx, V0=-65, x0=0.5):
    gridV = gridV - V0 * np.ones(gridx.shape)
    gridx = gridx - x0 * np.ones(gridx.shape)
    funflat = fun.flatten()[:,np.newaxis]
    phi = np.zeros((len(funflat), 4))
    ind = 0
    from itertools import product
    for ind1, ind2 in product(range(gridx.shape[0]), range(gridx.shape[1])):
        phi[ind,:] = np.array([gridx[ind1, ind2], gridV[ind1, ind2], 
                        gridx[ind1, ind2]*gridV[ind1, ind2], gridV[ind1, ind2]**2])
        ind += 1
    mat1 = np.dot(np.transpose(phi), phi)
    mat2 = np.dot(np.transpose(phi), funflat)
    w = np.dot(np.linalg.inv(mat1), mat2)
    return w, w[0] * gridx + w[1] * gridV + \
                    w[2] * gridx*gridV + w[3] * gridV**2


def calc_statevar_expansion(order=2, plot=False, **kwargs): # fstatevar, varinf, spV, statevars, E_eq, method='expansion'):
    if order==0:
        return {'0': np.zeros(len(kwargs['statevars']))}
    elif 'fstatevar' in kwargs.keys():
        # derivatives of rate equations
        fstatevar = kwargs['fstatevar']; varinf = kwargs['varinf']; spV = kwargs['spV']
        statevars = kwargs['statevars']; E_eq = kwargs['E_eq']
        dfstatevars = []
        ddfstatevars = []
        for ind, var in enumerate(statevars):
            dfstatevars.append([sp.diff(fstatevar[ind], spV, 1), \
                                sp.diff(fstatevar[ind], var, 1)])
            ddfstatevars.append([sp.diff(fstatevar[ind], spV, 2), \
                                sp.diff(sp.diff(fstatevar[ind], var, 1), spV, 1)])
            for ind2, var2 in enumerate(statevars):
                dfstatevars[-1][0] = dfstatevars[-1][0].subs(var2, varinf[ind2])
                dfstatevars[-1][1] = dfstatevars[-1][1].subs(var2, varinf[ind2])
                ddfstatevars[-1][0] = ddfstatevars[-1][0].subs(var2, varinf[ind2])
                ddfstatevars[-1][1] = ddfstatevars[-1][1].subs(var2, varinf[ind2])
            dfstatevars[-1][0] = float(dfstatevars[-1][0].subs(spV, E_eq))
            dfstatevars[-1][1] = float(dfstatevars[-1][1].subs(spV, E_eq))
            ddfstatevars[-1][0] = float(ddfstatevars[-1][0].subs(spV, E_eq))
            ddfstatevars[-1][1] = float(ddfstatevars[-1][1].subs(spV, E_eq))
            
        dfstatevars0 = np.array(dfstatevars, dtype=complex)
        ddfstatevars0 = np.array(ddfstatevars, dtype=complex)
        
        return {'0': np.zeros(len(statevars)), '1': dfstatevars0, '2': ddfstatevars0}
    elif 'fun_alpha' in kwargs.keys(): 
        statevars = kwargs['statevars']
        statevars1 = []
        statevars2 = []
        Vmin = -68.
        Vmax = -55.
        E_eq = kwargs['E_eq']
        
        if plot:
            import matplotlib.pyplot as pl
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            pl.figure(figsize=(14,len(statevars)*5))
        
        for ind, var in enumerate(statevars):
            xinf = kwargs['fun_alpha'][str(var)](E_eq) / (kwargs['fun_alpha'][str(var)](E_eq) + kwargs['fun_beta'][str(var)](E_eq))
            xinf0 = kwargs['fun_alpha'][str(var)](Vmin) / (kwargs['fun_alpha'][str(var)](Vmin) + kwargs['fun_beta'][str(var)](Vmin))
            xinf1 = kwargs['fun_alpha'][str(var)](Vmax) / (kwargs['fun_alpha'][str(var)](Vmax) + kwargs['fun_beta'][str(var)](Vmax))
            if xinf0 < xinf1: xmin = xinf0; xmax = xinf1
            else: xmin = xinf1; xmax = xinf0
            
            V = np.arange(Vmin, Vmax, 0.6)
            x = np.arange(xmin, xmax, 0.02)
            VV, xx = np.meshgrid(V, x)
            
            aV = kwargs['fun_alpha'][str(var)](VV) *1.0e3   #1/s
            bV = kwargs['fun_beta'][str(var)](VV) *1.0e3    #1/s
            func = - aV + (aV + bV) * xx
            
            w, func_fit = leastsquaresfit(func, VV, xx, V0=E_eq, x0=xinf)
            statevars1.append([w[1], w[0]])
            statevars2.append([w[3], w[2]])
            
            if plot:
                ax = pl.subplot2grid((len(statevars),2), (ind,0), projection='3d')
                ax.plot_surface(VV, xx, func, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
                ax.plot_wireframe(VV, xx, func_fit, color='r', rstride=1, cstride=1)
                ax.set_xlabel('V (mV)')
                ax.set_ylabel(str(var))
                ax.set_zlabel('f(V,' + str(var) + ')')
                
                ax2 = pl.subplot2grid((len(statevars),2),(ind,1))
                colors = ['g','b','r','c','m']
                for i, ind in enumerate([0, int((len(x)-1)/4.), 2*int((len(x)-1)/4.), 3*int((len(x)-1)/4.), len(x)-1]):
                    ax2.plot(V, func[ind,:], ls='-', c=colors[i], label=str(var)+' = %.3f' % x[ind])
                    ax2.plot(V, func_fit[ind,:], ls='-.', c=colors[i])
                ax2.set_xlabel('V (mV)')
                ax2.set_ylabel('f(V,' + str(var) + ')')
                ax2.legend(loc=1)
        
        if plot:
            pl.show()
        
        statevars1 = np.array(statevars1)
        statevars2 = np.array(statevars2)
        
        return {'0': np.zeros(len(statevars)), '1': statevars1, '2': statevars2}
    elif 'fun_inf' in kwargs.keys(): 
        statevars = kwargs['statevars']
        statevars1 = []
        statevars2 = []
        Vmin = -68.
        Vmax = -55.
        E_eq = kwargs['E_eq']
        
        if plot:
            import matplotlib.pyplot as pl
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            pl.figure(figsize=(14,len(statevars)*5))
        
        for ind, var in enumerate(statevars):
            xinf = kwargs['fun_inf'][str(var)](E_eq)
            xinf0 = kwargs['fun_inf'][str(var)](Vmin)
            xinf1 = kwargs['fun_inf'][str(var)](Vmax)
            if xinf0 < xinf1: xmin = xinf0; xmax = xinf1
            else: xmin = xinf1; xmax = xinf0
            
            V = np.arange(Vmin, Vmax, 0.6)
            x = np.arange(xmin, xmax, 0.02)
            VV, xx = np.meshgrid(V, x)
            
            xinfV = kwargs['fun_inf'][str(var)](VV)
            tauxV = kwargs['fun_tau'][str(var)](VV) *1.0e-3    #s
            func = (xx - xinfV) / tauxV
            
            w, func_fit = leastsquaresfit(func, VV, xx, V0=E_eq, x0=xinf)
            statevars1.append([w[1], w[0]])
            statevars2.append([w[3], w[2]])
            
            if plot:
                ax = pl.subplot2grid((len(statevars),2), (ind,0), projection='3d')
                ax.plot_surface(VV, xx, func, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
                ax.plot_wireframe(VV, xx, func_fit, color='r', rstride=1, cstride=1)
                ax.set_xlabel('V (mV)')
                ax.set_ylabel(str(var))
                ax.set_zlabel('f(V,' + str(var) + ')')
                
                ax2 = pl.subplot2grid((len(statevars),2),(ind,1))
                colors = ['g','b','r','c','m']
                for i, ind in enumerate([0, int((len(x)-1)/4.), 2*int((len(x)-1)/4.), 3*int((len(x)-1)/4.), len(x)-1]):
                    ax2.plot(V, func[ind,:], ls='-', c=colors[i], label=str(var)+' = %.3f' % x[ind])
                    ax2.plot(V, func_fit[ind,:], ls='-.', c=colors[i])
                ax2.set_xlabel('V (mV)')
                ax2.set_ylabel('f(V,' + str(var) + ')')
                ax2.legend(loc=1)
        
        if plot:
            pl.show()
        
        statevars1 = np.array(statevars1)
        statevars2 = np.array(statevars2)
        
        return {'0': np.zeros(len(statevars)), '1': statevars1, '2': statevars2}
    elif 'statevar_fun_vals' in kwargs.keys():
        statevars = kwargs['statevars']
        statevars1 = []
        statevars2 = []
        E_eq = kwargs['E_eq']
        
        if plot:
            import matplotlib.pyplot as pl
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            pl.figure(figsize=(14,len(statevars)*5))
            
        for ind, var in enumerate(statevars):
            xinf = kwargs['varinf'][ind]
            
            func = kwargs['statevar_fun_vals'][ind]
            xx = kwargs['xxs'][ind]
            VV = kwargs['VVs'][ind]
            
            w, func_fit = leastsquaresfit(func, VV, xx, V0=E_eq, x0=xinf)
            statevars1.append([w[1], w[0]])
            statevars2.append([w[3], w[2]])
            
            if plot:
                ax = pl.subplot2grid((len(statevars),2), (ind,0), projection='3d')
                ax.plot_surface(VV, xx, func, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
                ax.plot_wireframe(VV, xx, func_fit, color='r', rstride=1, cstride=1)
                ax.set_xlabel('V (mV)')
                ax.set_ylabel(str(var))
                ax.set_zlabel('f(V,' + str(var) + ')')
                
                ax2 = pl.subplot2grid((len(statevars),2),(ind,1))
                colors = ['g','b','r','c','m']
                for i, ind in enumerate([0, int((len(x)-1)/4.), 2*int((len(x)-1)/4.), 3*int((len(x)-1)/4.), len(x)-1]):
                    ax2.plot(V, func[ind,:], ls='-', c=colors[i], label=str(var)+' = %.3f' % x[ind])
                    ax2.plot(V, func_fit[ind,:], ls='-.', c=colors[i])
                ax2.set_xlabel('V (mV)')
                ax2.set_ylabel('f(V,' + str(var) + ')')
                ax2.legend(loc=1)
                
        if plot:
            pl.show()

        statevars1 = np.array(statevars1)
        statevars2 = np.array(statevars2)
        
        return {'0': np.zeros(len(statevars)), '1': statevars1, '2': statevars2}
    else:
        print 'Error: invalid method'
        exit(1)


def calc_membrane_expansion(fv, varinf, spV, statevars, E_eq, order=2):
    if fv == 0:
        return {'0': 0}
    else:
        returndict = {}
        
        fv0 = fv
        for ind, var in enumerate(statevars):
            fv0 = fv0.subs(var, varinf[ind])
        fv0 = float(fv0.subs(spV, E_eq))
        returndict['0'] = fv0
        
        if order==0:
            return returndict
        else:
            dfv = [sp.diff(fv, spV, 1)]
            for ind, var in enumerate(statevars):
                dfv[0] = dfv[0].subs(var, varinf[ind])
            dfv[0] = float(dfv[0].subs(spV, E_eq))

            ddfv = []
            ddfv_varvar = np.empty((len(statevars), len(statevars)), dtype=object)
            for ind, var in enumerate(statevars):
                dfv.append(sp.diff(fv, var, 1))
                ddfv.append(sp.diff(sp.diff(fv, spV, 1), var, 1))
                for ind2, var2 in enumerate(statevars):
                    dfv[-1] = dfv[-1].subs(var2, varinf[ind2])
                    ddfv[-1] = ddfv[-1].subs(var2, varinf[ind2])
                dfv[-1] = float(dfv[-1].subs(spV, E_eq))
                ddfv[-1] = float(ddfv[-1].subs(spV, E_eq))
                
                for ind3, var3 in enumerate(statevars):
                    ddfv_varvar[ind, ind3] = sp.diff(sp.diff(fv, var, 1), var3, 1)
                    for ind4, var4 in enumerate(statevars):
                        ddfv_varvar[ind, ind3] = ddfv_varvar[ind, ind3].subs(var4, varinf[ind4])
                    ddfv_varvar[ind, ind3] = float(ddfv_varvar[ind, ind3].subs(spV, E_eq))
            
            dfv0 = np.array(dfv, dtype=complex)
            ddfv0 = np.array(ddfv, dtype=complex)
            ddfv_varvar0 = np.array(ddfv_varvar, dtype=complex)
        
            return {'0': fv0, '1': dfv0, '2vx': ddfv0, '2xy': ddfv_varvar0}
    

## generic ion channel class ###########################################
class ionChannel:
    '''
    Super class for all different ion channel types. 
    
    The constructer should have the following keyword arguments:
    [g]: float, the maximal conductance (uS/cm^2)
    [e]: float, the reversal (mV)
    [V0]: float, the equilibrium potential of the neuron (mV)
    [nonlinear]: boolean, whether to return the nonlinear part of the current
    [calc]: boolean that specifies if the ion channels is to be use for impedance
        calculations
    
    The constructor should define the following variables:
    
    [self.g]: float, max conductance (uS/cm^2)
    [self.e]: float, reversal (mV)
    
    [self.statevar]: 2d np array of floats, values of the state variables
    [self.powers]: 2d np array of floats, powers of the corresponding state variables
    [self.factors]: 1d numpy array of factors to multiply terms in sum
    
        Suppose self.statevar = [[a11,a12],[a21,a22]], self.powers = [[n11,n12],[n21,n22]]
        and self.factors = [f1, f2]. Then the corresponding transmembrane current is
        I = g * (f1 * a11^n11 * a12^n12 + f2 * a21^n21 * a22^n22) * (V - e)
    
    [self.nonlinear]: boolean, whether only the nonlinear part of the current is
        considered by subtracting the linear part
    [self.calc]: boolean, whether the ion channels serve to compute kernels
    
    [self.V0]: resting potential of the neuron
    
    When self.nonlinear or self.calc are True, we need to set some sympy variables to 
    perform computations
    [self.spV]: sympy variable for voltage
    [self.statevars]: sympy variables for state variables
    [self.fun]: sympy expression for function of statevariables that in the membrane current
    [self.fstatevar]: sympy expression for statevariable functions
    [self.varinf]: expressions for the limits of the variables
    [self.tau]: expressions for the timescales of the state variables
    
    When self.nonlinear is True, the function set [self.lincoeffI] and [self.lincoeffstatevar]
    by:
        self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
    and 
    [self.linstatevar]: array with the statevariables of the linearized component of the current
    
    When self.calc is True, set
    [self.fv]: the coefficients of the expansion of the current
    and compute the expansion coefficients by using self.set_expansion_coeffs().
    '''
        
    def advance(self, V, dt, conc={}):
        '''
        advance the state variables of the ion channel
        '''
        if self.nonlinear:
            self.advance_linear(V, dt)
        else:
            self.statevar += dt * self.fun_statevar(V, self.statevar, conc=conc)
            
    def advance_linear(self, V, dt, conc={}):
        '''
        advance the state variables in a linear fashion
        '''
        x_nl = self.fun_statevar(V, self.statevar, conc=conc) - \
                            self.lincoeffstatevar[0] * (V-self.V0) - \
                            self.lincoeffstatevar[1] * (self.statevar - self.linstatevar)
        self.Istatevar = np.exp(self.lincoeffstatevar[1]*dt) * self.Istatevar - \
                    (1. - np.exp(self.lincoeffstatevar[1]*dt)) / \
                    self.lincoeffstatevar[1] * x_nl
        self.Vstatevar = np.exp(self.lincoeffstatevar[1]*dt) * self.Vstatevar - \
                    (1. - np.exp(self.lincoeffstatevar[1]*dt)) / \
                    self.lincoeffstatevar[1] * (V - self.V0)
        self.statevar = self.lincoeffstatevar[0] * self.Vstatevar + \
                    self.Istatevar + \
                    self.linstatevar

    def getCurrent(self, V):
        '''
        returns the transmembrane current in nA, if self.nonlinear is True,
        returns the nonlinear part of the current
        '''
        I = self.get_full_current(V)
        I_nl = self.get_linear_current(V)
        return I - I_nl


    def get_full_current(self, V):
        '''
        Returns the full current of the ionchannel in nA
        '''
        return - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None]) * (V - self.e)
        
    def get_linear_current(self, V):
        '''
        returns the linear part of the transmembrane current in nA
        '''
        if self.nonlinear:
            I = - self.g * (
                np.sum(self.factors * np.prod(self.linstatevar**self.powers, 1)[:,None]) * (self.V0 - self.e) + \
                np.sum((self.lincoeffI[1] * (self.statevar-self.linstatevar)).flatten()) * (self.V0 - self.e) + self.lincoeffI[0] * (V - self.V0) - \
                np.sum((self.lincoeffI[1] * self.Istatevar).flatten()) * (self.V0 - self.e) )
        else:
            return 0.#self.g0 * (V - self.e)
        return I

    def get_current_part(self):
        g1, c1 = self.get_full_current_part()
        g2, c2 = self.get_linear_current_part()
        return g1-g2, c1-c2

    def get_linear_current_part(self):
        if self.nonlinear:
            g = - self.g*self.lincoeffI[0]
            c = - self.g * (
                np.sum(self.factors * np.prod(self.linstatevar**self.powers, 1)[:,None]) * (self.V0 - self.e) + \
                np.sum((self.lincoeffI[1] * (self.statevar-self.linstatevar)).flatten()) * (self.V0 - self.e) - \
                np.sum((self.lincoeffI[1] * self.Istatevar).flatten()) * (self.V0 - self.e) - self.lincoeffI[0] * self.V0 )
        else:
            g = 0.#self.g0
            c = 0.#-self.g0 * (self.e - self.V0)
        return g, c

    def get_full_current_part(self):
        geff = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        return - geff, geff * (self.e - self.V0)
        
    def fun_statevar(self, V): return 0.

    # def set_I0(self):
    #     svar = self.fun_statevar(self.V0, self.statevar, conc=self.conc0)
        # self.I0 = - self.g * np.sum(self.factors * np.prod(svar**self.powers, 1)[:,None]) * (self.V0 - self.e)
        
    def set_expansion_point(self, E_eq=-65.):
        self.E_eq = E_eq

    def set_expansion_coeffs(self, method='expansion', order=2):
        '''
        computes the expansion coefficients for the volterra expansion and for
        the Greens function formalism
        '''
        self.coeff_v = calc_membrane_expansion(self.fv, self.varinf.flatten(), self.spV, 
                                            self.statevars.flatten(), self.V0, order)
        # print 'coeff_v: ', self.coeff_v
        if method=='fit':
            # compute arrays with state variable values within the limits determined
            # by the voltage limits
            Vmin = -68.; Vmax = -54.
            funvals = []
            xxs = []
            VVs = []
            V = np.arange(Vmin, Vmax, 0.6)
            varinf = []
            for ind, var in enumerate(self.statevars):
                fun = sp.utilities.lambdify(self.spV, self.varinf[ind], "numpy")
                varinf.append(fun(self.V0))
                var0 = fun(Vmin); var1 = fun(Vmax)
                if var0 < var1: varmin = var0; varmax = var1
                else: varmin = var1; varmax = var0
                varvals = np.arange(varmin, varmax, 0.02)
                VV, xx = np.meshgrid(V, varvals)
                xxs.append(xx); VVs.append(VV)
                fun = sp.utilities.lambdify((self.spV, var), -self.fstatevar[ind]*1e3, "numpy") # timescale in seconds
                funvals.append(fun(VV, xx))
            # pass array of function values to fitting function
            self.coeff_statevar = calc_statevar_expansion(order, statevars=self.statevars, E_eq=self.V0,
                            statevar_fun_vals=funvals, VVs=VVs, xxs=xxs, varinf=varinf)
            #~ print 'coeff_statevar: ', self.coeff_statevar
        elif method=='expansion':
            self.coeff_statevar = calc_statevar_expansion(order, fstatevar=self.fstatevar.flatten() * 1e3, 
                            varinf=self.varinf.flatten(), spV=self.spV, statevars=self.statevars.flatten(), E_eq=self.V0)
            # print 'coeff_statevar: ', self.coeff_statevar
        else:
            print 'Error: invalid method'
            exit(1)

    def calc_offset(self, freqs=None):
        '''
        computes the channel current at equilibirum potential
        '''
        svinf = np.zeros(self.varinf.shape)
        for ind, var in np.ndenumerate(self.varinf):
            svinf[ind] = var.subs(self.spV, self.V0)
        if freqs==None:
            return - self.g * np.sum(self.factors * np.prod(svinf[ind]**self.powers, 1)[:,None]) * (self.V0 - self.e)
        else:
            return - self.g * np.sum(self.factors * np.prod(svinf[ind]**self.powers, 1)[:,None]) * (self.V0 - self.e) * np.ones(freqs.shape, dtype=complex)
        

    def calc_passive(self, freqs, conc0=[], conc_coeff=[]):
        return - self.g0 * np.ones(freqs.shape, dtype=complex)

    def calc_linear(self, freqs, conc0=[], conc_coeff=[]):
        '''
        Computes contribution of ion channel to membrane impedance
        '''
        coeffI, coeffstatevar = self.compute_lincoeff()
        # convert units of coeffstatevar to 1/s (instead of 1/ms)
        for ind, var in np.ndenumerate(self.statevars):
            coeffstatevar[0][ind] *= 1e3
            coeffstatevar[1][ind] *= 1e3
        returndict = {}
        imp = coeffI[0] * np.ones(freqs.shape, dtype=complex)
        for ind, var in np.ndenumerate(self.statevars):
            # response function for state variable given the voltage
            returndict[var] = coeffstatevar[0][ind] / (freqs - coeffstatevar[1][ind])
            # response function
            # contribution of state variable to membrane impedance
            imp += coeffI[1][ind] * (self.V0 - self.e) * returndict[var]
        returndict['V'] = self.g * imp
        return returndict
        
    def compute_lincoeff(self, conc0=[]):
        '''
        computes coefficients for linear simulation
        '''
        # coefficients for computing current
        fun = self.fun #statevars**self.powers
        coeff = np.zeros(self.statevar.shape, dtype=object)
        # differentiate
        for ind, var in np.ndenumerate(self.statevars):
            coeff[ind] = sp.diff(fun, var,1)
        # substitute
        for ind, var in np.ndenumerate(self.statevars):
            fun = fun.subs(var, self.varinf[ind])
            for ind2, coe in np.ndenumerate(coeff):
                coeff[ind2] = coe.subs(var, self.varinf[ind])
        fun = fun.subs(self.spV, self.V0)
        for ind, coe in np.ndenumerate(coeff):
            coeff[ind] = coe.subs(self.spV, self.V0)
        coeffI = [np.float64(fun), coeff.astype(float)]
        
        # coefficients for state variable equations
        dfdv = np.zeros(self.statevar.shape, dtype=object)
        dfdx = np.zeros(self.statevar.shape, dtype=object)
        dfdc = [np.zeros(self.statevar.shape, dtype=object) for c in self.conc]
        # differentiate
        for ind, var in np.ndenumerate(self.statevars):
            dfdv[ind] = sp.diff(self.fstatevar[ind], self.spV, 1)
            dfdx[ind] = sp.diff(self.fstatevar[ind], var, 1)
            for ind2, c in enumerate(self.concentrations):
                dfdc[ind2][ind] = sp.diff(self.fstatevar[ind], c, 1)
        # substitute state variables by their functions
        for ind, var in np.ndenumerate(self.statevars):
            dfdv[ind] = dfdv[ind].subs(var, self.varinf[ind])
            for ind2, c in enumerate(self.concentrations):
                dfdc[ind2][ind].subs(var, self.varinf[ind])
        # substitute voltage by its value
        for ind, var in np.ndenumerate(self.statevars):
            dfdv[ind] = dfdv[ind].subs(self.spV, self.V0)
            dfdx[ind] = dfdx[ind].subs(self.spV, self.V0)
            for ind2, c in enumerate(self.concentrations):
                dfdc[ind2][ind].subs(self.spV, self.V0)
        # substitute the concentrations by their equilibrium values (conc0)
        for ind, conc in self.concentrations:
            for ind2, var in np.ndenumerate(self.statevars):
                dfdv[ind2] = dfdv[ind2].subs(self.conc[ind], conc0[ind])
                dfdx[ind2] = dfdx[ind2].subs(self.conc[ind], conc0[ind])
                for ind3, c in enumerate(self.concentrations):
                    dfdc[ind3][ind2] = dfdc[ind3][ind2].subs(self.conc[ind], conc0[ind])

        coeffstatevar = [dfdv.astype(float), dfdx.astype(float), [f.astype(float) for f in dfdc]]
        # print 'coeffstatevar: ', coeffstatevar 
        
        return coeffI, coeffstatevar
       
    def write_mod_file(self):
        '''
        Writes a modfile of the ion channel for simulations with neuron
        '''
        f = open('../mech/I' + self.__class__.__name__ + '.mod', 'w')
        
        f.write(': This mod file is automaticaly generated by the ionc.write_mode_file() function in /source/ionchannels.py \n\n')
        
        f.write('NEURON {\n')
        f.write('    SUFFIX I' + self.__class__.__name__ + '\n')
        if self.ion == '':
            f.write('    NONSPECIFIC_CURRENT i' + '\n')
        else:
            # f.write('    USEION ' + self.ion + ' READ e' + self.ion + ' WRITE i' + self.ion + '\n')
            f.write('    USEION ' + self.ion + ' WRITE i' + self.ion + '\n')
        if len(self.concentrations) > 0:
            for concstring in self.concentrations:
                f.write('    USEION ' + concstring + ' READ ' + concstring + 'i' + '\n')
        f.write('    RANGE  g, e' + '\n')
        varstring = 'var0inf'
        taustring = 'tau0'
        for ind in range(len(self.varinf.flatten()[1:])):
            varstring += ', var' + str(ind+1) + 'inf'
            taustring += ', tau' + str(ind+1)
        f.write('    GLOBAL ' + varstring + ', ' + taustring + '\n')
        f.write('    THREADSAFE' + '\n')
        f.write('}\n\n')
        
        f.write('PARAMETER {\n')
        f.write('    g = ' + str(self.g*1e-6) + ' (S/cm2)' + '\n')
        f.write('    e = ' + str(self.e) + ' (mV)' + '\n')
        for ion in self.concentrations:
            f.write('    ' + ion + 'i (mM)' + '\n')
        f.write('}\n\n')
        
        f.write('UNITS {\n')
        f.write('    (mA) = (milliamp)' + '\n')
        f.write('    (mV) = (millivolt)' + '\n')
        f.write('    (mM) = (milli/liter)' + '\n')
        f.write('}\n\n')
        
        f.write('ASSIGNED {\n')
        f.write('    i' + self.ion + ' (mA/cm2)' + '\n')
        # if self.ion != '':
        #     f.write('    e' + self.ion + ' (mV)' + '\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + 'inf' + '\n')
            f.write('    tau' + str(ind) + ' (ms)' + '\n')
        f.write('    v (mV)' + '\n')
        f.write('}\n\n')
        
        f.write('STATE {\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + '\n')
        f.write('}\n\n')
        
        f.write('BREAKPOINT {\n')
        f.write('    SOLVE states METHOD cnexp' + '\n')
        calcstring = '    i' + self.ion + ' = g * ('
        l = 0
        for i in range(self.statevar.shape[0]):
            for j in range(self.statevar.shape[1]):
                for k in range(self.powers[i,j]):
                    calcstring += ' var' + str(l) + ' *'
                l += 1
            calcstring += str(self.factors[i,0])
            if i < self.statevar.shape[0] - 1:
                calcstring += ' + '
        # calcstring += ') * (v - e' + self.ion + ')'
        calcstring += ') * (v - e)'
        f.write(calcstring + '\n')
        f.write('}\n\n')
        
        concstring = ''
        for ion in self.concentrations:
            concstring += ', ' + ion + 'i'
        f.write('INITIAL {\n')
        f.write('    rates(v' + concstring + ')' + '\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + ' = var' + str(ind) + 'inf' + '\n')
        f.write('}\n\n')
        
        f.write('DERIVATIVE states {\n')
        f.write('    rates(v' + concstring + ')' + '\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + '\' = (var' + str(ind) + 'inf - var' + str(ind) + ') / tau' + str(ind) + '\n')
        f.write('}\n\n')
        
        concstring = ''
        for ion in self.concentrations:
            concstring += ', ' + ion
        f.write('PROCEDURE rates(v' + concstring + ') {\n')
        for ind, varinf in enumerate(self.varinf.flatten()):
            f.write('    var' + str(ind) + 'inf = ' + sp.printing.ccode(varinf) + '\n')
            f.write('    tau' + str(ind) + ' = ' + sp.printing.ccode(self.tau.flatten()[ind]) + '\n')
        f.write('}\n\n')
        
        f.close()

    def write_cpp_code(self):
        fcc = open('cython_code/ionchannels.cc', 'a')
        fh = open('cython_code/ionchannels.h', 'a')
        fstruct = open('cython_code/channelstruct.h', 'a')
        fh.write('\n')
        fcc.write('\n')

        fh.write('class ' + self.__class__.__name__ + ': public ion_channel{' + '\n')
        fh.write('public:' + '\n')
        fh.write('    void calc_fun_statevar(double v);' + '\n')
        fh.write('};' + '\n')

        fcc.write('void ' + self.__class__.__name__ + '::calc_fun_statevar(double v){' + '\n')
        for ind, varinf in np.ndenumerate(self.varinf):
            fcc.write('    ion_channel::svinf[' + str(ind[0]) + '][' + str(ind[1]) + '] = ' + sp.printing.ccode(varinf) + ';' + '\n')
            fcc.write('    ion_channel::taus[' + str(ind[0]) + '][' + str(ind[1]) + '] = ' + sp.printing.ccode(self.tau[ind]) + ';' + '\n')
        fcc.write('}' + '\n')

        fstruct.write('    ' + self.__class__.__name__ + ' ' + self.__class__.__name__ + '_;' + '\n')

        fh.close()
        fcc.close()
        fstruct.close()
    
    
class h(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-43., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        Hcn channel from (Bal and Oertel, 2000)
        '''
        self.ion = ''
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        self.tauf = 40. # ms
        self.taus = 300. # ms
        
        self.tau_array = np.array([[self.tauf], [self.taus]])
        
        self.varnames = np.array([['hf'], ['hs']])
        self.statevar = np.array([[1./(1.+np.exp((V0+82.)/7.))], [1./(1.+np.exp((V0+82.)/7.))]])
        self.powers = np.array([[1],[1]], dtype=int)
        self.factors = np.array([[1.-self.ratio], [self.ratio]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = (1.-self.ratio)*self.statevars[0,0] + self.ratio*self.statevars[1,0]
            
            self.varinf = np.array([[1./(1.+sp.exp((self.spV+82.)/7.))], [1./(1.+sp.exp((self.spV+82.)/7.))]])
            # make array of sympy floats to render it sympy compatible
            self.tau = np.zeros(self.tau_array.shape, dtype=object)
            for ind, tau in np.ndenumerate(self.tau_array):
                self.tau[ind] = sp.Float(self.tau_array[ind])
            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.],[0.]])
            self.Vstatevar = np.array([[0.],[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        return (1./(1.+np.exp((V+82.)/7.)) - sv) / self.tau_array


class h_HAY(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-45., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        Hcn channel from (Kole, Hallermann and Stuart, 2006)

        Used in (Hay, 2011)
        '''
        self.ion = ''
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        
        self.varnames = np.array([['m']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0))]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]
            
            spalpham = 0.001 * 6.43 * (self.spV + 154.9) / (sp.exp((self.spV + 154.9) / 11.9) - 1.)
            spbetam = 0.001 * 193. * sp.exp(self.spV / 33.1)

            self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
            # make array of sympy floats to render it sympy compatible
            self.tau = np.array([[1. / (spalpham + spbetam)]])

            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def alpham(self, V):
        return 0.001 * 6.43 * (V + 154.9) / (np.exp((V + 154.9) / 11.9) - 1.)

    def betam(self, V):
        return 0.001 * 193. * np.exp(V / 33.1)

    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        svinf = np.array([[am / (am + bm)]])
        taus = np.array([[1. / (am + bm)]])
        return (svinf - sv) / taus


class Na(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=6.3):
        '''
        Sodium channel from the HH model
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2

        self.q10 = 3.**((temp - 6.3) / 10.)
        # self.q10 = 1.
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0)), \
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0))]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])

        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham = .1 * -(self.spV+40.)/(sp.exp(-(self.spV+40.)/10.) - 1.)  #1/ms
            spbetam = 4. * sp.exp(-(self.spV+65.)/18.)  #1/ms
            spalphah = .07 * sp.exp(-(self.spV+65.)/20.)   #1/ms
            spbetah = 1. / (sp.exp(-(self.spV+35.)/10.) + 1.)   #1/ms
            
            self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
            self.tau = np.array([[1. / (self.q10*(spalpham + spbetam)), 1. / (self.q10*(spalphah + spbetah))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm), ah / (ah + bh)]])
        taus = np.array([[1. / (self.q10*(am + bm)), 1. / (self.q10*(ah + bh))]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        if type(V) is np.float64 or type(V) is float:
            return .1  * vtrap(-(V+40.),10.)
        else:
            return .1  * -(V+40.) / (np.exp(-(V+40.)/10.) - 1.) 
        
    def betam(self, V):
        return  4.   * np.exp(-(V+65.)/18.)
        
    def alphah(self, V):
        return .07 * np.exp(-(V+65.)/20.)
        
    def betah(self, V):
        return 1.   / (np.exp(-(V+35.)/10.) + 1.)


class Na_Branco(ionChannel):
    def __init__(self, g=0.120*1e6, e=50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.): 
        ''' sodium channel found in (Branco, 2011) code'''
        self.ion = 'na'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0)), \
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0))]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0

    def alpham(self, V):
        if type(V) is np.float64 or type(V) is float:
            return .1  * vtrap(-(V+40.),10.)
        else:
            return .1  * -(V+40.) / (np.exp(-(V+40.)/10.) - 1.) 
        
    def betam(self, V):
        return  4.   * np.exp(-(V+65.)/18.)
        
    def alphah(self, V):
        return .07 * np.exp(-(V+65.)/20.)
        
    def betah(self, V):
        return 1.   / (np.exp(-(V+35.)/10.) + 1.)

    def trap(self, V, p1, p2, p3): pass


class Na_p(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Derived by (Hay, 2011) from (Magistretti and Alonso, 1999)

        Used in (Hay, 2011)

        !!! Does not work !!!
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ 1. / (1. + np.exp(-(V0 + 52.6) / 4.6)) ,
                                    1. / (1. + np.exp( (V0 + 48.8) / 10.)) ]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham =   0.182   * (self.spV + 38. ) / (1. - sp.exp(-(self.spV + 38. ) / 6.  ))  #1/ms
            spbetam  = - 0.124   * (self.spV + 38. ) / (1. - sp.exp( (self.spV + 38. ) / 6.  ))  #1/ms
            spalphah = - 2.88e-6 * (self.spV + 17. ) / (1. - sp.exp( (self.spV + 17. ) / 4.63))   #1/ms
            spbetah  =   6.94e-6 * (self.spV + 64.4) / (1. - sp.exp(-(self.spV + 64.4) / 2.63))   #1/ms
            
            self.varinf = np.array([[   1. / (1. + sp.exp(-(self.spV + 52.6) / 4.6)) , 
                                        1. / (1. + sp.exp( (self.spV + 48.8) / 10.)) ]])
            self.tau = np.array([[(6./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        if type(V) == np.float64 or type(V) == float:
            if V == -38. or V == -17. or V == -64.4:
                V += 0.0001
        else:
            ind = np.where(V == -38. or V == -17. or V == -64.4)[0]
            V[ind] += 0.0001
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[1. / (1. + np.exp(-(V + 52.6) / 4.6)) , 
                           1. / (1. + np.exp( (V + 48.8) / 10.)) ]])
        taus = np.array([[  (6. / (am + bm)) / 2.95 , 
                            (1. / (ah + bh)) / 2.95 ]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return 0.182 * (V + 38.) / (1. - np.exp(-(V + 38.) / 6.))
        
    def betam(self, V):
        return - 0.124 * (V + 38.) / (1. - np.exp((V + 38.) / 6.))
        
    def alphah(self, V):
        return -2.88e-6 * (V + 17.) / (1. - np.exp((V + 17.) / 4.63))
        
    def betah(self, V):
        return  6.94e-6 * (V + 64.4) / (1. - np.exp(-(V + 64.4) / 2.63))


class Na_Ta(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ self.alpham(V0) / (self.alpham(V0) + self.betam(V0)) ,
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0)) ]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham =   0.182 * (self.spV + 38.) / (1. - sp.exp(-(self.spV + 38.) / 6.))  #1/ms
            spbetam  = - 0.124 * (self.spV + 38.) / (1. - sp.exp( (self.spV + 38.) / 6.))  #1/ms
            spalphah = - 0.015 * (self.spV + 66.) / (1. - sp.exp( (self.spV + 66.) / 6.))   #1/ms
            spbetah  =   0.015 * (self.spV + 66.) / (1. - sp.exp(-(self.spV + 66.) / 6.))  #1/ms
            
            self.varinf = np.array([[   spalpham / (spalpham + spbetam) , 
                                        spalphah / (spalphah + spbetah) ]])
            self.tau = np.array([[(1./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        if type(V) == np.float64 or type(V) == float:
            if V == -38. or V == -66.:
                V += 0.0001
        else:
            ind = np.where(V == -38. or V == -66.)[0]
            V[ind] += 0.0001
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm) , 
                           ah / (ah + bh) ]])
        taus = np.array([[  (1. / (am + bm)) / 2.95 , 
                            (1. / (ah + bh)) / 2.95 ]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return   0.182 * (V + 38.) / (1. - np.exp(-(V + 38.) / 6.))
        
    def betam(self, V):
        return - 0.124 * (V + 38.) / (1. - np.exp( (V + 38.) / 6.))
        
    def alphah(self, V):
        return - 0.015 * (V + 66.) / (1. - np.exp( (V + 66.) / 6.))
        
    def betah(self, V):
        return   0.015 * (V + 66.) / (1. - np.exp(-(V + 66.) / 6.))


class Na_Ta2(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        (Colbert and Pan, 2002) 

        Shifted by 6 mV from Na_Ta

        Used in (Hay, 2011)
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ self.alpham(V0) / (self.alpham(V0) + self.betam(V0)) ,
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0)) ]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham =   0.182 * (self.spV + 32.) / (1. - sp.exp(-(self.spV + 32.) / 6.))  #1/ms
            spbetam  = - 0.124 * (self.spV + 32.) / (1. - sp.exp( (self.spV + 32.) / 6.))  #1/ms
            spalphah = - 0.015 * (self.spV + 60.) / (1. - sp.exp( (self.spV + 60.) / 6.))   #1/ms
            spbetah  =   0.015 * (self.spV + 60.) / (1. - sp.exp(-(self.spV + 60.) / 6.))  #1/ms
            
            self.varinf = np.array([[   spalpham / (spalpham + spbetam) , 
                                        spalphah / (spalphah + spbetah) ]])
            self.tau = np.array([[(1./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        if type(V) == np.float64 or type(V) == float:
            if V == -32. or V == -60.:
                V += 0.0001
        else:
            ind = np.where(V == -32. or V == -60.)[0]
            V[ind] += 0.0001
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm) , 
                           ah / (ah + bh) ]])
        taus = np.array([[  (1. / (am + bm)) / 2.95 , 
                            (1. / (ah + bh)) / 2.95 ]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return   0.182 * (V + 32.) / (1. - np.exp(-(V + 32.) / 6.))
        
    def betam(self, V):
        return - 0.124 * (V + 32.) / (1. - np.exp( (V + 32.) / 6.))
        
    def alphah(self, V):
        return - 0.015 * (V + 60.) / (1. - np.exp( (V + 60.) / 6.))
        
    def betah(self, V):
        return   0.015 * (V + 60.) / (1. - np.exp(-(V + 60.) / 6.))


class K(ionChannel):
    def __init__(self,  g=0.036*1e6, e=-77., V0=-65, conc0=[], nonlinear=False, calc=False, temp=6.3):
        '''
        Potassium channel from HH model
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2

        self.q10 = 3.**((temp-6.3) / 10.)
        # self.q10 = 1.
        
        self.varnames = np.array([['n']])
        self.statevar = np.array([[self.alphan(V0) / (self.alphan(V0) + self.betan(V0))]])
        self.powers = np.array([[4]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**4
            
            spalphan = -0.01 * (self.spV + 55.) / (sp.exp(-(self.spV + 55.)/10.) - 1.)
            spbetan = .125* sp.exp(-(self.spV + 65.)/80.)
            
            self.varinf = np.array([[spalphan / (spalphan + spbetan)]])
            self.tau = np.array([[1. / (self.q10*(spalphan + spbetan))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        an = self.alphan(V); bn = self.betan(V)
        svinf = np.array([[an / (an + bn)]])
        taus = np.array([[1. / (self.q10*(an + bn))]])
        return (svinf - sv) / taus
        
    def alphan(self, V):
        if type(V) is np.float64 or type(V) is float:
            return .01 * vtrap(-(V+55.),10.)
        else:
            return .01 * -(V+55.) / (np.exp(-(V+55.)/10.) - 1)
    
    def betan(self, V): 
        return .125* np.exp(-(V+65.)/80.)


class Klva(ionChannel):
    def __init__(self, g=0.001*1e6, e=-106., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Low voltage activated potassium channel (Kv1) from (Mathews, 2010)
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.g = g #uS/cm^2
        self.e = e #mV
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[1./(1.+np.exp(-(V0+57.34)/11.7)),  0.73/(1.+np.exp((V0+67.)/6.16)) + 0.27]])
        self.powers = np.array([[4, 1]], dtype=int)
        self.factors = np.array([[1.]])
        
        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])

        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**4 * self.statevars[0,1]
            
            self.varinf = np.array([[1./(1.+sp.exp(-(self.spV+57.34)/11.7)), 0.73/(1.+sp.exp((self.spV+67.)/6.16)) + 0.27]])
            self.tau = np.array([[(21.5/(6.*sp.exp((self.spV+60.)/7.) + 24.*sp.exp(-(self.spV+60.)/50.6)) + 0.35), \
                                (170./(5.*sp.exp((self.spV+60.)/10.) + sp.exp(-(self.spV+70.)/8.)) + 10.7)]]) # ms
            # self.tau = np.array([[1.080549, 58.346879]])
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[1./(1.+np.exp(-(V+57.34)/11.7)),  0.73/(1.+np.exp((V+67.)/6.16)) + 0.27]])
        taus = np.array([[(21.5/(6.*np.exp((V+60.)/7.) + 24.*np.exp(-(V+60.)/50.6)) + 0.35), \
                            (170./(5.*np.exp((V+60.)/10.) + np.exp(-(V+70.)/8.)) + 10.7)]]) # ms
        # taus = np.array([[1.080549, \
        #                     58.346879]]) # ms
        return (svinf - sv) / taus


class m(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-80., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        M-type potassium current (Adams, 1982)

        Used in (Hay, 2011)

        !!! does not work when e > V0 !!!
        '''
        self.ion = 'k'
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        
        self.varnames = np.array([['m']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0))]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]
            
            spalpham = 3.3e-3 * sp.exp( 2.5 * 0.04 * (self.spV + 35.))
            spbetam = 3.3e-3 * sp.exp(-2.5 * 0.04 * (self.spV + 35.))

            self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
            # make array of sympy floats to render it sympy compatible
            self.tau = np.array([[(1. / (spalpham + spbetam)) / 2.95]])# 

            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def alpham(self, V):
        return 3.3e-3 * np.exp( 2.5 * 0.04 * (V + 35.))

    def betam(self, V):
        return 3.3e-3 * np.exp(-2.5 * 0.04 * (V + 35.))

    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        svinf = np.array([[am / (am + bm)]])
        taus = np.array([[(1. / (am + bm)) / 2.95]])# 
        return (svinf - sv) / taus


class Kv3_1(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-80., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        Shaw-related potassium channel (Rettig et al., 1992)

        Used in (Hay et al., 2011)
        '''
        self.ion = 'k'
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        
        self.varnames = np.array([['m']])
        self.statevar = np.array([[1. / (1. + np.exp(-(V0 - 18.7) / 9.7))]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]

            self.varinf = np.array([[ 1. / (1. + sp.exp(-(self.spV - 18.7) / 9.7)) ]])
            self.tau = np.array([[ 4. / (1. + sp.exp(-(self.spV + 46.56) / 44.14)) ]])

            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[ 1. / (1. + np.exp(-(V - 18.7) / 9.7)) ]])
        taus = np.array([[ 4. / (1. + np.exp(-(V + 46.56) / 44.14)) ]])
        return (svinf - sv) / taus


class Kpst(ionChannel):
    def __init__(self, g=0.001*1e6, e=-106., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Persistent Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.g = g #uS/cm^2
        self.e = e #mV
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ 1. / (1. + np.exp(-(V0 + 11.) / 12.)),  
                                    1. / (1. + np.exp( (V0 + 64.) / 11.))]])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            self.varinf = np.array([[1. / (1. + sp.exp(-(self.spV + 11.) / 12.)) , 
                                     1. / (1. + sp.exp( (self.spV + 64.) / 11.)) ]])
            self.tau = np.array([[(3.04 + 17.3 * sp.exp(-((self.spV + 60.) / 15.9)**2) + 25.2 * sp.exp(-((self.spV + 60.) / 57.4)**2)) / 2.95, \
                                (360. + (1010. + 24. * (self.spV + 65.)) * sp.exp(-((self.spV + 85.) / 48.)**2)) / 2.95]]) # ms
            # self.tau = np.array([[1.080549, 58.346879]])
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[ 1. / (1. + np.exp(-(V + 11.) / 12.)) ,  
                            1. / (1. + np.exp( (V + 64.) / 11.)) ]])
        # taum fitted to:
        # if V < -50.:
        #     return (1.25 + 175.03 * np.exp( 0.026 * V)) / 2.95
        # else:
        #     return (1.25 + 13.    * np.exp(-0.026 * V)) / 2.95
        taus = np.array([[ (3.04 + 17.3 * np.exp(-((V + 60.) / 15.9)**2) + 25.2 * np.exp(-((V + 60.) / 57.4)**2)) / 2.95, 
                            (360. + (1010. + 24. * (V + 65.)) * np.exp(-((V + 85.) / 48.)**2)) / 2.95 ]]) # ms
        return (svinf - sv) / taus


class Ktst(ionChannel):
    def __init__(self, g=0.001*1e6, e=-106., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Transient Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.g = g #uS/cm^2
        self.e = e #mV
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ 1. / (1. + np.exp(-(V0 + 10.) / 19.)),  
                                    1. / (1. + np.exp( (V0 + 76.) / 10.))]])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            self.varinf = np.array([[   1. / (1. + sp.exp(-(self.spV + 10.) / 19.)) ,  
                                        1. / (1. + sp.exp( (self.spV + 76.) / 10.)) ]])
            self.tau = np.array([[  (0.34 + 0.92 * sp.exp(-((self.spV + 81.) / 59.)**2)) / 2.95 , 
                                    (8.   + 49.  * sp.exp(-((self.spV + 83.) / 23.)**2)) / 2.95]]) # ms
            # self.tau = np.array([[1.080549, 58.346879]])
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[ 1. / (1. + np.exp(-(V + 10.) / 19.)) ,  
                            1. / (1. + np.exp( (V + 76.) / 10.)) ]])
        taus = np.array([[  (0.34 + 0.92 * np.exp(-((V + 81.) / 59.)**2)) / 2.95 , 
                            (8.   + 49.  * np.exp(-((V + 83.) / 23.)**2)) / 2.95 ]]) # ms
        return (svinf - sv) / taus


class KA(ionChannel):
    def __init__(self, g=0.0477*1e6, e=-75., V0=-65., conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        A-type potassium (Abbott, 2000) (Connor-Stevens model)
        '''
        self.ion = 'k'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['a', 'b']])
        self.statevar = np.array([[(0.0761 * np.exp(0.0314 * (V0+94.22)) / (1. + np.exp(0.0346 * (V0+1.17))))**(1./3.),
                                    (1. / (1. + np.exp(0.0688 * (V0 + 53.3))))**4]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            self.varinf = np.array([[(0.0761 * sp.exp(0.0314 * (self.spV+94.22)) / (1. + sp.exp(0.0346 * (self.spV+1.17))))**(1./3.),
                                     (1. / (1. + sp.exp(0.0688 * (self.spV + 53.3))))**4]])
            self.tau = np.array([[0.3632 + 1.158 / (1. + sp.exp(0.0497 * (self.spV+55.96))), 
                                    1.24 + 2.678 / (1. + sp.exp(0.0624 * (self.spV+50.)))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[(0.0761 * np.exp(0.0314 * (V+94.22)) / (1. + np.exp(0.0346 * (V+1.17))))**(1./3.),  
                                (1. / (1. + np.exp(0.0688 * (V + 53.3))))**4]])
        taus = np.array([[0.3632 + 1.158 / (1. + np.exp(0.0497 * (V+55.96))), \
                            1.24 + 2.678 / (1. + np.exp(0.0624 * (V+50.)))]]) # ms
        return (svinf - sv) / taus


class KA_prox(ionChannel): # TODO finish implementation
    def __init__(self, g=0.0477*1e6, e=-90., V0=-65., conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        A-type potassium (Kellems, 2010)

        !!! works in testsuite, but unstable in certain cases !!!
        '''
        self.ion = 'k'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['n', 'l']])
        self.statevar = np.array([[1. / (1.+self.alphan(V0)), 1. / (1.+self.alphal(V0))]])
        self.powers = np.array([[1,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 #* self.statevars[0,1]

            alphan = sp.exp(-0.038*(1.5 + 1./(1.+sp.exp((self.spV+40.)/5.))) * (self.spV-11.))
            betan = sp.exp(-0.038*(0.825 + 1./(1.+sp.exp((self.spV+40.)/5.))) * (self.spV-11.))
            alphal = sp.exp(0.11*(self.spV+56.))
            
            self.varinf = np.array([[1. / (1.+alphan), 1. / (1.+alphal)]])
            self.tau = np.array([[4.*betan / (1.+alphan), 0.2 + 27. / (1. + sp.exp(0.2-self.spV/22.))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def fun_statevar(self, V, sv, conc={}):
        taul = 0.2 + 27. / (1. + np.exp(0.2-V/22.))
        an = self.alphan(V); bn = self.betan(V); al = self.alphal(V)
        svinf = np.array([[1. / (1. + an), 1./(1. + al)]])
        taus = np.array([[4.*bn / (1+an), taul]])
        return (svinf - sv) / taus
        
    def alphan(self, V):
        return np.exp(-0.038*(1.5 + 1./(1.+np.exp((V+40.)/5.))) * (V-11.))
    
    def betan(self, V): 
        return np.exp(-0.038*(0.825 + 1./(1.+np.exp((V+40.)/5.))) * (V-11.))

    def alphal(self, V):
        return np.exp(0.11*(V+56.))


class SK(ionChannel):
    def __init__(self, g=0.00001*1e6, e=-80, V0=-65., conc0=[1e-4], nonlinear=False, calc=False, temp=0.):
        '''
        SK-type calcium-activated potassium current (Kohler et al., 1996)

        !!!Untested, not functional yet!!!

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = ['ca']

        self.e = e  # mV
        self.g = g  # uS/cm2

        self.varnames = np.array([['z']])
        self.statevar = np.array([[ 1./(1. + (0.00043/conc0[0])**4.8) ]])
        self.tau_array = np.array([[1.]])
        self.powers = np.array([[1.]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])

        self.nonlinear = nonlinear
        self.calc = calc

        self.V0 = V0
        self.conc0 = conc0

        if self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.zeros(len(self.concentrations), dtype=object)
            for ind, name in enumerate(self.concentrations):
                self.conc[ind] = sp.symbols(name)

            self.fun = self.statevars[0,0]**self.powers[0,0]

            self.varinf = np.array([[1./(1. + (0.00043/self.conc[0])**4.8)]], dtype=object)
            # make array of sympy floats to render it sympy compatible
            self.tau = np.zeros(self.tau_array.shape, dtype=object)
            for ind, tau in np.ndenumerate(self.tau_array):
                self.tau[ind] = sp.Float(self.tau_array[ind])
            self.fstatevar = (self.varinf - self.statevars) / self.tau

        if self.nonlinear:
            self.I0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None]) * (self.V0 - self.e)

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            # self.set_expansion_coeffs()

    def calc_linear(self, freqs):
        return {'V': np.zeros(freqs.shape, dtype=complex)} 

    def advance(self, V, dt, conc={'Ca':0.}):
        '''
        advance the state variables of the ion channel
        '''
        self.statevar += dt * self.fun_statevar(V, self.statevar, conc=conc)

    def getCurrent(self, V):
        '''
        returns the transmembrane current in nA, if self.nonlinear is True,
        returns the current without the equilibrium part
        '''
        I = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None]) * (V - self.e)
        if self.nonlinear:
            I -= self.I0
        return I

    def fun_statevar(self, V, sv, conc={'Ca':0.}):
        svinf = np.array([[1./(1. + (0.00043/conc[self.concentrations[0]])**4.8)]])
        return (svinf - sv) / self.tau_array


class Ca_LVA(ionChannel):
    def __init__(self, g=0.00001*1e6, e=50., V0=-65., conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        LVA calcium channel (Avery and Johnston, 1996; tau from Randall, 1997)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[1. / (1. + np.exp(-(V0 + 40.)/6.)), \
                                    1. / (1. + np.exp((V0 + 90.)/6.4))]])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            self.varinf = np.array([[1. / (1. + sp.exp(-(self.spV + 40.)/6.)), \
                                    1. / (1. + sp.exp((self.spV + 90.)/6.4))]])
            self.tau = np.array([[(5. + 20./(1. + sp.exp((self.spV  + 35.)/5.)))/2.95, 
                                    (20. + 50./(1. + sp.exp((self.spV + 50.)/7.)))/2.95]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[1. / (1. + np.exp(-(V + 40.)/6.)), \
                                1. / (1. + np.exp((V + 90.)/6.4))]])
        taus = np.array([[(5. + 20./(1. + np.exp((V + 35.)/5.)))/2.95, 
                                (20. + 50./(1. + np.exp((V + 50.)/7.)))/2.95]]) # 1/ms
        return (svinf - sv) / taus


class Ca_HVA(ionChannel):
    def __init__(self, g=0.00001*1e6, e=50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        High voltage-activated calcium channel (Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0)), \
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0))]])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            spalpham = -0.055 * (27. + self.spV) / (sp.exp(-(27. + self.spV)/3.8) - 1.)  #1/ms
            spbetam = 0.94 * sp.exp(-(75. + self.spV)/17.)  #1/ms
            spalphah = 0.000457 * sp.exp(-(13. + self.spV)/50.)   #1/ms
            spbetah = 0.0065 / (sp.exp(-(self.spV + 15.)/28.) + 1.)   #1/ms
            
            self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
            self.tau = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm), ah / (ah + bh)]])
        taus = np.array([[1. / (am + bm), 1. / (ah + bh)]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return -0.055 * (27. + V) / (np.exp(-(27. + V)/3.8) - 1.)
        
    def betam(self, V):
        return 0.94 * np.exp(-(75. + V)/17.)
        
    def alphah(self, V):
        return 0.000457 * np.exp(-(13. + V)/50.)
        
    def betah(self, V):
        return 0.0065 / (np.exp(-(V + 15.)/28.) + 1.)


class regenerative(ionChannel):
    def __init__(self):
        pass
        
        
class L(ionChannel):
    def __init__(self,  g=0.0003*1e6, e=-54.4, V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Leak current
        '''
        self.ion = ''
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = []
        self.statevar = np.array([[1.]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.array([[sp.symbols('x')]])
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]
            
            self.varinf = np.array([[sp.Float(1.)]])
            self.tau = np.array([[sp.Float(1.)]]) # 1/ms
            self.fstatevar = np.array([[sp.Float(0.)]])  
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, dt, conc={}):
        return np.array([[0.]])
        
    def set_expansion_coeffs(self, order=2): pass
    
    def calc_offset(self, freqs=None):
        if freqs==None:
            return self.g*(self.V0 - self.e)
        else:
            return self.g*(self.V0 - self.e)*np.ones(freqs.shape, dtype=complex)
        
    def calc_linear(self, freqs):
        return {'V': self.g*np.ones(freqs.shape, dtype=complex)}
    
    def calc_quadratic(self, freqs):
        return {'V': np.zeros((len(freqs), len(freqs)), dtype=complex)}


class conc_dynamics:
    # def __init__(self): 

    def advance(self, I, dt):
        self.conc += dt*self.fun_statevar(I, dt)
        # print self.conc

    def getConc(self, I):
        return self.conc #+ self.conc0

    def calc_linear(self, freqs):
        '''
        computes contribution of conc to other impedances
        '''
        coeffstatevar = self.compute_lincoeff()

        coeff = {}
        coeff['v'] = coeffstatevar['v'] / (freqs - coeffstatevar['conc'])
        coeff['i'] = coeffstatevar['i'] / (freqs - coeffstatevar['conc'])

        return coeff

    def compute_lincoeff(self):
        coeffstatevar = {'conc': sp.diff(self.fstatevar, self.conc, 1),
                            'v': sp.diff(self.fstatevar, self.spV, 1),
                            'i': sp.diff(self.fstatevar, self.spI, 1)}
        for var in coeffstatevar.keys():
            coeffstatevar['key'].subs(self.spV, self.V0)
            coeffstatevar['key'].subs(self.spI, self.I0)
            coeffstatevar['key'].subs(self.conc, self.conc0)

        return coeffstatevar

    def write_mod_file(self):
        '''
        Writes a modfile of the ion channel for simulations with neuron
        '''
        f = open('../mech/' + self.__class__.__name__ + '.mod', 'w')
        
        f.write(': This mod file is automaticaly generated by the ionc.write_mode_file() function in /source/ionchannels.py \n\n')
        
        f.write('NEURON {\n')
        f.write('    SUFFIX ' + self.__class__.__name__ + '\n')
        f.write('    USEION ' + self.ion + ' READ i' + self.ion + ' WRITE ' + self.ion + 'i' + '\n')
        f.write('    RANGE  gamma, gamma_frac, tau, inf' + '\n')
        f.write('    THREADSAFE' + '\n')
        f.write('}\n\n')
        
        f.write('PARAMETER {\n')
        f.write('    gamma_frac = ' + str(0.05) + '\n')
        f.write('    gamma = ' + str(self.gamma_orig) + '\n')
        f.write('    tau = ' + str(self.tau) + ' (ms)' + '\n')
        f.write('    inf = ' + str(self.inf) + ' (mM)' + '\n')
        f.write('}\n\n')
        
        f.write('UNITS {\n')
        f.write('    (mA) = (milliamp)' + '\n')
        f.write('    (mV) = (millivolt)' + '\n')
        f.write('    (mM) = (milli/liter)' + '\n')
        f.write('}\n\n')
        
        f.write('ASSIGNED {\n')
        f.write('    i' + self.ion + ' (mA/cm2)' + '\n')
        f.write('}\n\n')
        
        f.write('STATE {\n')
        f.write('    ' + self.ion + 'i (mM)' + '\n')
        f.write('}\n\n')
        
        f.write('BREAKPOINT {\n')
        f.write('    SOLVE states METHOD cnexp' + '\n')
        f.write('}\n\n')

        f.write('INITIAL {\n')
        f.write('    gamma = gamma*gamma_frac' + '\n')
        f.write('    ' + self.ion + 'i = inf' + '\n')
        f.write('}\n\n')
        
        f.write('DERIVATIVE states {\n')
        f.write('    ' + self.ion + 'i\' = gamma * i' + self.ion + ' - (' + self.ion + 'i - inf) / tau' + '\n')
        f.write('}\n\n')
        
        f.close()


class conc_ca(conc_dynamics):
    def __init__(self, gamma=0.05, tau=80., inf=1e-4, V0=-65., conc0=0., I0=0., nonlinear=False, calc=False):
        self.ion = 'ca'

        self.inf = inf  # (mM)
        self.tau = tau     # (ms)
        self.gamma_orig = 0.1*1e3 / (2.*0.1*9.64853399)
        self.gamma = self.gamma_orig*gamma*1e-7
        # print 'gamma_orig_ionc=', self.gamma_orig
        # print 'gamma_ionc=', self.gamma

        self.V0 = V0#; self.I0 = I0

        # der = lambda c: -(c-self.inf) / self.tau + self.gamma*self.I0

        self.conc0 = conc0
        self.conc = conc0

        self.nonlinear = nonlinear
        self.calc = calc

        if self.nonlinear or self.calc:    
            # sympy expressions
            self.spV = sp.symbols('v')
            self.conc = sp.symbols(self.ion)
            self.spI = sp.symbols('i'+self.ion)
            
            self.fstatevar = - (self.conc - self.inf) / self.tau + self.gamma * self.spI

    def fun_statevar(self, I, dt):
        return 1./dt * ( (np.exp(-dt / self.tau) - 1.) * self.conc + \
                (1. - np.exp(-dt / self.tau)) * self.tau * ( self.inf/self.tau + self.gamma*I ) )
        # return (self.inf - self.conc) / self.tau + self.gamma * I


        
########################################################################
 

## make mod files ######################################################
if __name__ == "__main__":
    fcc = open('cython_code/ionchannels.cc', 'w')
    fh = open('cython_code/ionchannels.h', 'w')
    fstruct = open('cython_code/channelstruct.h', 'w')

    fh.write('#include <iostream>' + '\n')
    fh.write('#include <string>' + '\n')
    fh.write('#include <vector>' + '\n')
    fh.write('#include <string.h>' + '\n')
    fh.write('#include <stdlib.h>' + '\n')
    fh.write('#include <algorithm>' + '\n')
    fh.write('#include <math.h>' + '\n\n')
    fh.write('#include "memcurrent.h"' + '\n\n')
    fh.write('using namespace std;' + '\n\n')
    
    fcc.write('#include \"ionchannels.h\"' + '\n\n')

    fstruct.write('struct ionc_set{' + '\n')
    
    fstruct.close()
    fcc.close()
    fh.close()
    
    hchan = h(nonlinear=True)
    hchan.write_mod_file()
    hchan.write_cpp_code()
    
    h_HAYchan = h_HAY(nonlinear=True)
    h_HAYchan.write_mod_file()
    h_HAYchan.write_cpp_code()
    
    Nachan = Na(nonlinear=True)
    Nachan.write_mod_file()
    Nachan.write_cpp_code()
    
    Na_pchan = Na_p(nonlinear=True)
    Na_pchan.write_mod_file()
    Na_pchan.write_cpp_code()
    
    Na_Tachan = Na_Ta(nonlinear=True)
    Na_Tachan.write_mod_file()
    Na_Tachan.write_cpp_code()
    
    Na_Ta2chan = Na_Ta2(nonlinear=True)
    Na_Ta2chan.write_mod_file()
    Na_Ta2chan.write_cpp_code()
    
    Klvachan = Klva(nonlinear=True)
    Klvachan.write_mod_file()
    Klvachan.write_cpp_code()
    
    Kchan = K(nonlinear=True)
    Kchan.write_mod_file()
    Kchan.write_cpp_code()
    
    Kpstchan = Kpst(nonlinear=True)
    Kpstchan.write_mod_file()
    Kpstchan.write_cpp_code()
    
    Ktstchan = Ktst(nonlinear=True)
    Ktstchan.write_mod_file()
    Ktstchan.write_cpp_code()
    
    Kv3_1chan = Kv3_1(nonlinear=True)
    Kv3_1chan.write_mod_file()
    Kv3_1chan.write_cpp_code()
    
    mchan = m(nonlinear=True)
    mchan.write_mod_file()
    mchan.write_cpp_code()
    
    KAchan = KA(nonlinear=True)
    KAchan.write_mod_file()
    KAchan.write_cpp_code()
    
    KAproxchan = KA_prox(nonlinear=True)
    KAproxchan.write_mod_file()    
    KAproxchan.write_cpp_code()    
    
    SKchan = SK(nonlinear=True, calc=True)
    SKchan.write_mod_file()
    
    Ca_HVAchan = Ca_HVA(nonlinear=True)
    Ca_HVAchan.write_mod_file()
    Ca_HVAchan.write_cpp_code()
    
    Ca_LVAchan = Ca_LVA(nonlinear=True)
    Ca_LVAchan.write_mod_file()
    Ca_LVAchan.write_cpp_code()
    
    Lchan = L(nonlinear=True)
    Lchan.write_mod_file()
    Lchan.write_cpp_code()

    Caconc = conc_ca()
    Caconc.write_mod_file()

    fstruct = open('cython_code/channelstruct.h', 'a')
    fstruct.write('};')
    fstruct.close()
########################################################################
            
    
