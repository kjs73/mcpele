# distutils: language = c++
# distutils: sources = actions.cpp

cimport cython
cimport numpy as np

import numpy as np
import sys

#===============================================================================
# Adjust step size
#===============================================================================
        
cdef class _Cdef_AdjustStep(_Cdef_Action):
    """This class is the python interface for the c++ pele::AdjustStep action class implementation
    """
    cdef cppAdjustStep* newptr
    def __cinit__(self, target, factor, niter, navg):
        self.thisptr = <cppAction*>new cppAdjustStep(target, factor, niter, navg)
        self.newptr = <cppAdjustStep*> self.thisptr
    
    def __dealloc__(self):
        del self.thisptr
        
class AdjustStep(_Cdef_AdjustStep):
    """This class is the python interface for the c++ AdjustStep implementation.
    """

#===============================================================================
# Record Energy Histogram
#===============================================================================        

cdef class _Cdef_RecordEnergyHistogram(_Cdef_Action):
    """This class is the python interface for the c++ pele::RecordEnergyHistogram acceptance test class implementation
    """
    cdef cppRecordEnergyHistogram* newptr
    def __cinit__(self, min, max, bin, eqsteps):
        self.thisptr = <cppAction*>new cppRecordEnergyHistogram(min, max, bin, eqsteps)
        self.newptr = <cppRecordEnergyHistogram*> self.thisptr
    
    def __dealloc__(self):
        del self.thisptr
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_histogram(self):
        """return a histogram array"""
        cdef _pele.Array[double] histi = self.newptr.get_histogram()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
              
        return hist
        
    def print_terminal(self, ntot):
        self.newptr.print_terminal(ntot)
    
    def get_bounds_val(self):
        Emin = self.newptr.get_min()
        Emax = self.newptr.get_max()
        return Emin, Emax
    
    def get_mean_variance(self):
        mean = self.newptr.get_mean()
        variance = self.newptr.get_variance()
        return mean, variance
        
class RecordEnergyHistogram(_Cdef_RecordEnergyHistogram):
    """This class is the python interface for the c++ RecordEnergyHistogram implementation.
    """

#===============================================================================
# RecordEnergyTimeseries
#===============================================================================
        
cdef class _Cdef_RecordEnergyTimeseries(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordEnergyTimeseries action class implementation
    """
    cdef cppRecordEnergyTimeseries* newptr
    def __cinit__(self, niter, record_every):
        self.thisptr = <cppAction*>new cppRecordEnergyTimeseries(niter, record_every)
        self.newptr = <cppRecordEnergyTimeseries*> self.thisptr
    
    def __dealloc__(self):
        del self.thisptr
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_time_series(self):
        """return a energy time series array"""
        cdef _pele.Array[double] seriesi = self.newptr.get_time_series()
        cdef double *seriesdata = seriesi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(seriesi.size())
        cdef size_t i
        for i in xrange(seriesi.size()):
            series[i] = seriesdata[i]
              
        return series
    
    def clear(self):
        """clears time series"""
        self.newptr.clear()
    
class RecordEnergyTimeseries(_Cdef_RecordEnergyTimeseries):
    """This class is the python interface for the c++ RecordEnergyTimeseries implementation.
    """