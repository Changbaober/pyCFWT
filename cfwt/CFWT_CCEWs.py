
"""
Combined FFT-Wavelet method for Convectively Coupled Equatorial Waves (CCEWs)
Created by: Chang Xu
xuch54@gmail.com
2025

References:

Kikuchi, K., and B. Wang (2010). Spatiotemporal Wavelet Transform and the 
Multiscale Behavior of the Madden–Julian Oscillation. 
Journal of Climate. 23, 3814–3834. 
https://doi.org/10.1175/2010JCLI2693.1.

Kikuchi, K. (2014). An introduction to combined Fourier–wavelet transform 
and its application to convectively coupled equatorial waves. 
Climate Dynamics. 43, 1339–1356. 
https://doi.org/10.1007/s00382-013-1949-8

The corresponding Fortran code comes from:
http://iprc.soest.hawaii.edu/users/kazuyosh

Functions in pre-processing data comes from Alejandro Jaramillo's wk-package: 
https://github.com/mmaiergerber/wk_spectra

Thanks for the help from Qin Jiang in comparing and correcting the python 
version of the CFWT with the original Fortran version.
"""

from __future__ import annotations
from typing import Optional, Tuple, Literal
from dataclasses import dataclass

import numpy as np
from dataclasses import dataclass
from typing import Optional
import netCDF4 as nc

import xarray as xr
import datetime
import os
# from scipy.signal import signal
import scipy.signal as signal
from scipy import stats
import math

import matplotlib.pyplot as plt

# SciPy filter (If the environment does not have scipy, the following two types of filters can be changed to FFT versions)
from scipy.signal import butter, filtfilt

import matplotlib.colors as mcolors


#------------------------------------------------------------------------------
# Part 1: Processing Data
#------------------------------------------------------------------------------

"""
Removing the long term linear trend (conserving the mean) and by 
eliminating the annual cycle by removing all time periods less than 
a corresponding critical frequency.

"""

def _detrend_1d(y):
    y = np.asarray(y, dtype=np.float64)
    t = np.arange(y.size, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(y, np.nan)
    if not np.all(ok):
        y[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), y[ok])
    p = np.polyfit(t, y - y.mean(), 1)
    return y - (p[0]*t + p[1])

def _hp_1d(y, spd, cutoff_days=120, order=4):
    """零相位 Butterworth 高通；极短序列退化为去均值+去趋势"""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(y, np.nan)
    if not np.all(ok):
        y[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), y[ok])

    fs = float(spd)
    fc = 1.0 / float(cutoff_days)
    Wn = (fc / (0.5 * fs))
    # 守护：极端采样或 cutoff 异常
    Wn = np.clip(Wn, 1e-6, 0.999999)

    b, a = butter(order, Wn, btype='high', analog=False)
    # SciPy 默认 padlen = 3*(max(len(a),len(b))-1)
    padlen_req = 3 * (max(len(a), len(b)) - 1)
    if y.size <= padlen_req + 1 or y.size < 2*order + 5:
        out = _detrend_1d(y)
    else:
        out = filtfilt(b, a, y, axis=0)
    # 保证零均值
    return out - np.nanmean(out)

def _apply_timewise(da, func, **kwargs):
    return xr.apply_ufunc(
        func, da,
        input_core_dims=[['time']], output_core_dims=[['time']],
        vectorize=True, dask='parallelized',
        kwargs=kwargs, output_dtypes=[np.float64]
    ).astype(da.dtype)

def anomalies_for_projectA(da: xr.DataArray, spd: int = 1, cutoff_days: int = 120):
    """项目A：统一 HP120；长度太短时自动退化为去均值+去趋势"""
    if 'time' not in da.dims:
        da = da.transpose('time', ...)
    anom = _apply_timewise(da, _hp_1d, spd=spd, cutoff_days=cutoff_days, order=4)
    return anom

# ---------- 方案 B/C：按长度自动分流（更“标准”） ----------

def _remove_seasonal_harmonics(da, spd=1, n_harm=3):
    """去 1..n 年循环谐波；适合 1-2 年及以上"""
    if 'time' not in da.dims:
        da = da.transpose('time', ...)
    nt = da.sizes['time']
    # rFFT
    cf = xr.apply_ufunc(np.fft.rfft, da, input_core_dims=[['time']],
                        output_core_dims=[['freq']], vectorize=True,
                        dask='parallelized', output_dtypes=[np.complex128])
    # 频率坐标（cycles/day）
    freq = np.fft.rfftfreq(nt, d=1.0/spd)
    # 年循环谐波索引（1/365, 2/365, ...）
    tol = 0.5/nt  # 分辨率容差
    mask = np.zeros_like(freq, dtype=bool)
    for n in range(1, n_harm+1):
        target = n/365.0
        mask |= np.abs(freq - target) < max(tol, 1.5/nt)
    # 清零这些谐波
    cf = cf.where(~xr.DataArray(mask, dims=['freq']), 0.0)
    # IFFT
    rec = xr.apply_ufunc(np.fft.irfft, cf, input_core_dims=[['freq']],
                         output_core_dims=[['time']], vectorize=True,
                         dask='parallelized', kwargs={'n': nt},
                         output_dtypes=[np.float64]).astype(da.dtype)
    # 再去一次线性趋势
    rec = _apply_timewise(rec, _detrend_1d)
    return rec

def _climatology_anom_long(da, group='dayofyear'):
    """>5 年：做完整气候态异常。group 可选 'dayofyear'/'month'。"""
    clim = da.groupby(f'time.{group}').mean('time')
    anom = da.groupby(f'time.{group}') - clim
    # 去趋势以消除残余慢变
    anom = _apply_timewise(anom, _detrend_1d)
    return anom

def anomalies_auto(da: xr.DataArray, spd: int = 1):
    """
    更“教科书”的自动分流（供结构/长期研究）：
    < 90d: 去均值+去趋势
    90–365d: HP120
    365–730d: 线性去趋势 + 去年循环谐波（1–3阶）
    2–5y: 同上 +（可选）再做 >400–730d 的年际去除
    >5y: 完整气候态（按DOY或月） + 趋势
    """
    if 'time' not in da.dims:
        da = da.transpose('time', ...)
    N = da.sizes['time']

    if N < 90:
        anom = _apply_timewise(da, _detrend_1d)
    elif N < 365 and N >= 90:
        anom = _apply_timewise(da, _hp_1d, spd=spd, cutoff_days=120, order=4)
    elif N <= 730 and N >= 365:
        # 你 anom_3d.py 的风格：线性趋势 + 年循环谐波
        try:
            from anom_3d import calcClimTLL, calcAnomTLL
            x = da.transpose('time', 'lat', 'lon')
            clim = calcClimTLL(x, spd=spd, smooth=True, nsmth=4)
            anom = calcAnomTLL(x, clim, spd=spd).transpose(*da.dims).astype(da.dtype)
        except Exception:
            # 兜底：谐波法
            anom = _remove_seasonal_harmonics(da, spd=spd, n_harm=3)
    elif N <= 1825 and N>730:
        # 2–5年
        anom = _remove_seasonal_harmonics(da, spd=spd, n_harm=3)
        # 可选：再高通去年际（例如 cutoff=450–730 天）
        anom = _apply_timewise(anom, _hp_1d, spd=spd, cutoff_days=450, order=4)
    else:
        # >5年：完整气候态
        anom = _climatology_anom_long(da, group='dayofyear')

    return anom.astype(da.dtype)

#------------------------------------------------------------------------------
# Part 2: CFWT Method_Setup
#------------------------------------------------------------------------------

"""
Part 2 Setup.

    2.1 Parameters setting (CFWTParams)

    2.2 Parameters calculation (CFWTSetup)
        2.2.1 Scale range determination (setup_jrange)
        2.2.2 Scale array generation (setup_scale) 
        2.2.3 Time series padding setting (setup_padding)
        2.2.4 Frequency array setting (setup_frq)
        2.2.5 Wavelet parameter calculation (get_Cdelta, get_Cpsi)

Output: Parameters
"""

@dataclass
class CFWTParams:
    """
    Parameters class for Combined Fourier-Wavelet Transform (CFWT).

    This class defines all necessary parameters following Section 2.2 of the paper.
    The parameters control the wavelet scales, frequency resolution, and 
    computational grid specifications.

    Attributes from paper:
    - w0 = 6.0: Central frequency of Morlet wavelet (Section 2.1)
    - dj = 0.2: Scale resolution parameter (Section 2.2)
    - Scales follow: s = s0*2^(j*dj), j = -Jmax,...,Jmax

    Grid Parameters
    --------------
    Nx : int
        Number of longitude points (zonal dimension)
    Ny : int
        Number of latitude points (meridional dimension) 
    Nt : int 
        Number of time points
    dx : float
        Longitude grid spacing in degrees
    dt : float
        Time step in days
    jlats, jlatn : int
        Southern and northern latitude indices for analysis

    Wavelet Parameters  
    -----------------
    w0 : float
        Morlet wavelet central frequency (Equation 3)
    dj : float
        Scale resolution parameter
    df : float
        Frequency interval for output
    Nfrq : int
        Number of frequency points
    
    Computed Parameters
    ------------------
    s0 : float, optional
        Smallest wavelet scale (computed in setup)
    scale : np.ndarray, optional 
        Array of wavelet scales (computed in setup)
    Cpsi, Cdelta : float, optional
        Wavelet normalization constants (computed in setup)
    """
    def __init__(self):
        # Grid parameters (corresponding Nx,Ny,Nt in FORTRAN)
        self.Nx: int = 360  # Number of longitude points
        self.Ny: int = 42  # Number of latitude points
        self.Nt: int = 40  # Number of time points
        
        # Spatial and temporal resolution (corresponding dx,dt in FORTRAN)
        self.dx: float = 1.0   # Longitude resolution
        self.dt: float = 1 # Time resolution /day
        self.jlats: int = 0           # South boundary latitude index
        self.jlatn: int = self.Ny-1   # North boundary latitude index
        
        # Wavelet parameters (corresponding w0,dj in FORTRAN)
        self.w0: float = 6.0   # Central frequency of Morlet wavelet
        self.dj: float = 0.2   # Scale resolution ??????? 0.4
        
        # Frequency parameters (corresponding df, Nfrq in FORTRAN)
        self.df: float = 1.0/40  # Frequency interval
        self.Nfrq: int = int(1.0/(2*self.dt*self.df)) + 1  # Number of frequencies
        
        # Missing value (corresponding rmiss in FORTRAN)
        self.rmiss: float = -9999.0
        
        # Variables to be initialized
        self.s0: Optional[float] = None     # Smallest wavelet scale
        self.frq: Optional[np.ndarray] = None  # Frequency array
        self.scale: Optional[np.ndarray] = None  # Scale array
        self.pi: float = np.pi
        self.Cpsi: Optional[float] = None   # Wavelet admissibility constant
        self.Cdelta: Optional[float] = None # Wavelet normalization constant
        
        # Additional attributes for Python implementation
        self.Jmax: Optional[int] = None  # Maximum scale index

        self.Npad: Optional[int] = None
    
    def validate(self) -> None:
        """
        Validate that all necessary parameters are initialized
        """
        required_attrs = ['s0', 'frq', 'scale', 'Cpsi', 'Cdelta', 'Jmax', 'jlats', 'jlatn']
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} has not been initialized")

    def update_grid(self, Nx: int, Ny: int, Nt: int):
        self.Nx, self.Ny, self.Nt = int(Nx), int(Ny), int(Nt)
    
        # 重新计算与时间维相关的频率网格
        self.Npad = 1 << int(np.ceil(np.log2(max(1, self.Nt))))  # 下一个2的幂
        self.df = 1.0 / (self.Npad * self.dt)                    # cycles per day
        self.Nfrq = self.Npad // 2 + 1
        self.frq = (np.arange(self.Nfrq, dtype=np.float32) * self.df)
        # 注意：Jmax 和 scale 通常固定于你设定的频带；若你另有根据 frq 自动算 Jmax/scale 的函数，可在此调用。
        return self


class CFWTSetup:
    """
    Setup class for initializing CFWT computation parameters.

    Implements the initialization procedures described in Section 2.2:
    1. Scale range determination
    2. Wavelet scale array generation 
    3. FFT padding setup
    4. Frequency array definition
    5. Wavelet constant computation

    All initialization follows the procedures from paper Appendix A.
    """
    def __init__(self, params: CFWTParams):
        """
        Initialize CFWTSetup class
        
        Parameters
        ----------
        params : CFWTParams
            Parameter object containing all CFWT parameters
        """
        self.params = params
    
    def initialize(self, check: bool = False) -> None:
        """
        Execute complete initialization process

        Parameters
        ----------
        check : bool, optional
            Validation mode flag (default: False)
        
        Notes
        -----
        Follows initialization sequence from Article Section 2.2:
        1. Scale range setup
        2. Scale array generation
        3. Time series padding
        4. Frequency array setup
        5. Wavelet constants computation
        """
        # Set check mode
        self.params.check = check
        self.setup_padding() # Setup FFT padding

        self.params.df   = 1.0 / (self.params.Npad * self.params.dt)      # cycles/day
        self.params.Nfrq = self.params.Npad // 2 + 1
        self.setup_frq() # Setup frequency array
        
        # Setup sequence based on FORTRAN code
        self.setup_jrange(check) # This sets s0 and Jmax
        self.setup_scale() # Setup scale array
        self.get_Cpsi() # Compute wavelet admissibility
        self.get_Cdelta() # Compute reconstruction factor
        
        assert self.params.Nfrq == self.params.Npad // 2 + 1
        J = int(self.params.Jmax)
        sc = np.asarray(self.params.scale, float)
        
        # Two layouts are allowed:
        if sc.size == (J + 1):
            # Traditional "full positive scale": strictly increasing
            assert np.all(sc > 0)
            assert np.all(np.diff(sc) > 0)
        elif sc.size == (2*J + 1): # This is what we based
            # "Signed scale": [-J..-1, 0, 1..J], increasing by absolute value
            assert np.isclose(sc[J], 0.0)
            assert np.all(sc[:J] < 0) and np.all(sc[J+1:] > 0)
            assert np.all(np.diff(np.abs(sc[J+1:])) > 0) and np.all(np.diff(np.abs(sc[:J])) < 0)
        else:
            raise AssertionError("scale length must be J+1 (positive) or 2*J+1 (signed)")
        
        # Validate initialization
        self.params.validate()

        # Print settings
        print(f"Npad = {self.params.Npad}")
        print(f"Jmax = {self.params.Jmax}")
        print(f"Cdelta = {self.params.Cdelta}")
        print(f"Cpsi = {self.params.Cpsi}")
        print(f"Nfrq = {self.params.Nfrq}")
        print(f"s0 = {self.params.s0}")

    def setup_jrange(self, check: bool = False) -> None:
        """
        Setup scale range parameters following Section 2.2.

        Determines smallest scale s0 and maximum scale index Jmax
        based on:
        - Nyquist frequency f_nyq = 1/(2*dt)
        - Desired frequency resolution df
        - Wavelet parameter w0

        Parameters
        ----------
        check : bool
            If True, use parameters for code validation
            If False, use parameters for analysis

        Notes
        -----
        Scale range is set to cover frequencies from
        Nyquist frequency to lowest resolvable frequency 1/df
        """
        # Calculate Nyquist frequency: f_nyq = 1/(2*dt)
        f_nyq = 1.0/(2*self.params.dt)
        s0_coefficient = 0.9  # Scale coefficient

        if check:
            # Check mode: used for code validation
            self.params.s0 = 0.7 * self.params.dt
            # Calculate maximum scale: smax = (w0 + sqrt(w0^2 + 2))/(4π/(Nt*dt))
            smax = (self.params.w0 + np.sqrt(self.params.w0**2 + 2.0)) / \
                (4.0*self.params.pi/(self.params.Nt*self.params.dt))
            # Calculate Jmax: ceiling(log(smax/s0)/(dj*log(2))) + 2
            self.params.Jmax = int(np.ceil(
                (np.log(smax/self.params.s0))/(self.params.dj*np.log(2.0)))) + 2
        else:
            # Normal mode: used for analysis
            # Calculate smallest scale: s0 = (w0 + sqrt(w0^2 + 2))/(4π*f_nyq)*0.9
            self.params.s0 = (self.params.w0 + np.sqrt(self.params.w0**2 + 2.0)) / \
                            (4.0*self.params.pi*f_nyq) * s0_coefficient
            # Calculate maximum scale: smax = (w0 + sqrt(w0^2 + 2))/(4π*df)
            smax = (self.params.w0 + np.sqrt(self.params.w0**2 + 2.0)) / \
                (4.0*self.params.pi*self.params.df)
            # Calculate Jmax: ceiling(log(smax/s0)/(dj*log(2))) + 1
            self.params.Jmax = int(np.ceil(
                (np.log(smax/self.params.s0))/(self.params.dj*np.log(2.0)))) + 1
    
    def setup_scale(self) -> None:
        """
        Setup wavelet scale array following Section 2.1.

        Generates array of scales following the dyadic form:
        s_j = s0 * 2^(j*dj)
        where:
        - s0 is smallest scale
        - j ranges from -Jmax to Jmax
        - dj is scale resolution parameter

        Sets the scale array attribute:
        self.params.scale = [s_{-Jmax}, ..., s_0, ..., s_{Jmax}]

        Notes
        -----
        Scale distribution follows wavelet analysis conventions
        discussed in Section 2.1 and Appendix A.
        """
        # Initialize scale array with length 2*Jmax+1
        J  = int(self.params.Jmax)
        dj = float(self.params.dj)
        s0 = float(self.params.s0)
        sc = np.empty(2*J + 1, dtype=np.float64)
        for j in range(-J, J+1):
            sc[j + J] = np.sign(j) * s0 * (2.0 ** (abs(j) * dj))
        self.params.scale = sc
    
    def setup_padding(self) -> None:
        """
        Setup padding for FFT computation.

        Determines padding length for efficient FFT computation
        as discussed in Section 2.2.

        In analysis mode:
        - Finds next power of 2 >= Nt
        In validation mode:
        - Uses original length Nt

        Sets the padding length attribute:
        self.params.Npad = 2^ceiling(log2(Nt))

        Notes
        -----
        Padding prevents wraparound effects in FFT computation
        as explained in paper Section 2.2.
        """
        if self.params.check:
            self.params.Npad = self.params.Nt
        else:
            self.params.Npad = int(2 ** np.ceil(np.log2(float(self.params.Nt))))
    
    def setup_frq(self) -> None:
        """
        Define frequency array for output spectrum.

        Sets up frequencies following Section 2.2:
        f_n = (n-1)*df, n = 1,...,Nfrq
        where:
        - df is frequency resolution
        - Nfrq determined by Nyquist frequency

        Sets the frequency array attribute:
        self.params.frq = [0, df, 2df, ..., (Nfrq-1)*df]

        Notes
        -----
        Frequency array is used for final spectrum representation
        as described in Section 3.1.
        """
        self.params.frq = np.arange(self.params.Nfrq, dtype=np.float32) * float(self.params.df)
            
    def get_Cdelta(self) -> None:
        """
        Compute wavelet normalization constant Cdelta.

        Implements numerical computation of normalization constant
        following Equation (6):
        Cdelta = sqrt(2π) ∫ exp(-0.5(w-w0)²)/w dw

        This constant is used in signal reconstruction:
        g(x,t) = (1/Cdelta) * Real{sum[T{g}(k,s,t)]}
        """
        imax = 10000  # Number of integration points
        norm = np.float64(1.0)    # Normalization factor
        # maxw = np.float64(self.params.w0 * 2)  # Maximum frequency
        maxw = float(self.params.w0 + 8.0) # Chang Fix 2508
        dw = maxw / np.float64(imax)  # Frequency increment
        acc = 0.0
        
        # Numerical integration using loop
        for i in range(imax):
            w = dw * np.float64(i + 1)  # Current frequency
            acc += np.exp(-0.5 * (w - self.params.w0) ** 2) / w
            
        # Apply final normalization
        self.params.Cdelta = np.sqrt(np.float64(2.0)*self.params.pi) * dw * acc
        
    def get_Cpsi(self) -> None:
        """
        Compute wavelet admissibility constant Cpsi.

        Implements numerical computation of admissibility constant
        following Equation (8):
        Cpsi = 2π ∫ |ψ(w)|²/w dw

        This constant is used for energy conservation:
        ∫|g(x,t)|²dt = (1/Cpsi)∫|T{g}(k,s,t)|²dsdt
        """
        imax = 10000  # Number of integration points
        maxw = float(self.params.w0 + 8.0)  # Maximum frequency
        dw = maxw / np.float64(imax)  # Frequency increment
        acc = 0.0
        # Numerical integration using loop
        for i in range(imax):
            w = dw * np.float64(i + 1)  # Current frequency
            acc += (np.exp(-0.5 * (w - self.params.w0) ** 2)) ** 2 / w
            
        # Apply final normalization
        self.params.Cpsi = 2.0 * np.pi * dw * acc


#------------------------------------------------------------------------------
# Part 3: CFWT Method_Core Cal for spectrum
#------------------------------------------------------------------------------

"""
Part 3 Core Calculation.

    3.1 _compute_cfwt_single: core function to calculate spectrum in scale domain
    3.2 compute_cfwt: main function to calculate spectrum in frequency domain
    3.3 _cfftf_xt: lon-time fft calculation in G(k,ω)
    3.4 arrange_array: rearrange frequency to make sure 0 is in the middle
    3.5 spec_scale2frq: main function to convert spectrum from scale domain to frequency domain
    3.6 _spec_scale2frq_i: core function to convert spectrum from scale domain to frequency domain

Output: CFWT coefficient and power spectrum
"""

class CFWTComputation:
    """
    Core computation class for Combined Fourier-Wavelet Transform (CFWT).
    
    This class implements the CFWT algorithm described in:
    "An introduction to combined Fourier–wavelet transform and its application 
    to convectively coupled equatorial waves" (Kikuchi, 2014)
    
    The CFWT is defined as a combination of:
    - Fourier transform in longitude (space)
    - Wavelet transform in time
    
    Key equations from the paper:
    - Equation (1): Basic CFWT definition
      T{g}(k,s,τ) = (1/2π) ∫[0→2π] dx ∫[-∞→∞] g(x,t) ψ*s,τ(t) e^{ikx} dt
    - Equation (3): Morlet wavelet definition
      ψ(t) = e^{iω0t} e^{-t²/2}
    - Equation (9): Local power spectrum
      P(k,s,t) = |T{g}(k,s,t)|²
    """
    def __init__(self, params: CFWTParams):
        """
        Initialize computation with pre-initialized parameters
        
        Parameters
        ----------
        params : CFWTParams
            Must be initialized using CFWTSetup before computation
        """
        self.params = params
        # print(f'\nfreq:{self.params.frq}')
        if not self._check_initialization():
            raise ValueError("Parameters must be initialized using CFWTSetup before computation")
    
    def _check_initialization(self) -> bool:
        """Check if parameters are properly initialized"""
        try:
            self.params.validate()
            return True
        except ValueError:
            return False
    
    # core function
    def _compute_cfwt_single(self, data_xt: np.ndarray, scale1: float) -> np.ndarray:
        """
        Execute CFWT for a single scale.
        
        Implements the core CFWT computation following Equation (2) from the paper:
        T{g}(k,s,τ) = |s|^(1/2) ∫[-∞→∞] ψ^*(sω) G(k,ω) e^{iωτ}dω
        
        The computation is done in following steps:
        1. Remove time mean (corresponds to data preprocessing)
        2. Compute 2D FFT to get G(k,ω)
        3. Multiply with scaled Morlet wavelet (Equation 3)
        4. Inverse FFT to get CFWT coefficients

        Parameters
        ----------
        data_xt : np.ndarray, shape (Nx, Nt)
            Input longitude-time section data.
            Should be real-valued with:
            - Nx: number of longitude points
            - Nt: number of time points
        scale1 : float
            Current wavelet scale value.
            Relates to frequency via Equation (5) in paper:
            s = (ω0 + (ω0² + 2)^(1/2))/(4πf)

        Returns
        -------
        np.ndarray, shape (Nx, Npad)
            Complex CFWT coefficients.
            - First dimension: longitude (k)
            - Second dimension: time (τ)
            Padded to length Npad for FFT computation
            Wavelet coefficients of the i-th longitude point at time step j
        
        Notes
        -----
        The normalization follows Equation (2) in paper:
        G(k,ω) = (2π)^(-3/2) ∫[0→2π]dx ∫[-∞→∞]g(x,t)e^{-ikx}e^{-iωt}dt
        where G(k,ω) is the Fourier transform of g(x,t)
        """
        # 1.1 Initialize arrays (corresponding to FORTRAN allocate)
        cdata = np.zeros((self.params.Nx, self.params.Npad), dtype=np.complex64)
        cdaughter = np.zeros(self.params.Npad, dtype=np.complex64)
        daughter = np.zeros(self.params.Npad, dtype=np.complex128)
        fwave = np.zeros(self.params.Npad, dtype=np.float64)

        # 1.2 Initialize output array
        cfwt = np.zeros((self.params.Nx, self.params.Npad), dtype=np.complex64)

        # 1.3 Setup frequencies grids for wavelet
        freq1 = np.float64(2.0) * self.params.pi / (self.params.Npad * self.params.dt) # fundamental frequency in Fourier
        # print(f'freq1: {freq1}')
        fwave = np.fft.fftfreq(self.params.Npad, d=self.params.dt) * np.float64(2.0) * self.params.pi
        
        # 2 Compute G(k,w). 2D FFT to ori data
        # 2.1 Remove time-average for each longitude
        cdata[:, :self.params.Nt] = data_xt - data_xt.mean(axis=1, keepdims=True)
        # print(f'\nMax ori_data in single_cal: {np.max(cdata)}, Min ori_data in single_cal: {np.min(cdata)}')
        # print(f'\nMax remove_mean data in single_cal: {np.max(cdata)}, Min remove_mean data in single_cal: {np.min(cdata)}')

        # 2.2.1 Compute 2D FFT to get G(k,w)
        cdata = self._cfftf_xt(cdata)
        # print(f'\nMax FFT data in single_cal: {np.max(cdata)}, Min FFT data in single_cal: {np.min(cdata)}')

        # 2.2.2 Normalize G(k,w)
        cdata = cdata / (np.sqrt(2*self.params.pi) * self.params.Nx) * self.params.dt
        # print(f'\nMax FFT data after norm in single_cal: {np.max(cdata)}, Min FFT data after norm in single_cal: {np.min(cdata)}')

        # 3 Compute the character of Morlet wavelet in frequency domain ψ^*(sω)
        # (using double precision for higher accuracy)
        # print(f'\ndaughter cal: scale1 {scale1}, w0 {self.params.w0}')
        daughter = np.exp(-0.5 * (scale1 * fwave - self.params.w0) ** 2) # exp(−1/2(w0-w)**2)
        # print(f'\ndaugter in single_cal: max {np.max(daughter)}, min {np.min(daughter)}')
        
        # 4 Integration to get T{g}(k,s,τ)
        # Compute for positve & negative wavenumbers, separately
        # Compute wavelet transform for each longitude
        for i in range(self.params.Nx):
            # Multiply FFT with wavelet and normalize
            cdaughter = np.sqrt(abs(scale1)) * cdata[i,:] * daughter # |s|^(1/2)
            # print(f"After multiply: max {np.max((cdaughter))}, min {np.min((cdaughter))}")
            # Inverse FFT e^{-iωτ}
            cfwt[i,:] = np.fft.ifft(cdaughter, norm=None) * freq1 * self.params.Npad
            # cfwt[i,:] = np.fft.fft(cdaughter, norm='backward') * freq1 #Qin Modified
            # print(f"After IFFT: max {np.max(np.abs(cfwt))}, min {np.min(np.abs(cfwt))}")

        return cfwt

    # main calculation function
    def compute_cfwt(self, data: np.ndarray, COI=True, local_spectrum=False) -> tuple:
        """
        Main CFWT computation following Section 2.1 of paper.
        
        Implements full CFWT analysis including:
        1. Symmetric/antisymmetric decomposition (Section 2.1)
        2. Scale-dependent spectrum computation (Equation 9)
        3. Global spectrum averaging outside cone of influence
        4. Conversion from scale to frequency domain

        Parameters
        ----------
        data : np.ndarray, shape (Nx, Ny, Nt)
            Input data array with dimensions:
            - Nx: number of longitude points
            - Ny: number of latitude points 
            - Nt: number of time points

        Returns
        -------
        tuple of np.ndarray:
            - power_sym_global_frq : Global symmetric spectrum (Nx, Nfrq)
            - power_asym_global_frq : Global antisymmetric spectrum (Nx, Nfrq)
            - power_sym_frq : Local symmetric spectrum (Nx, Nfrq, Nt)
            - power_asym_frq : Local antisymmetric spectrum (Nx, Nfrq, Nt)

        Notes
        -----
        The power spectra follow Equations (9) and (10):
        - Local spectrum: P(k,s,t) = |T{g}(k,s,t)|²
        - Global spectrum: P(k,s) = ∫[t1→t2] P(k,s,τ)/(t2-t1) dτ
        """
        # 0 Check the data and update the dimensions in the settings
        if data.ndim != 3:
            raise ValueError(f"expect (lon, lat, time), got shape={data.shape}")
        Nx0, Ny0, Nt0 = map(int, data.shape)
        if (Nx0 != self.params.Nx) or (Ny0 != self.params.Ny) or (Nt0 != self.params.Nt):
            # Key: Update params with the shape of the input data and rebuild Npad/df/Nfrq/frq
            self.params.update_grid(Nx0, Ny0, Nt0)
        data = np.asarray(data, dtype=np.float32, order="C")
        Nx, Ny, Nt = self.params.Nx, self.params.Ny, self.params.Nt
        Jmax = self.params.Jmax
        s_pos = (self.params.s0 * (2.0 ** (self.params.dj * np.arange(Jmax + 1, dtype=np.float64)))).astype(np.float64)

        # 0.1 Pre-synthesis: Equatorial symmetric/antisymmetric components
        # Note: The Fortran version assumes that the input is "symmetric components in the southern hemisphere,antisymmetric components in the northern hemisphere."
        # Here, the equivalent synthesis is performed internally in Python: symmetric = ½(N+S), antisymmetric = ½(N−S), ignoring the equatorial strip (when Ny is odd).
        half = Ny // 2
        has_equator = (Ny % 2 == 1)
        # Only pairs of latitudes (half) are kept to align with Fortran's "South = symmetric North = antisymmetric" accumulation logic
        data_sym  = np.zeros((Nx, half, Nt), dtype=np.float32)
        data_asym = np.zeros_like(data_sym)
        for j in range(half):
            jS = half - 1 - j
            jN = half + (1 if has_equator else 0) + j
            data_sym[:, j, :]  = 0.5 * (data[:, jN, :] + data[:, jS, :])
            data_asym[:, j, :] = 0.5 * (data[:, jN, :] - data[:, jS, :])

        # 1 Preallocation
        power_sym = np.zeros((Nx, Jmax + 1, Nt), dtype=np.float32)
        power_asym = np.zeros_like(power_sym)
        power_sym_global = np.zeros((Nx, Jmax + 1), dtype=np.float32)
        power_asym_global = np.zeros_like(power_sym_global)
        power_sym_global_frq = np.zeros((Nx, self.params.Nfrq), dtype=np.float32)
        power_asym_global_frq = np.zeros_like(power_sym_global_frq)
        if local_spectrum:
            power_sym_frq = np.zeros((Nx, self.params.Nfrq, Nt), dtype=np.float32)
            power_asym_frq = np.zeros_like(power_sym_frq)
            
        # 2 Main loop (scale by scale)
        for js in range(Jmax + 1):
            s = float(s_pos[js])
        
            # Symmetrical components
            for j in range(half):
                cfwt = self._compute_cfwt_single(data_sym[:, j, :], s)
                p = np.real(cfwt[:, :Nt] * cfwt[:, :Nt].conj())   # 形状 (Nx, Nt)
                power_sym[:, js, :] += p
            
            # Antisymmetric component
            for j in range(half):
                cfwt = self._compute_cfwt_single(data_asym[:, j, :], s)
                p = np.real(cfwt[:, :Nt] * cfwt[:, :Nt].conj())
                power_asym[:, js, :] += p

        # 3 Cone of Influence to minimum boundary effect
        use_coi = COI  # 兼容原参数名
        for js in range(Jmax + 1):
            coi_radius = np.sqrt(2.0) * s_pos[js] 
            Npoints = 0
            for n in range(Nt):
                keep = True
                if use_coi:
                    t_left  = (n + 1) * self.params.dt
                    t_right = (Nt - (n + 1)) * self.params.dt
                    keep = (t_left > coi_radius) and (t_right > coi_radius)
                if keep:
                    power_sym_global[:, js]  += power_sym[:, js, n]
                    power_asym_global[:, js] += power_asym[:, js, n]
                    Npoints += 1
            if Npoints > 0:
                power_sym_global[:, js]  /= Npoints
                power_asym_global[:, js] /= Npoints
        
        print(f'\nMin sym cfwt global before spec_scale2frq:{np.min(power_sym_global)}, Max sym cfwt global before spec_scale2frq:{np.max(power_sym_global)}')
        print(f'\nMin asym cfwt global before spec_scale2frq:{np.min(power_asym_global)}, Max asym cfwt global before spec_scale2frq:{np.max(power_asym_global)}')

        # 4 Convert to frequency domain
        power_sym_global_frq  = self.spec_scale2frq(power_sym_global,  Jmax)
        power_asym_global_frq = self.spec_scale2frq(power_asym_global, Jmax)
        
        print(f'\nMin sym global frq after spec_scale2frq:{np.min(power_sym_global_frq)}, Max sym global frq after spec_scale2frq:{np.max(power_sym_global_frq)}')
        print(f'\nMin asym global frq after spec_scale2frq:{np.min(power_asym_global_frq)}, Max asym global frq after spec_scale2frq:{np.max(power_asym_global_frq)}')
    
        if local_spectrum:
            power_sym_frq  = np.zeros((Nx, self.params.Nfrq, Nt), dtype=np.float32)
            power_asym_frq = np.zeros_like(power_sym_frq)
            for n in range(Nt):
                power_sym_frq[:,  :, n] = self.spec_scale2frq(power_sym[:,  :, n], Jmax)
                power_asym_frq[:, :, n] = self.spec_scale2frq(power_asym[:, :, n], Jmax)
            return power_sym_global_frq, power_asym_global_frq, power_sym_frq, power_asym_frq
        else:
            return power_sym_global_frq, power_asym_global_frq
    
    # Assistant functions
    def _cfftf_xt(self, cdata: np.ndarray) -> np.ndarray:
        """
        Compute 2D Fast Fourier Transform matching paper implementation.

        Implements the spatial-temporal Fourier transform required for
        CFWT calculation in Equation (1).

        The transform is applied in sequence:
        1. Along longitude (space)
        2. Along time

        Parameters
        ----------
        cdata : np.ndarray, shape (Nx, Npad)
            Input complex data array, pre-padded to Npad length

        Returns
        -------
        np.ndarray, shape (Nx, Npad)
            FFT transformed data G(k,ω)

        Notes
        -----
        Part of G(k,ω) calculation in Section 2.1:
        G(k,ω) = (2π)^(-3/2) ∫∫g(x,t)e^{-ikx}e^{-iωt}dxdt
        """
        Nx, Npad = cdata.shape

        # Create working copy to avoid modifying input
        cdata_fft = cdata.copy()

        # First FFT along longitude (for each time point)
        # for n in range(Npad):
        #     cdata_fft[:, n] = np.fft.fft(cdata[:, n], norm=None)
            # Modified Nyquist
            # cdata_fft[Nx//2, n] = cdata_fft[Nx//2, n] / 2.0
        cdata_fft = np.fft.fft(cdata_fft, axis=0)
        # cdata_fft = np.fft.fft(cdata_fft, axis=0, norm="forward")*self.params.Nx #Qin 1->0

        # Then FFT along time (for each longitude)
        # for i in range(Nx):
        #     cdata_fft[i, :] = np.fft.fft(cdata_fft[i, :], norm=None)
            # Modified Nyquist
            # cdata_fft[i, Npad//2] = cdata_fft[i, Npad//2] / 2.0
        cdata_fft = np.fft.fft(cdata_fft, axis=1)
        # cdata_fft = np.fft.fft(cdata_fft, axis=1,norm="forward")*self.params.Npad #Qin 0->1
        
        return cdata_fft
    
    def arrange_array(self, x: np.ndarray, idim: int, jdim: int, use_fftshift: bool = True) -> np.ndarray:
        """
        Rearrange array for wavenumber ordering.

        Reorders wavenumbers from [0,1,...,N] to [-N/2,...,0,...,N/2]
        format as required for spectral analysis in Section 2.2.

        Parameters
        ----------
        x : np.ndarray, shape (idim, jdim)
            Input array to be rearranged
        idim : int
            First dimension size (typically Nx)
        jdim : int
            Second dimension size (typically Jmax+1 or Nfrq)

        Returns
        -------
        np.ndarray, shape (idim, jdim)
            Rearranged array with centered wavenumber ordering

        Notes
        -----
        This reordering is necessary for proper power spectrum
        representation as discussed in paper Appendix A.
        """
        x = np.asarray(x)
        if use_fftshift:
            return np.fft.fftshift(x, axes=0).astype(np.float32, copy=False)
        else:
            return x.astype(np.float32, copy=False)
    
    def spec_scale2frq(self, power_scale: np.ndarray, Jmax: int) -> np.ndarray:
        """
        Convert power spectrum from wavelet scale to frequency domain.

        Implements the scale-to-frequency conversion described in 
        Section 2.2 using the relationship from Equation (5):
        s = (ω0 + (ω0² + 2)^(1/2))/(4πf)

        Parameters
        ----------
        power_scale : np.ndarray, shape (Nx, Jmax+1)
            Power spectrum in wavelet scale domain
        Jmax : int
            Maximum scale index

        Returns
        -------
        np.ndarray, shape (Nx, Nfrq)
            Power spectrum in frequency domain
        
        Notes
        -----
        The conversion preserves the total energy following
        Parseval's relation (Equation 8)
        """
        Nx = self.params.Nx
        Nf = self.params.Nfrq
        power_frq = np.zeros((Nx, Nf), dtype=np.float32)
    
        # Positive scale (monotonically increasing, J+1; consistent with that used in compute_cfwt)
        s_pos = (self.params.s0 * (2.0 ** (self.params.dj * np.arange(Jmax + 1, dtype=np.float64)))).astype(np.float64)
    
        for i in range(Nx):
            power_frq[i, :] = self._spec_scale2frq_i(power_scale[i, :], s_pos)
    
        return power_frq
    
    def _spec_scale2frq_i(self, power_scale_1k: np.ndarray, s_pos: np.ndarray) -> np.ndarray:
        """
        Convert power spectrum from scale to frequency domain for single longitude.

        Implements the scale-to-frequency conversion detailed in
        Appendix A. Uses the wavelet scale-frequency relationship:
        s = (ω0 + sqrt(ω0² + 2))/(4πf)

        Parameters
        ----------
        power_scale : np.ndarray, shape (Jmax+1,)
            Power spectrum in scale domain for one longitude
        Jmax : int
            Maximum scale index

        Returns
        -------
        np.ndarray, shape (Nfrq,)
            Power spectrum in frequency domain

        Notes
        -----
        Conversion preserves total power as required by
        energy conservation (Equation 8).
        Uses linear interpolation between scale points.
        """
        Nf = self.params.Nfrq
        df = float(self.params.df)
        frq = self.params.frq
        w0  = float(self.params.w0)
        const = (w0 + np.sqrt(w0*w0 + 2.0)) / (4.0 * np.pi)
    
        J = len(s_pos) - 1
        power_frq_1k = np.zeros(Nf, dtype=np.float64)
        # Normalization: According to 1/s^2 (same as Fortran/paper)
        p_norm = np.zeros(J + 1, dtype=np.float64)
        for j in range(J + 1):
            sj = float(s_pos[j])
            p_norm[j] = (float(power_scale_1k[j]) / (sj*sj)) if sj > 0 else 0.0
        
        # n=0 assumes missing measurement
        power_frq_1k[0] = self.params.rmiss
        # Integrate for each frequency band (n=1..Nf-1)
        for n in range(1, Nf):
            f_center = float(frq[n])
            f_lo = max(f_center - 0.5*df, 1e-12)   # Prevent 0 frequency
            f_hi = f_center + 0.5*df
    
            # Frequency to scale interval: [s_lo, s_hi] = [const/f_hi, const/f_lo], note s_lo < s_hi
            s_lo = const / f_hi      # small scale
            s_hi = const / f_lo      # large scale
    
            # Traverse each scale segment [s_j, s_{j+1}]
            acc = 0.0
            for j in range(J):
                s_j  = s_pos[j]
                s_j1 = s_pos[j+1]
                # Overlap with [s_lo, s_hi]
                sa = max(s_lo, s_j)
                sb = min(s_hi, s_j1)
                if sb <= sa:
                    continue
                # Linear interpolation p(sa), p(sb)
                pj  = p_norm[j]
                pj1 = p_norm[j+1]
                slope = (pj1 - pj) / (s_j1 - s_j) if s_j1 > s_j else 0.0
                p_sa = pj + slope * (sa - s_j)
                p_sb = pj + slope * (sb - s_j)
                acc += 0.5 * (p_sa + p_sb) * (sb - sa)
    
            # Cψ is normalized; if you want to get the "density", you can add /df (only necessary when aligning with the WK caliber)
            power_frq_1k[n] = acc / float(self.params.Cpsi)
            # power_frq_1k[n] /= df  # Optional
    
        return power_frq_1k.astype(np.float32)

#------------------------------------------------------------------------------
# Part 4: CFWT Method_Reconstruction & Function Checks
#------------------------------------------------------------------------------

"""
Part 4 Reconstruction & Function Checks.

    4.1 _reconstruct: core function to reconstruct physical fields
    4.2 check_reconstruction: main function to reconstruct and check energy conservation
    4.3 check_energy: core function to check energy conservation

Output: Reconstructed physical field and the results of energy
"""

class CFWTValidation:
    """
    Validation class implementing verification procedures from Section 2.2.

    Provides methods to verify:
    1. Signal reconstruction accuracy
    2. Energy conservation
    3. Scale-frequency conversion accuracy

    These tests are described in paper Section 3.1.
    """
    def __init__(self, params: CFWTParams, computation: CFWTComputation):
        """
        Initialize validation class
        
        Parameters
        ----------
        params : CFWTParams
            Parameter object containing all CFWT parameters
        computation : CFWTComputation
            Computation object for CFWT calculations
        """
        self.params = params
        self.computation = computation
    
    def _reconstruct(self, cfwt: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal from CFWT coefficients

        Parameters
        ----------
        cfwt : np.ndarray
            CFWT coefficients (Nx, 0:2*Jmax+1/(-Jmax, Jmax), Npad)

        Returns
        -------
        np.ndarray
            Reconstructed signal (Nx, Nt)
        """
        Nx, Jjp1, Npad = cfwt.shape
        assert Jjp1 == len(self.params.scale) == (2*self.params.Jmax + 1), \
            f"cfwt scale-dim={Jlen}, but expected 2*Jmax+1={2*self.params.Jmax+1}"
        w = np.log(2.0) * self.params.dj / float(self.params.Cdelta)
        recon_k_tau = np.zeros((Nx, Npad), dtype=np.complex64)
        for j in range(-self.params.Jmax, self.params.Jmax + 1):
            s = abs(float(self.params.scale[j + self.params.Jmax]))
            if s > 0:
                recon_k_tau += (cfwt[:, j+self.params.Jmax, :] / np.sqrt(s, dtype=np.float64)).astype(np.complex64)

        recon_k_tau *= w

        # For each τ, do IFFT along k → x (longitude)
        recon = np.zeros((Nx, self.params.Nt), dtype=np.float32)
        for n in range(self.params.Nt):
            # Note: Frequency/time domain normalization selection for _compute_cfwt_single
            # Here we keep it consistent: k-axis IFFT with default normalization, then multiply by Nx
            recon[:, n] = np.real(np.fft.ifft(recon_k_tau[:, n])) * self.params.Nx

        return recon
    
    def check_reconstruction(self, data_xt_in: np.ndarray) -> tuple:
        """
        Verify signal reconstruction from CFWT coefficients.

        Implements reconstruction test following Equation (6):
        g(x,t) = (1/Cdelta) * sum[T{g}(k,s,t)/sqrt(s)] * dj/log(2)

        Parameters
        ----------
        data_xt_in : np.ndarray, shape (Nx, Nt)
            Original signal in longitude-time section

        Returns
        -------
        tuple (data_xt, recon)
            Original and reconstructed signals for comparison
        """
        Nx, Nt = data_xt_in.shape
        assert Nx == self.params.Nx and Nt == self.params.Nt
        J = self.params.Jmax

        # Remove the time mean to keep it consistent with the main process
        data_xt = data_xt_in.copy().astype(np.float32)
        data_xt -= np.nanmean(data_xt, axis=1, keepdims=True)

        # Calculate wavelet coefficients at all scales
        nscale = len(self.params.scale)
        cfwt = np.zeros((Nx, nscale, self.params.Npad), dtype=np.complex64)
        for j in range(-J, J+1):
            s = float(self.params.scale[j+J])
            cfwt[:, j+J, :] = self.computation._compute_cfwt_single(data_xt, s)

        # Reconstruction
        recon = self._reconstruct(cfwt)

        # Physical Domain Energy
        energy_phys = float(np.sum(np.real(data_xt) ** 2))
        # Wavelet domain energy (Parseval): (log2 * dj / Cpsi) * ∑ |T|^2 / s^2, time only integrated to Nt
        log2dj_over_Cpsi = np.log(2.0) * self.params.dj / float(self.params.Cpsi)
        energy_cfwt = 0.0
        for j in range(J + 1):
            s = float(self.params.scale[j])
            if s <= 0:
                continue
            # Only use 0..Nt-1 (consistent with the main process, not including padding)
            energy_cfwt += np.sum(np.abs(cfwt[:, j, :Nt]) ** 2) / (s * s)
        energy_cfwt *= log2dj_over_Cpsi

        # Reconstructing Energy
        recon_phys = float(np.sum(np.real(recon) ** 2))

        diff_percent = 0.0 if energy_phys == 0 else (energy_cfwt - energy_phys) / energy_phys * 100.0
        recon_diff_percent = 0.0 if energy_phys == 0 else (recon_phys - energy_phys) / energy_phys * 100.0
        
        print("\n Energy conservation in the process of CFWT and Reconstruction:")
        print("CFWT coefficients range:", np.min(np.abs(cfwt)), np.max(np.abs(cfwt)))
        print("Scale values:", self.params.scale)
        print(f"Physical space energy: {energy_phys:.2e}")
        print(f"CFWT space energy: {energy_cfwt:.2e}")
        print(f"Reconstructed space energy: {recon_phys:.2e}")
        print(f"CFWT space Difference: {diff_percent:.2f}%")
        print(f"Recon Difference: {recon_diff_percent:.2f}%")
        if abs(diff_percent) > 1.0:  # allow 1% bias
            print("Warning: Energy conservation in CFWT exceeds 1% threshold!")

        return data_xt, recon
    
    def check_energy(self, data_xt: np.ndarray) -> tuple:
        """
        Verify energy conservation in CFWT transform.

        Implements energy conservation test following Equation (8):
        (1/Cpsi)∫(1/s²)ds∫|T{g}(k,s,t)|²dt = (2*pi)*∫∫|g(x,t)|²dt 
        = 2*pi*Σ(k=-∞ to ∞)∫|G(k,w)|²dw

        Parameters
        ----------
        data_xt : np.ndarray, shape (Nx, Nt)
            Input signal in longitude-time section

        Returns
        -------
        tuple (energy_phys, energy_cfwt, diff_percent)
            Physical space energy, CFWT space energy, 
            and percentage difference
        """
        # Initialize arrays matching FORTRAN declarations
        Nx, Nt = data_xt.shape
        assert Nx == self.params.Nx and Nt == self.params.Nt

        # Remove the mean
        x = data_xt.astype(np.float32).copy()
        x -= np.nanmean(x, axis=1, keepdims=True)

        energy_phys = float(np.sum(x ** 2))

        # Compute coefficients and perform Parseval integration
        J = self.params.Jmax
        log2dj_over_Cpsi = np.log(2.0) * self.params.dj / float(self.params.Cpsi)

        energy_cfwt = 0.0
        for j in range(J + 1):
            s = float(self.params.scale[j])
            if s <= 0:
                continue
            cfwt = self.computation._compute_cfwt_single(x, s)   # (Nx, Npad)
            energy_cfwt += np.sum(np.abs(cfwt[:, :Nt]) ** 2) / (s * s)

        energy_cfwt *= log2dj_over_Cpsi

        diff_percent = 0.0 if energy_phys == 0 else (energy_cfwt - energy_phys) / energy_phys * 100.0
        return energy_phys, energy_cfwt, diff_percent


#------------------------------------------------------------------------------
# Part 5: CFWT Method_Filter & Reconstruction.
#------------------------------------------------------------------------------

"""
Part 5 CCEWs Filter & Reconstruction.

    5.1 energy_calculation: check energy conseration
    5.2 filter_reconstruction: core function to filter and reconstruct CCEWs
    5.3 filter_multiple_waves: main function to process CCEWs filter and reconstruction
    5.4 save_results_to_netcdf: result saving

Output: Original field and reconstructed CCEWs' physical field
"""

class CCEWFilter:
    """
    1. Select a specific wave mode in the frequency-wavenumber domain
    2. Distinguish between symmetric and antisymmetric components
    3. Separately process eastward and westward waves
    4. Reconstruct the filtered physical field

    The filtering follows the standard conventions of CCEW analysis:
    - Kelvin waves: symmetric component, k=1-14, period=2.5-30 days, eastward
    - Equatorial Rossby waves (ER waves): symmetric component, k=-10--1, period=9-72 days, westward
    - Hybrid Rossby Gravity waves (MRG waves): antisymmetric component, k=-10--1, period=3-9 days, westward
    - MJO: symmetric component, k=1-5, period=30-96 days, eastward
    - IG0 waves: symmetric component, k=-15-15, period=2.5-3.5 days, Eastward and westward
    - IG1 wave: antisymmetric component, k=-15-15, period=2-2.5 days, eastward and westward
    - IG2 wave: symmetric component, k=-15-15, period=1.5-2 days, eastward and westward
    - Tropical cyclone disturbance (TD): symmetric component, k=-20--6, period=2-5 days, westward
    """

    def __init__(self, params: CFWTParams, computation: CFWTComputation, validation:CFWTValidation):
        """
        Initialize CCEW filter
        
        Parameters
        ----------
        params : CFWTParams
            CFWT parameter object
        computation : CFWTComputation
            CFWT computation object
        """
        self.params = params
        self.computation = computation
        self.validation = validation

        # Standard filter ranges for common wave types if not custom
        self.wave_properties = {
            'kelvin': {'k_min': 1, 'k_max': 14, 'T_min': 2.5, 'T_max': 20, 'symmetry': 'symmetric', 'direction': 'eastward'},
            'er': {'k_min': 1, 'k_max': 10, 'T_min': 9, 'T_max': 72, 'symmetry': 'symmetric', 'direction': 'westward'},
            'mrg': {'k_min': 1, 'k_max': 10, 'T_min': 3, 'T_max': 9, 'symmetry': 'antisymmetric', 'direction': 'westward'},
            'mjo': {'k_min': 1, 'k_max': 5, 'T_min': 30, 'T_max': 96, 'symmetry': 'symmetric', 'direction': 'eastward'},
            'ig0': {'k_min': 1, 'k_max': 15, 'T_min': 2.5, 'T_max': 3.5, 'symmetry': 'antisymmetric', 'direction': 'eastward'},
            'ig1': {'k_min': 1, 'k_max': 15, 'T_min': 2.0, 'T_max': 2.5, 'symmetry': 'symmetric', 'direction': 'both'},
            'ig2': {'k_min': 1, 'k_max': 15, 'T_min': 1.5, 'T_max': 2.0, 'symmetry': 'antisymmetric', 'direction': 'both'},
            'td': {'k_min': 6, 'k_max': 20, 'T_min': 2.0, 'T_max': 5.0, 'symmetry': 'symmetric', 'direction': 'westward'},
        }
        
    def _sync_params_with_data(self, data_3d: np.ndarray):
        Nx0, Ny0, Nt0 = map(int, data_3d.shape)
        if (Nx0 != self.params.Nx) or (Ny0 != self.params.Ny) or (Nt0 != self.params.Nt):
            # Update Nx/Ny/Nt, Npad/df/Nfrq/frq according to data dimensions
            self.params.update_grid(Nx0, Ny0, Nt0)
        # Make the latitudinal cycle range completely consistent with the data
        self.params.jlats = 0
        self.params.jlatn = Ny0 - 1

    def _get_positive_scales(self) -> np.ndarray:
        """
        Extract Jmax+1 strictly increasing positive scales from the global signed scale array.
        Compatibility: If Jmax+1 positive scales are still given elsewhere, this function will also return the same result.
        """
        J = int(self.params.Jmax)
        s = np.asarray(self.params.scale, dtype=np.float64)
        if s.size == 2*J + 1:
            s_pos = np.abs(s[J:J+J+1])            # The right half contains s0 and all positive scales
        else:
            s_pos = np.abs(s[:J+1])
        # Conservative: Ensure strict increment
        s_pos = np.asarray(s_pos, dtype=np.float64)
        if not np.all(np.diff(s_pos) > 0):
            s_pos = np.sort(np.unique(s_pos))
            assert s_pos.size >= J+1, "The number of positive scales is insufficient to cover Jmax+1"
            s_pos = s_pos[:J+1]
        return s_pos

    # Parseval-caliber CFWT energy (aligned with positive scale implementation)
    def energy_calculation(self, cfwt: np.ndarray) -> float:
        """
        cfwt: (Nx, Jmax+1, Npad) or (Nx, Ny, Jmax+1, Npad)
        Integrate only over time samples 0..Nt-1
        """
        log2dj_over_Cpsi = np.log(2.0) * self.params.dj / float(self.params.Cpsi)
        Nt = self.params.Nt
        if cfwt.ndim == 3:
            # (Nx, J+1, Npad)
            s2 = (self.params.scale[np.newaxis, :, np.newaxis] ** 2).astype(np.float64)
            val = np.sum(np.abs(cfwt[:, :, :Nt])**2 / s2, dtype=np.float64)
        elif cfwt.ndim == 4:
            # (Nx, Ny, J+1, Npad)
            s2 = (self.params.scale[np.newaxis, np.newaxis, :, np.newaxis] ** 2).astype(np.float64)
            val = np.sum(np.abs(cfwt[:, :, :, :Nt])**2 / s2, dtype=np.float64)
        else:
            raise ValueError(f"Unexpected cfwt ndim={cfwt.ndim}")
        return float(log2dj_over_Cpsi * val)

    # 2D composite version (returns the original/reconstructed latitudinal mean), avoiding "blanks"
    def filter_reconstruction(self, data_xt_in: np.ndarray, wave_type: str, 
                                      eq_lat_idx: int, custom_range: dict = None, 
                                      direction: str = None, symmetry: str = None) -> tuple:
        """
        Enhanced filter and reconstruction function that properly handles wave direction.
        
        Parameters
        ----------
        data_xt_in : np.ndarray
            Input data array (Nx, Ny, Nt)
        wave_type : str
            Wave type to filter ('kelvin', 'er', 'mrg', 'mjo', 'ig0', 'ig1', 'ig2', 'td', 'custom')
        custom_range : dict, optional
            Custom filter range with keys: k_min, k_max, T_min, T_max
        direction : str, optional
            Force specific wave direction ('eastward', 'westward', 'both')
        symmetry : str, optional
            Force specific wave symmetry ('symmetric', 'antisymmetric')
        
        Returns
        -------
        tuple
            (original data, filtered data)
        """
        self._sync_params_with_data(data_xt_in)
        Nx, Ny, Nt = self.params.Nx, self.params.Ny, self.params.Nt
        assert data_xt_in.shape == (Nx, Ny, Nt)
        
        # 1. Get wave properties
        if wave_type.lower() == 'custom' and custom_range is not None:
            props = {'symmetry': symmetry or 'symmetric', 'direction': direction or 'both',
                     'k_min': custom_range['k_min'], 'k_max': custom_range['k_max'],
                     'T_min': custom_range['T_min'], 'T_max': custom_range['T_max']}
        else:
            props = dict(self.wave_properties[wave_type.lower()])
            if symmetry:  props['symmetry']  = symmetry
            if direction: props['direction'] = direction
        
        # Generate a "positive and increasing" local scale grid from a signed scale
        s_all = np.asarray(self.params.scale, dtype=np.float64)
        s_pos = np.sort(np.unique(np.abs(s_all[s_all != 0.0])))
        J = len(s_pos) - 1
        assert s_pos.ndim == 1 and len(s_pos) == (J+1)
    
        # 2 Pre-allocation
        recon_k_tau_accum = np.zeros((Nx, self.params.Npad), dtype=np.complex64)
    
        # k axis and direction (consistent with the FFT direction of compute_cfwt)
        k_bins = np.fft.fftfreq(Nx) * Nx                    # 0..+..,-.. order
        # In the current implementation: k<0 is considered an eastward pass; k>0 is considered a westward pass (if you have changed the global number, please change these two lines at the same time)
        is_east = (k_bins < 0)
        is_west = (k_bins > 0)
    
        # 3 Period range
        w0 = float(self.params.w0); const = (w0 + np.sqrt(w0*w0 + 2.0)) / (4.0*np.pi)
        def period_from_scale(s):    # s>0
            f = const / float(s)
            return np.inf if f <= 0 else 1.0/f
        T_min, T_max = float(props['T_min']), float(props['T_max'])
    
        # 4 Weight (constant factor in the reconstruction formula)
        w_scale = (np.log(2.0) * float(self.params.dj)) / float(self.params.Cdelta)
    
        # 5 Latent pairing (ignoring the equatorial strip)
        half = Ny // 2
        has_eq = (Ny % 2 == 1)
    
        for jpair in range(half):
            jS = half - 1 - jpair
            jN = half + (1 if has_eq else 0) + jpair
    
            # (lon,time) of two latitudes
            xt_N = data_xt_in[:, jN, :]
            xt_S = data_xt_in[:, jS, :]
    
            # Scale by scale
            for js in range(J + 1):
                s = float(s_pos[js])
                T = period_from_scale(s)
                if not (T_min <= T <= T_max):
                    continue
    
                # Calculate the CFWT of two parallels (Nx, Npad)
                cfwt_N = self.computation._compute_cfwt_single(xt_N, s)
                cfwt_S = self.computation._compute_cfwt_single(xt_S, s)
    
                # Symmetric/antisymmetric
                if props['symmetry'].lower() == 'symmetric':
                    cfwt_pair = 0.5 * (cfwt_N + cfwt_S)
                elif props['symmetry'].lower() == 'antisymmetric':
                    cfwt_pair = 0.5 * (cfwt_N - cfwt_S)
                else:  # both
                    cfwt_pair = cfwt_N  # Then both East and West will be taken, which is equivalent to the sum of N and S. If strict both is required, 0.5*(N+S)+0.5*(N-S)=N
                    cfwt_pair += 0.0    # (Avoid warnings about unused variables)
    
                # Directional filtering with |k|
                keep = np.ones(Nx, dtype=bool)
                if props['direction'].lower() == 'eastward':
                    keep &= is_east
                elif props['direction'].lower() == 'westward':
                    keep &= is_west
    
                kmin, kmax = int(props['k_min']), int(props['k_max'])
                keep &= (np.abs(k_bins) >= kmin) & (np.abs(k_bins) <= kmax)
    
                # Half spectrum ×2 compensation (k=0 and Nyquist not multiplied by 2)
                hemi_gain = np.ones(Nx, dtype=np.float32)
                if props['direction'].lower() in ('eastward', 'westward'):
                    hemi_gain[keep] = 2.0
                    hemi_gain[k_bins == 0] = 1.0
                    if Nx % 2 == 0:
                        hemi_gain[Nx//2] = 1.0  # Nyquist
    
                # Accumulate (by 1/sqrt(s) with constant weight)
                recon_k_tau_accum[keep, :] += (hemi_gain[keep, None] *
                                               (cfwt_pair[keep, :] / np.sqrt(s, dtype=np.float64)) * w_scale).astype(np.complex64)
    
        # 6 k→x inverse transform and latitudinal averaging (number of pairs = half)
        filtered_latmean = np.zeros((Nx, Nt), dtype=np.float32)
        for n in range(Nt):
            filtered_latmean[:, n] = np.real(np.fft.ifft(recon_k_tau_accum[:, n])) * Nx
        if half > 0:
            filtered_latmean /= float(half)
    
        # Latent average of the original field (for side-by-side comparison)
        original_latmean = np.zeros((Nx, Nt), dtype=np.float32)
        for j in range(Ny):
            original_latmean += data_xt_in[:, j, :] / float(Ny)
    
        return original_latmean, filtered_latmean

    def filter_reconstruction_full3d(self, data_xt_in: np.ndarray, wave_type: str, 
                                     sym_asy=False, custom_range: dict = None, 
                                     direction: str = None, symmetry: str = None) -> np.ndarray:
        """
        Preserve the (lon, lat, time) fluctuation structure after filtering out specific fluctuations.
        Input: data_xt_in shape = (lon, lat, time)
        Return: filtered_field shape = (lon, lat, time)
        """
        # 0 Basic parameters
        self._sync_params_with_data(data_xt_in)
        Nx, Ny, Nt = self.params.Nx, self.params.Ny, self.params.Nt
        assert data_xt_in.shape == (Nx, Ny, Nt)

        s_all = np.asarray(self.params.scale, dtype=np.float64)
        s_pos = np.sort(np.unique(np.abs(s_all[s_all != 0.0])))
        J = len(s_pos) - 1
        assert s_pos.ndim == 1 and len(s_pos) == (J+1)
    
        # Select waveform parameters
        if wave_type.lower() == 'custom' and custom_range is not None:
            cfg = custom_range.copy()
            cfg.setdefault('direction', direction or 'both')
            cfg.setdefault('symmetry',  symmetry or 'both')
        else:
            cfg = self.wave_properties[wave_type.lower()].copy()
            if direction is not None: cfg['direction'] = direction
            if symmetry  is not None: cfg['symmetry']  = symmetry
    
        kmin = int(cfg['k_min']); kmax = int(cfg['k_max'])
        Tmin = float(cfg['T_min']); Tmax = float(cfg['T_max'])
        want_dir = cfg['direction'].lower()      # 'eastward' | 'westward' | 'both'
        want_sym = cfg['symmetry'].lower()       # 'symmetric' | 'antisymmetric' | 'both'
    
        # k-axis index and positive/negative side
        k_fft = np.fft.fftfreq(Nx) * Nx
        pos = np.arange(1, Nx//2, dtype=int)     # "front and side"
        neg = Nx - pos                            # "negative side" (pair with pos)
    
        # Keep only the indices with |k| ∈ [kmin, kmax]
        mask_k = (np.abs(k_fft[pos]) >= kmin) & (np.abs(k_fft[pos]) <= kmax)
        pos_sel = pos[mask_k]
        neg_sel = neg[mask_k]
    
        # Morlet frequency constant (s->f)
        w0 = float(self.params.w0)
        const_sf = (w0 + np.sqrt(w0*w0 + 2.0)) / (4.0 * np.pi)
    
        # Accumulation container (first store the (k,τ) complex spectrum of each latitude, then unify the IFFT)
        recon_k_tau_lat = np.zeros((Ny, Nx, self.params.Npad), dtype=np.complex64)
    
        # 1 Filter and accumulate by dimension and scale
        # Direction convention: When k = fftfreq(Nx)*Nx, k<0 = eastward transmission; k>0 = westward transmission
        k = np.fft.fftfreq(Nx) * Nx
        K = np.abs(k).astype(int)
        
        # Effective k: Remove k=0 and Nyquist
        valid_k = (k != 0)
        if (Nx % 2) == 0:
            valid_k[Nx // 2] = False
        
        # Wavenumber window and direction window
        k_band = (K >= kmin) & (K <= kmax)
        if want_dir == 'eastward':
            dir_mask = (k < 0)
        elif want_dir == 'westward':
            dir_mask = (k > 0)
        else:  # 'both'
            dir_mask = np.ones_like(k, dtype=bool)
        
        # The final retained k masks (fixed, not changing with scale/latitude)
        k_keep = valid_k & k_band & dir_mask
        
        # Unified reconstruction weight
        w = np.log(2.0) * float(self.params.dj) / float(self.params.Cdelta)
        
        for jlat in range(Ny):
            xt = data_xt_in[:, jlat, :]  # (Nx, Nt)
        
            for j in range(J + 1):
                s = float(s_pos[j])                # Positive scale
                if s <= 0:
                    continue
        
                # Cycle determination
                f_equiv = const_sf / s             # cycles/day
                T = 1.0 / f_equiv if f_equiv > 0 else np.inf
                if not (Tmin <= T <= Tmax):
                    continue
        
                # CFWT of this scale: returns (Nx, Npad)
                cfwt = self.computation._compute_cfwt_single(xt, s)
        
                # Only the selected k are kept and added to (k,τ) with consistent weights
                if np.any(k_keep):
                    cf = (cfwt / np.sqrt(s)).astype(np.complex64)
                    cf[~k_keep, :] = 0.0
                    recon_k_tau_lat[jlat, :, :] += cf
        
        # Unified multiplication and reconstruction constant
        recon_k_tau_lat *= w
        
        # 2 k-axis IFFT → (lon,lat,time)
        out = np.zeros((Nx, Ny, Nt), dtype=np.float32)
        for jlat in range(Ny):
            for n in range(Nt):
                out[:, jlat, n] = np.real(np.fft.ifft(recon_k_tau_lat[jlat, :, n])) * Nx
                
        if sym_asy==False:
            out = out
        elif sym_asy==True:
            # 3 Optional: Do symmetric/antisymmetric projection in the "space domain"
            if want_sym in ('symmetric', 'antisymmetric'):
                half = Ny // 2
                has_eq = (Ny % 2 == 1)
                out_sym = np.zeros_like(out)
            
                if want_sym == 'symmetric':
                    for p in range(half):
                        jS = half - 1 - p
                        jN = half + (1 if has_eq else 0) + p
                        sym = 0.5 * (out[:, jN, :] + out[:, jS, :])
                        out_sym[:, jN, :] = sym
                        out_sym[:, jS, :] = sym
                    if has_eq:
                        out_sym[:, half, :] = out[:, half, :]
                else:  # 'antisymmetric'
                    for p in range(half):
                        jS = half - 1 - p
                        jN = half + (1 if has_eq else 0) + p
                        asym = 0.5 * (out[:, jN, :] - out[:, jS, :])
                        out_sym[:, jN, :] =  asym
                        out_sym[:, jS, :] = -asym
                    if has_eq:
                        out_sym[:, half, :] = 0.0
            
                out = out_sym
        
        return out

    def filter_multiple_waves(self, data_xt_in: np.ndarray, wave_types: list, eq_lat_idx: int) -> dict:
        """
        Filter and reconstruct multiple wave types.
        
        Parameters
        ----------
        data_xt_in: np.ndarray
        Input data array (Nx, Nt)
        wave_types: list
        List of wave types to filter
        
        Returns
        -------
        dict
        {wave type: filtered data} dictionary
        """
        result = {'original': data_xt_in.copy()}
        
        for wave_type in wave_types:
            print(f"\nProcessing {wave_type} ...")
            _, filtered = self.filter_reconstruction_updated(data_xt_in, wave_type, eq_lat_idx)
            result[wave_type] = filtered
            
        return result

    def save_results_to_netcdf(self, filtered_results, ds, outfile):
        """Save the filtering results to a netCDF file"""
        # Create a data variable dictionary
        data_vars = {'olr_original': (['time', 'lon'], ds['olr'].mean(dim='lat').values)}
        
        # Add each filter result
        for wave_type, filtered_data in filtered_results.items():
            if wave_type == 'original':
                continue
            data_vars[f'olr_{wave_type}'] = (['time', 'lon'], filtered_data.T)
        ds_filtered = xr.Dataset(
            data_vars=data_vars,
            coords={'time': ds.time, 'lon': ds.lon},
            attrs={'description': 'CCEWFilter OLR', 'creation_date': str(datetime.datetime.now())}
        )
        for wave_type in filtered_results.keys():
            if wave_type in ('original',):
                continue
            props = self.wave_properties.get(wave_type, None)
            if props:
                var_name = f'olr_{wave_type}'
                ds_filtered[var_name].attrs.update({
                    'long_name': f"{wave_type.upper()}_OLR",
                    'symmetry': props['symmetry'].capitalize(),
                    'direction': props['direction'].capitalize()
                })
        output_file = os.path.join(outfile)
        ds_filtered.to_netcdf(output_file)
        ds_filtered.close()
        print(f"Results saved to {output_file}")
        

#------------------------------------------------------------------------------
# Part 6: CFWT Method_Visualization
#------------------------------------------------------------------------------

"""
Part 6. Integrated CFWT (Combined Fourier-Wavelet Transform) Analysis Module

This module integrates CFWT computation, background spectrum calculation,
and visualization tools for climate data analysis.

    Main components:
    6.1 Core CFWT computation
    6.2 Background spectrum calculation
    6.3 Visualization of raw, background, and background-removed spectra
"""
# class CFWTVisualization:
#     """
#     Visualization functions for CFWT analysis results.
#     """
#     def __init__(self, params: CFWTParams, computation: CFWTComputation, validation:CFWTValidation):
#         """
#         Initialize CCEW filter
        
#         Parameters
#         ----------
#         params : CFWTParams
#             CFWT parameter object
#         computation : CFWTComputation
#             CFWT computation object
#         """
#         self.params = params
#         self.computation = computation
#         self.validation = validation
    
#     def plot_raw_spectrum(power_sym_global, power_asym_global, wavenumber, frequency, nlat=params.Ny,
#                         figsize=(18, 8), contour_range=(-2.0, -0.2 + 1e-9, 0.1)):
#         """
#         Plot raw CFWT spectrum before removing background.
        
#         Parameters:
#         -----------
#         power_sym_global : ndarray
#             Symmetric power spectrum array
#         power_asym_global : ndarray
#             Antisymmetric power spectrum array
#         wavenumber : ndarray
#             Wavenumber array
#         frequency : ndarray
#             Frequency array
#         nlat : int, optional
#             Number of latitudes (default: 40)
#         figsize : tuple, optional
#             Figure size (default: (18, 8))
#         contour_range : tuple, optional
#             Contour range (min, max, step) (default: (-1.8, 0, 0.2))
            
#         Returns:
#         --------
#         fig : Figure
#             Matplotlib figure
#         """
#         power_sym_global  = power_sym_global.astype('float64')
#         power_asym_global = power_asym_global.astype('float64')
#         power_sym_global[~np.isfinite(power_sym_global) | (power_sym_global <= 0)]   = np.nan
#         power_asym_global[~np.isfinite(power_asym_global) | (power_asym_global <= 0)] = np.nan
        
#         # 1. Intercept the wave number range from -15 to 15
#         wn_start = np.where(wavenumber == -15)[0][0] if -15 in wavenumber else 0
#         wn_end = np.where(wavenumber == 15)[0][0] + 1 if 15 in wavenumber else len(wavenumber)
#         wavenumber_subset = wavenumber[wn_start:wn_end]
#         power_sym_subset = power_sym_global[wn_start:wn_end, :]
#         power_asym_subset = power_asym_global[wn_start:wn_end, :]

#         mask_frequency = frequency > 0
#         frequency_subset = frequency[mask_frequency]
#         power_sym_subset = power_sym_subset[:, mask_frequency]      # (K, Fpos)
#         power_asym_subset = power_asym_subset[:, mask_frequency]
        
#         # 2. Adjusted
#         power_sym_rearranged = power_sym_subset.T
#         power_asym_rearranged = power_asym_subset.T

#         # Data preparation
#         psumsym_r  = np.log10(power_sym_rearranged)
#         psumanti_r = np.log10(power_asym_rearranged)

#         Zsym  = np.ma.masked_invalid(psumsym_r)
#         Zanti = np.ma.masked_invalid(psumanti_r)

#         cmap = plt.get_cmap('YlOrRd').copy()
#         cmap.set_bad('white')   # NaN -> white
#         cmap.set_under('white') # Color values below the minimum value white (optional)
#         line_levels = np.arange(-2.0, -0.2 + 1e-9, 0.1)
#         fill_levels = np.linspace(line_levels.min(), line_levels.max(), 80)
#         norm = mcolors.BoundaryNorm(line_levels, cmap.N, clip=False)
        
#         # Create figure
#         fig, axAS = plt.subplots(1, 2, figsize=figsize)
#         plt.subplots_adjust(bottom=0.2)
        
#         X, Y = np.meshgrid(wavenumber_subset, frequency_subset)
#         # Draw a graph (both graphs share the same norm/color scale)
#         fig, axAS = plt.subplots(1, 2, figsize=(18, 8))
#         plt.subplots_adjust(bottom=0.2)
        
#         # Plotting
#         # Plotting the antisymmetric part
#         cset_0 = axAS[0].contourf(X, Y, Zanti, levels=fill_levels, cmap=cmap, norm=norm, extend='both')
#         cset1_0 = axAS[0].contour( X, Y, Zanti, levels=line_levels, colors='k', linestyles='--', linewidths=1.5)
#         # Plotting the symmetric part        
#         cset_1 = axAS[1].contourf(X, Y, Zsym,  levels=fill_levels, cmap=cmap, norm=norm, extend='both')
#         cset1_1 = axAS[1].contour( X, Y, Zsym,  levels=line_levels, colors='k', linestyles='--', linewidths=1.5)

#         cpd_lines = [3, 6, 30, 60]  # Common time periods
#         # Settings
#         # Set the basic properties of the two panels
#         for ax in axAS:
#             ax.xaxis.set_tick_params(labelsize=12)
#             ax.yaxis.set_tick_params(labelsize=12)
#             ax.axvline(x=0, color='k', linestyle='--')
#             ax.set_xlabel('Zonal Wavenumber', size=12, fontweight='bold')
#             ax.set_ylabel('Frequency (CPD)', size=12, fontweight='bold')
#             ax.text(15-2*0.25*15, -0.01, 'EASTWARD', fontweight='bold', fontsize=10)
#             ax.text(-15+0.25*15, -0.01, 'WESTWARD', fontweight='bold', fontsize=10)
#             ax.set_xlim(-15, 15)
#             ax.set_ylim(0.02, 0.5)

#             # Add frequency lines and labels
#             for d in cpd_lines:
#                 if (1./d <= 0.5):
#                     ax.axhline(y=1./d, color='k', linestyle='--')
#                     ax.text(-15+0.2, (1./d+0.01), f'{d} days',
#                         size=12, bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'})

#         # Colorbar
#         cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
#         sm = plt.cm.ScalarMappable(norm=norm, cmap='YlOrRd'); sm.set_array([])
#         cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=line_levels[::2], extend='both')
#         cbar.set_label('log10 Power')
#         cbar.ax.tick_params(labelsize=12)
        
#         # Add title
#         axAS[0].set_title("Antisymmetric", fontsize=14)
#         axAS[1].set_title("Symmetric", fontsize=14)
        
#         # plt.tight_layout(rect=[0, 0.1, 1, 0.97])

#         return fig
    
#     @staticmethod
#     def plot_background_spectrum(background, wavenumber, frequency, nlat=params.Ny,
#                                figsize=(10, 8), contour_range=(-2.0, -0.2 + 1e-9, 0.1)):
#         """
#         Plot background spectrum.
        
#         Parameters:
#         -----------
#         background : ndarray
#             Background spectrum array
#         wavenumber : ndarray
#             Wavenumber array
#         frequency : ndarray
#             Frequency array
#         nlat : int, optional
#             Number of latitudes (default: 40)
#         figsize : tuple, optional
#             Figure size (default: (10, 8))
#         contour_range : tuple, optional
#             Contour range (min, max, step) (default: (-1.8, 0, 0.2))
            
#         Returns:
#         --------
#         fig : Figure
#             Matplotlib figure
#         """
#         min_val, max_val, step = contour_range
#         contour_levels = np.arange(min_val, max_val, step)
        
#         # Handle wavenumber range
#         wn_start = np.where(wavenumber >= -15)[0][0] if -15 in wavenumber else 0
#         wn_end = np.where(wavenumber <= 15)[0][-1] + 1 if 15 in wavenumber else len(wavenumber)
#         wavenumber_subset = wavenumber[wn_start:wn_end]
#         background_subset = background[wn_start:wn_end, :]

#         mask_frequency = frequency > 0
#         frequency = frequency[mask_frequency]
#         background_subset = background_subset[:, mask_frequency]

#         # Create figure
#         fig, ax = plt.subplots(figsize=figsize)
        
#         # Prepare grid
#         X, Y = np.meshgrid(wavenumber_subset, frequency)
        
#         # Background data (transposed since we want frequency on y-axis)
#         background_log = np.log10(np.maximum(background_subset.T, 1e-10))
        
#         # Plot background
#         cs = ax.contour(X, Y, background_log, 
#                         levels=contour_levels, colors='k')
#         cf = ax.contourf(X, Y, background_log,
#                         levels=contour_levels, cmap='YlOrRd', extend='both')
        
#         # Add frequency lines
#         for d in [3, 6, 30, 60]:
#             if (1./d <= 0.5):
#                 ax.axhline(y=1./d, color='k', linestyle='--')
#                 ax.text(-15+0.2, (1./d+0.01), f'{d} days',
#                        bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'})
        
#         # Set axis labels
#         ax.set_xlabel('Zonal Wavenumber', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Frequency (CPD)', fontsize=12, fontweight='bold')
#         ax.axvline(x=0, color='k', linestyle='--')
#         ax.set_xlim(-15, 15)
#         ax.set_ylim(0.02, 0.5)
        
#         # Add title
#         ax.set_title("Background Spectrum", fontsize=14)
        
#         # Add colorbar
#         cbar = plt.colorbar(cf)
#         cbar.set_label('Log10 Power', size=12)
        
#         # plt.tight_layout()

#         return fig
    
#     @staticmethod
#     def plot_rm_background_spectrum(power_sym, power_asym, background, wavenumber, frequency, nlat=params.Ny,
#                                  figsize=(18, 8), contour_range=(1.2, 3, 0.1), with_matsuno=True):
#         """
#         Plot CFWT spectrum with background removed.
        
#         Parameters:
#         -----------
#         power_sym : ndarray
#             Symmetric power spectrum array
#         power_asym : ndarray
#             Antisymmetric power spectrum array
#         background : ndarray
#             Background spectrum array
#         wavenumber : ndarray
#             Wavenumber array
#         frequency : ndarray
#             Frequency array
#         figsize : tuple, optional
#             Figure size (default: (18, 8))
#         contour_range : tuple, optional
#             Contour range (min, max, step) (default: (1.3, 3.0, 0.2))
#         with_matsuno : bool, optional
#             Whether to add Matsuno mode lines (default: True)
            
#         Returns:
#         --------
#         fig : Figure
#             Matplotlib figure
#         """        
#         power_sym_r  = (power_sym  / background).astype('float64')
#         power_asym_r = (power_asym / background).astype('float64')
        
#         # zwn from -15 to 15
#         wn_start = np.where(wavenumber == -15)[0][0] if -15 in wavenumber else 0
#         wn_end = np.where(wavenumber == 15)[0][0] + 1 if 15 in wavenumber else len(wavenumber)
#         wavenumber_subset = wavenumber[wn_start:wn_end]
#         power_sym_subset = power_sym_r[wn_start:wn_end]
#         power_asym_subset = power_asym_r[wn_start:wn_end]

#         mask_frequency = frequency > 0
#         frequency_subset = frequency[mask_frequency]
#         power_sym_subset = power_sym_subset[:, mask_frequency]      # (K, Fpos)
#         power_asym_subset = power_asym_subset[:, mask_frequency]

#         # from (wavenumber, frequency) to (frequency, wavenumber)
#         power_sym_rearranged = power_sym_subset.T
#         power_asym_rearranged = power_asym_subset.T

#         # Data preparation
#         psumsym_r  = power_sym_rearranged
#         psumanti_r = power_asym_rearranged
        
#         import matplotlib.colors as mcolors

#         cmap = plt.get_cmap('YlOrRd').copy()
#         cmap.set_bad('white')   # NaN -> white
#         cmap.set_under('white') 
#         line_levels = np.arange(1.2, 3, 0.2)
#         fill_levels = np.linspace(line_levels.min(), line_levels.max(), 80)
#         norm = mcolors.BoundaryNorm(line_levels, cmap.N, clip=False)
#         X, Y = np.meshgrid(wavenumber_subset, frequency_subset)
        
#         # Create figure
#         fig, axAS = plt.subplots(1, 2, figsize=figsize)
#         plt.subplots_adjust(bottom=0.2)

#         # Plotting 
#         # Plotting the antisymmetric part
#         cset_0 = axAS[0].contourf(X, Y, psumanti_r, levels=fill_levels, norm=norm, extend='both', cmap=cmap)
#         cset1_0 = axAS[0].contour(X, Y, psumanti_r, levels = line_levels, colors='k', linestyles='--', linewidths=1.5)

#         # Plotting the symmetrical part
#         cset_1 = axAS[1].contourf(X, Y, psumsym_r, levels=fill_levels, norm=norm, extend='both', cmap=cmap)
#         cset1_1 = axAS[1].contour(X, Y, psumsym_r, levels = line_levels, colors='k', linestyles='--', linewidths=1.5)

#         cpd_lines = [3, 6, 30, 60]
#         he = [12, 25, 50]  # Equivalent depths
#         meridional_modes = [1]  # Meridional modes
#         # Basic properties of the two panels
#         for ax in axAS:
#             ax.xaxis.set_tick_params(labelsize=12)
#             ax.yaxis.set_tick_params(labelsize=12)
#             ax.axvline(x=0, color='k', linestyle='--')
#             ax.set_xlabel('Zonal Wavenumber', size=12, fontweight='bold')
#             ax.set_ylabel('Frequency (CPD)', size=12, fontweight='bold')
#             ax.text(15-2*0.25*15, -0.01, 'EASTWARD', fontweight='bold', fontsize=10)
#             ax.text(-15+0.25*15, -0.01, 'WESTWARD', fontweight='bold', fontsize=10)
#             ax.set_xlim(-15, 15)
#             ax.set_ylim(0.02, 0.5)

#             # Add frequency lines and labels
#             for d in cpd_lines:
#                 if (1./d <= 0.5):
#                     ax.axhline(y=1./d, color='k', linestyle='--')
#                     ax.text(-15+0.2, (1./d+0.01), f'{d} days',
#                         size=12, bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'})
        
#         # Add title
#         axAS[0].set_title("Antisymmetric", fontsize=14)
#         axAS[1].set_title("Symmetric", fontsize=14)
        
#         # Add Matsuno modes if wk_spectra is available
#         if with_matsuno:
#             try:
#                 # Add Matsuno modal
#                 matsuno_modes = mp.matsuno_modes_wk(he=he, n=meridional_modes, max_wn=15)

#                 for key in matsuno_modes:
#                     # Symmetrical mode (right panel)
#                     axAS[1].plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)], color='k', linestyle='--')
#                     axAS[1].plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)], color='k', linestyle='--')
#                     axAS[1].plot(matsuno_modes[key]['EIG(n=1,he={}m)'.format(key)], color='k', linestyle='--')
#                     axAS[1].plot(matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)], color='k', linestyle='--')
                    
#                     # Antisymmetric modes (left panel)
#                     axAS[0].plot(matsuno_modes[key]['MRG(he={}m)'.format(key)], color='k', linestyle='--')
#                     axAS[0].plot(matsuno_modes[key]['EIG(n=0,he={}m)'.format(key)], color='k', linestyle='--')

#                 # Add wave type label
#                 key = list(matsuno_modes.keys())[len(list(matsuno_modes.keys()))//2]
#                 wn = matsuno_modes[key].index.values

#                 # Symmetrical Mode Label (right panel)
#                 # Kelvin Wave label
#                 i = int((len(wn)/2)+0.3*(len(wn)/2))
#                 i, = np.where(wn == wn[i])[0]
#                 axAS[1].text(wn[i]-1, matsuno_modes[key]['Kelvin(he={}m)'.format(key)].iloc[i], 'Kelvin',
#                             bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'}, fontsize=16)

#                 # ER wave label
#                 i = int(0.7*(len(wn)/2))
#                 i = np.where(wn == wn[i])[0]
#                 axAS[1].text(wn[i]-1, matsuno_modes[key]['ER(n=1,he={}m)'.format(key)].iloc[i]+0.01, 'ER',
#                             bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'}, fontsize=13)

#                 # Antisymmetric modal label (left panel)
#                 # EIG(n=0) label
#                 i = int((len(wn)/2)+0.1*(len(wn)/2))
#                 i, = np.where(wn == wn[i])[0]
#                 axAS[0].text(wn[i]-1, matsuno_modes[key]['EIG(n=0,he={}m)'.format(key)].iloc[i], 'EIG(n=0)',
#                             bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'}, fontsize=13)

#                 # MRG wave label
#                 i = int(0.7*(len(wn)/2))
#                 i = np.where(wn == wn[i])[0]
#                 axAS[0].text(wn[i]-1, matsuno_modes[key]['MRG(he={}m)'.format(key)].iloc[i], 'MRG',
#                             bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'}, fontsize=13)

#                 key2 = list(matsuno_modes.keys())[0]
#                 # Add EIG and WIG tags (right panel)
#                 i = int((len(wn)/2)+0.3*(len(wn)/2))
#                 axAS[1].text(wn[i]-1, matsuno_modes[key2]['EIG(n=1,he={}m)'.format(key2)].iloc[i], 'EIG',
#                             bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'}, fontsize=13)

#                 i = int(0.55*(len(wn)/2))
#                 axAS[1].text(wn[i]-1, matsuno_modes[key2]['WIG(n=1,he={}m)'.format(key2)].iloc[i], 'WIG',
#                             bbox={'facecolor':'white', 'alpha':0.9, 'edgecolor':'none'}, fontsize=13)
#             except:
#                 print("Warning: Error adding Matsuno modes to the plot")

#         # Colorbar
#         cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
#         sm = plt.cm.ScalarMappable(norm=norm, cmap='YlOrRd'); sm.set_array([])
#         cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=line_levels[::2], extend='both')
#         cbar.set_label('Power')
#         cbar.ax.tick_params(labelsize=12)
        
#         # plt.tight_layout(rect=[0, 0.1, 1, 0.97])

#         return fig

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(process):
    # Initialize params and CFWT
    params = CFWTParams()
    setup = CFWTSetup(params)
    setup.initialize()
    cfwt = CFWTComputation(params)
    validation = CFWTValidation(params, cfwt)
    ccew_filter = CCEWFilter(params, cfwt, validation)
    year = '2020'

    # wavenumber_ordering = ccew_filter.check_wavenumber_ordering()
    
    outpath = '/g/data/up6/cx5009/Task2_kmodel/P_FIELD/Results/Filters_test/0Filter_Analysis/'

    if process == 'spectrum':
        # Reading data
        ds = xr.open_dataset('/g/data/up6/cx5009/Task2_kmodel/Data/NOAA_olr/olr-daily_v01r02_20200101_20201231.nc').sel(lat=slice(-20, 20))
        # data preparasion
        olr_data = ds['olr'].transpose('time', 'lat', 'lon')
        olr_data = anomalies_for_projectA(olr_data, spd=1) 
        olr_ds = olr_data.transpose('time', 'lat', 'lon')
        
        olr_data_2d = olr_ds['olr'].transpose('lon', 'lat', 'time').mean(dim='lat').values
        
        # Calculating CFWT spectram
        power_sym_global_frq, power_asym_global_frq, \
        power_sym_frq, power_asym_frq = cfwt.compute_cfwt(olr_ds.transpose('lon', 'lat', 'time').values, 
                                                          COI=True, local_spectrum=True)
        
        data_xt, recon = validation.check_reconstruction(olr_data_2d)

        power_sym_global_frq  = np.fft.fftshift(power_sym_global_frq,  axes=0)
        power_asym_global_frq = np.fft.fftshift(power_asym_global_frq, axes=0)
        power_sym_frq         = np.fft.fftshift(power_sym_frq,         axes=0)
        power_asym_frq        = np.fft.fftshift(power_asym_frq,        axes=0)
        
        # Original (after fftshift) wavenumber
        k_fft   = np.fft.fftfreq(cfwt.params.Nx, d=1.0) * cfwt.params.Nx      # 0,1,...,Nx/2,-Nx/2+1,...,-1
        k_shift = np.fft.fftshift(k_fft).astype(int)                          # [-Nx/2,...,-1,0,1,...,Nx/2-1]
        # Flip number: Change "positive = west transmission" to "positive = east transmission"
        k_east  = -k_shift
        # In order to make the coordinates monotonically increasing, the spectrum needs to be rearranged at the same time
        order   = np.argsort(k_east)                                          # From small to large
        k_final = k_east[order]        
        # Do the same rearrangement for all spectra with k as the 0th dimension
        power_sym_global_frq  = power_sym_global_frq[order, :]
        power_asym_global_frq = power_asym_global_frq[order, :]
        power_sym_frq     = power_sym_frq[order, :, :]
        power_asym_frq    = power_asym_frq[order, :, :]        
        # 5) Save coordinates
        wavenumber = k_final.astype(int)
        
        ds_out = xr.Dataset(
            {
                'power_sym_global': (['wavenumber', 'frequency'], power_sym_global_frq),
                'power_asym_global': (['wavenumber', 'frequency'], power_asym_global_frq),
                'power_sym_frq': (['wavenumber', 'frequency', 'time'], power_sym_frq),
                'power_asym_frq': (['wavenumber', 'frequency', 'time'], power_asym_frq),
                'Original data': (['lon', 'time'], data_xt),
                'Reconstructed data': (['lon', 'time'], recon)
            },
            coords={
                'time': ds.time, 
                'frequency': cfwt.params.frq,
                'wavenumber': wavenumber,
                'lon': ds.lon
            },
            attrs={
                'creation_date': str(datetime.datetime.now()),
                'description': 'CFWT Python version from 10.1007/s00382-013-1949-8 \
                written based on Fortran code from http:// iprc.soest.hawaii.edu/users/kazuyosh/'
            }
        )
        
        ds_out.power_sym_global.attrs['long_name'] = 'Symmetric power spectrum'
        ds_out.power_sym_global.attrs['units'] = 'W^2/m^4'
        
        ds_out.to_netcdf(os.path.join(outpath, 'CFWT_test_2020_rearray.nc'))

    elif process == 'CCEWs':
        # 1 Reading data
        ds = xr.open_dataset(f'/g/data/up6/cx5009/Task2_kmodel/Data/NOAA_olr/olr-daily_v01r02_{year}0101_{year}1231.nc').sel(lat=slice(-20, 20))
        eq_lat_idx = abs(ds.lat).argmin().item()
        # print(eq_lat_idx)

        # Check and convert the longitude range to make sure it is from 0 to 360
        if (ds.lon < 0).any():
            # Convert longitude from [-180, 180) to [0, 360)
            ds = ds.assign_coords(lon=(ds.lon % 360))
            ds = ds.sortby('lon')
        
        # 2 Data preparation
        olr_data = ds['olr'].transpose('time', 'lat', 'lon')
        olr_data_processed = anomalies_for_projectA(olr_data, spd=1)
        # print("Before processing:", olr_data.shape)
        # print("After processing:", olr_data_processed.shape)
        
        # Preparing data in the right format
        olr_data_transposed = olr_data_processed.transpose('lon', 'lat', 'time').values
        
        # 3 Store all filter results
        filtered_results = {}
        print("\nBatch extraction of multiple fluctuations...")
        wave_types = ['kelvin', 'mjo']
        mode='3D'
        
        if mode=='2D':
            multi_filtered = ccew_filter.filter_multiple_waves(olr_data_transposed, wave_types, eq_lat_idx)
            for wave_type, filtered_data in multi_filtered.items():
                if wave_type != 'original':
                    filtered_results[wave_type] = filtered_data
            olr_ds.close()
            ccew_filter.save_results_to_netcdf(filtered_results, ds, os.path.join(outpath, f'CFWT_CCEW_filter_{year}.nc'))
            ds.close()
        elif mode=='3D':
            for wave_type in wave_types:
                print(f"\nProcessing {wave_type} ...")
                filtered = ccew_filter.filter_reconstruction_full3d(olr_data_transposed, wave_type)
                print(f"NaN after filter_reconstruction: {np.isnan(filtered).sum()}")
        
                k = np.fft.fftfreq(params.Nx)*params.Nx
                xt_eq = filtered[:, params.Ny//2, :]                # 重构后赤道剖面
                C = np.fft.fft(xt_eq, axis=0)                  # k-τ 频域能量
                E_east = np.sum(np.abs(C[k<0, :])**2)
                E_west = np.sum(np.abs(C[k>0, :])**2)
                print("East/West energy =", E_east/(E_west+1e-12))
                
                ds_lon = ds['lon'].values
                print("filtered.T shape:", filtered.T.shape)
                print("len(time):", len(ds.time))
                print("len(lon):", len(ds_lon))
            
                filtered_darray = xr.DataArray(
                    data=filtered,
                    dims=['lon', 'lat', 'time'],
                    coords={
                        'time': ds.time,
                        'lat': ds.lat,
                        'lon': ds_lon
                    })
                # filtered_darray = filtered_darray.transpose('time', 'lat', 'lon')
                print(filtered_darray.isnull().sum())
                filtered_results[f'NOAA_{wave_type}'] = filtered_darray
            
            filtered_results_ds = xr.Dataset(filtered_results)
            filtered_results_ds = filtered_results_ds.transpose('time', 'lat', 'lon')
            filtered_results_ds.to_netcdf(os.path.join(outpath, f'CFWT_CCEW_filter_{year}_3d_2508.nc'))
            print(f'Saved CFWT_CCEW_filter_{year}_3d_2508.nc')
            
            ds.close()
            filtered_results_ds.close()
    
if __name__ == "__main__":
    main('CCEWs')


