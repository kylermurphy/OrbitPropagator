# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 22:08:40 2015

@author: alek
"""

import cProfile
from pstats import SortKey, Stats

import matplotlib.pyplot, matplotlib.cm 
import matplotlib.ticker, matplotlib.font_manager
import csv 
import pandas as pd
import numpy as np
import pytz
import aacgmv2
import pdb

import sklearn

from datetime import datetime, timedelta

from scipy.integrate import solve_ivp
from pymsis import msis

from scipy.special import legendre

from numpy import power, sign, cos, sin, sqrt, zeros, array, cross
from numpy import isfinite
from numpy.linalg import norm

from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy import units as u


pi = np.pi
GM = 3986004.415E8 # Earth's gravity constant from EGM96, m**3/s**2.
atm_rot_vec = array([0.0,0.0,72.9211E-6]) # rads / s 

global mlt_l
mlt_l = False

global atm_d
atm_d = list()

def readEGM96Coefficients():
    """EGM 96 Coefficients.
    
    Read the EGM96 gravitational field model coefficients from EGM96coefficients
    file and parse them to be used with computeGravitationalPotential functions.
    
    Returns
    -------
    2-tuple of the C and S coefficnients of EGM96 model. They are stored in dictionaries
        of list. The keys are degrees of the potential expansion and the values
        of the list entries are the coefficients corresponding to the orders for
        the expansion to a given degree.
    
    Reference
    ---------
    EGM96 coefficients have been downloaded from:
        ftp://cddis.gsfc.nasa.gov/pub/egm96/general_info/readme.egm96
    """
    " Read the coefficients. "
    degrees = []; orders = []; CcoeffsTemp = []; ScoeffsTemp = [];
    with open("EGM96coefficients", "r") as egm96file:
        reader = csv.reader(egm96file, delimiter=" ")
        for row in reader:
            degrees.append( row[1] ) # There will be some "  " in row, the delimiter isn't always " ", sometimes it's "  "...
            orders.append( row[2] )
            CcoeffsTemp.append( row[3] )
            ScoeffsTemp.append( row[4] )
    
    # Change to numbers from str.
    degrees = [int(x) for x in degrees]
    orders = [int(x) for x in orders]
    CcoeffsTemp = [float(x) for x in CcoeffsTemp]
    ScoeffsTemp = [float(x) for x in ScoeffsTemp]
    
    " Parse C and S coefficients to an easily usable format. "
    # Store a list of coefficients corresponding to the given degree of len( no. orders corresponding to this degree ).
    Ccoeffs = {0:[1],1:[0,0]}; Scoeffs ={0:[0],1:[0,0]}; # Initial coefficients for spherical Earth. C_10, C_11, and S_11 are 0 if the origin is at the geocentre.
    for i in range(len(degrees)): # Initialise emoty lists.
        Ccoeffs[degrees[i]] = []
        Scoeffs[degrees[i]] = []
    
    for i in range(len(degrees)): # Store the coefficients.
        Ccoeffs[degrees[i]].append( CcoeffsTemp[i] )
        Scoeffs[degrees[i]].append( ScoeffsTemp[i] )
        
    return Ccoeffs, Scoeffs


"""
===============================================================================
    PROPAGATION FUNCTIONS.
===============================================================================
"""
def calculateGeocentricLatLon(stateVec, epoch):
    """Get the geocentric Lat/Lon.
    
    Calculate the geocentric co-latitude (measured from the noth pole not
    equator), longitude and radius corresponding to the state vector given in
    inertial frame at a certain epoch.
    
    Arguments
    ---------
    stateVec - numpy.ndarray of shape (1,6) with three Cartesian positions and
        three velocities in an inertial reference frame in metres and metres
        per second, respectively.
    epoch - datetime with the UTC epoch corresponding to the stateVect.
    
    Returns
    -------
    3-tuple of floats with geocentric latitude, longitude and radius in radians
        and distance units of stateVec.
        
    References
    ----------
    Conversions taken from:
    http://agamenon.tsc.uah.es/Asignaturas/it/rd/apuntes/RxControl_Manual.pdf
    """
    # Get the state vector and epoch in astropy's formats.
    #epochAstro = Time(epoch, scale='utc', format='datetime')
    stateVecAstro = CartesianRepresentation(x=stateVec[0], y=stateVec[1],
                                             z=stateVec[2], unit=u.m)
    
    # Convert from the inertial reference frame
    # (assume GCRS, which is practically
    # the same as J2000) to Earth-fixed ITRS.
    stateVec_GCRS = GCRS(stateVecAstro, obstime=epoch)
    stateVec_ITRS = stateVec_GCRS.transform_to(ITRS(obstime=epoch))

    # Compute the gravity acceleration in Earth-fixed frame.
    r = norm(stateVec[:3])
    lat = stateVec_ITRS.earth_location.lat.to_value(u.rad)
    colat = pi/2.0 - lat
    lon = stateVec_ITRS.earth_location.lon.to_value(u.rad)
    
    return colat, lat, lon, r

def calculateDragAcceleration(stateVec, geocent_pos, epoch, satMass):
    """Calculate Drag Acceleration.
    
    Calculate the acceleration due to atmospheric drag acting on the
    satellite at a given state (3 positions and 3 velocities) and epoch.
    Use NRLMSISE2000 atmospheric model with globally defined solar activity
    proxies:
        F10_7A - 81-day average F10.7.
        F10_7 - daily F10.7 for the previous day.
        MagneticIndex - daily magnetic index AP.
        NRLMSISEaph - nrlmsise_00_header.ap_array with magnetic values.#
    
    Arguments
    ---------
    numpy.ndarray of shape (1,6) with three Cartesian positions and three
        velocities in an inertial reference frame in metres and metres per
            second, respectively.
    epoch - datetime corresponding to the UTC epoch at which the rate of change
        is to be computed.
    
    Returns
    -------
    numpy.ndarray of shape (1,3) with three Cartesian components of the
        acceleration in m/s2 given in an inertial reference frame.
    """
    geo = -1 if USE_STORM else 1 
    _, lat, lon, r = geocent_pos
    alt_km = (r - EarthRadius)/1000. 
    
    
    try:
        d_prof = msis.run(epoch, lon*180./pi, 
                          lat*180./pi, 
                          [alt_km,400], geomagnetic_activity=geo)
    except Exception as e: 
        pdb.set_trace()

    v_rel = stateVec[3:] - cross(atm_rot_vec,stateVec[:3])
    
    " Use the calculated atmospheric density to compute the drag force. "
    if USE_RFML:
        global mlt_l
        # get the parameters we need for run the RF ML mode
        f_idx = rf_input.index.get_indexer([epoch.replace(tzinfo=None)], 
                                           method='nearest')
        f_dat = rf_input.iloc[f_idx].copy()
        
        alat = lat*180./pi
        alon = lon*180./pi
        
        alat = sign(alat)*20.1 if abs(alat) < 20. else alat
        
        _, _, mlt = aacgmv2.get_aacgm_coord(alat, alon, 400,
                                      epoch.replace(tzinfo=None),
                                      method='GEOCENTRIC')
        
        if isfinite(mlt):
            mlt_l = mlt
        else:
            if mlt_l:
                mlt = mlt_l
            else:
                dl = 0.5 if alat>0 else -0.5
                mlt_b = np.nan
                while not isfinite(mlt_b):
                    alat = alat+dl
                    _, _, mlt_b = aacgmv2.get_aacgm_coord(alat, alon, 400,
                                                  epoch.replace(tzinfo=None),
                                                  method='GEOCENTRIC')

                mlt_l = mlt_b
                mlt = mlt_b
                
                
                
        
        
        f_dat['SatLat'] = lat*180./pi
        f_dat['cos_SatMagLT'] = cos(mlt*2*pi/24.)
        f_dat['sin_SatMagLT'] = sin(mlt*2*pi/24.)
        
        try:
            d400 = rf_ml.predict(f_dat)*10**-12
        except:
            d400 = d_prof[0,0,0,0,0]
        
        #scale the 400 km density to the height of the satellite
        atmosphericDensity = d_prof[0,0,0,0,0]*d400/d_prof[0,0,0,1,0]
        
        
    else:
        atmosphericDensity = d_prof[0,0,0,0,0] # kg/m3
        
    global atm_d
    atm_d.append(atmosphericDensity)
        
    dragForce = -0.5*atmosphericDensity*dragArea*Cd*\
        power(v_rel,2)*sign(v_rel)# Drag foce in Newtons.
        
    return dragForce/satMass

def calculateGravityAcceleration(stateVec, geocent_pos):
    """Acceleration due to gravity. 
    
    Calculate the acceleration due to gravtiy acting on the satellite at
    a given state (3 positions and 3 velocities). Ignore satellite's mass,
    i.e. use a restricted two-body problem.
    
    Arguments
    ---------
    numpy.ndarray of shape (1,6) with three Cartesian positions and three
        velocities in an inertial reference frame in metres and metres per
            second, respectively.
    epoch - datetime corresponding to the UTC epoch at which the rate of change
        is to be computed.
    useGeoid - bool, whether to compute the gravity by using EGM geopotential
        expansion (True) or a restricted two-body problem (False).
    
    Returns
    -------
    numpy.ndarray of shape (1,3) with three Cartesian components of the
        acceleration in m/s2 given in an inertial reference frame.
    """

    " Compute geocentric co-latitude, longitude & radius. "
    colatitude, latitude, longitude, r = geocent_pos

    cos_colat = 1.0 if (abs(colatitude-pi/2. <= 1E-16)) or\
        (abs(colatitude-3*pi/2. <= 1E-16))\
            else cos(colatitude)

    " Find the gravitational potential at the desired point. "
    # See Eq. 1 in Cunningham (1996) for the general form of the geopotential expansion.
    gravitationalPotential = 0.0 # Potential of the gravitational field at the stateVec location.
    for degree in range(0, MAX_DEGREE+1): # Go through all the desired orders and compute the geoid corrections to the sphere.
        temp = 0. # Contribution to the potential from the current degree and all corresponding orders.
        legendreCoeffs = legendre(degree) # Legendre polynomial coefficients corresponding to the current degree.
        for order in range(degree+1): # Go through all the orders corresponding to the currently evaluated degree.
            temp += legendreCoeffs[order] * cos_colat * (Ccoeffs[degree][order]*cos( order*longitude ) + Scoeffs[degree][order]*sin( order*longitude ))

        gravitationalPotential += power(EarthRadius/r, degree) * temp # Add the contribution from the current degree.

    gravitationalPotential *= GM/r # Final correction (*GM for acceleration, /r to get r^(n+1) in the denominator).

    " Compute the acceleration due to the gravity potential at the given point."
    # stateVec is defined w.r.t. Earth's centre of mass, so no need to account
    # for the geoid shape here.
    gravityAcceleration = gravitationalPotential/r* (-1.*stateVec[:3]/r) # First divide by the radius to get the acceleration value, then get the direction (towards centre of the Earth).

    return gravityAcceleration
    
def computeRateOfChangeOfState(epoch, stateVector):
    """Compute the rate of change of the state vector.
    
    Arguments
    ---------
    stateVector - numpy.ndarray of shape (1,6) with three Cartesian positions
        and three velocities given in an inertial frame of reference.
    epoch - detetime corresponding to the UTC epoch at which the rate of change
        is to be computed.
        
    Returns
    -------
    numpy.ndarray of shape (1,6) with the rates of change of position and velocity
        in the same inertial frame as the one in which stateVector was given.
    """    
    t_epoch = epoch_0+timedelta(seconds=epoch)
    
    geocen_pos = calculateGeocentricLatLon(stateVector,t_epoch)
    
    if USE_GEOID:
        # A vector of the gravity force from EGM96 model.
        gravityAcceleration = calculateGravityAcceleration(stateVector, 
                                                           geocen_pos) 
    else:
        # Earth-centred radius
        r = norm(stateVector[:3])
        # First compute the magnitude, then get the direction 
        #(towards centre of the Earth)
        gravityAcceleration = GM/(r*r) * (-1.*stateVector[:3]/r)

    if USE_DRAG:
        #  A vector of the drag computed with NRLMSISE
        dragAcceleration = calculateDragAcceleration(stateVector, geocen_pos, 
                                                     t_epoch, satelliteMass) 
    else:
        dragAcceleration = zeros(3)
    
    stateDerivatives = zeros(6)
    # Velocity is the rate of change of position.
    stateDerivatives[:3] = stateVector[3:]; 
    # Compute the acceleration i.e. the rate of change of velocity.
    stateDerivatives[3:] = dragAcceleration+gravityAcceleration 
    return stateDerivatives

def calculateCircularPeriod(stateVec):
    """Calculate period.
    
    Calculate the orbital period of a circular, Keplerian orbit passing through
    the state vector (3 positions and velocities).
    
    Arguments
    ----------
    numpy.ndarray of shape (1,3) with three Cartesian positions and velocities,
        in mtres and m/s, respectively.
    
    Returns
    -------
    Orbital period of a circular orbit corresponding to the supplied state vector
        in seconds.
    """
    return 2*pi*sqrt(power(norm(stateVec[:3]),3)/GM)

# FORCE MODEL SETTINGS.
EarthRadius = 6378136.3 # Earth's equatorial radius from EGM96, m.
MAX_DEGREE = 2 # Maximum degree of the geopotential harmocic expansion to use. 0 equates to two-body problem.
USE_GEOID = True # Whether to account for Earth's geoid (True) or assume two-body problem (False).
USE_DRAG = True # Whether to account for drag acceleration (True), or ignore it (False).
USE_STORM = True # Whether to account for geomagnet storms in MSIS
USE_RFML = False
satelliteMass = 260. # kg
Cd = 5 # Drag coefficient, dimensionless.
dragArea = 24 # Area exposed to atmospheric drag, m2.
Ccoeffs, Scoeffs = readEGM96Coefficients() # Get the gravitational potential exampnsion coefficients.

# get the random forest model 
rf_ml = pd.read_pickle("D:\data\SatDensities\FI_GEO_RFdat_AIMFAHR.pkl")[0]

# sklearn train version must be the same as the installed
if sklearn.__version__ != rf_ml.__getstate__()['_sklearn_version']:
    USE_RFML = False
else:
    rf_input = pd.read_hdf(
        "D:\data\SatDensities\FI_GEO_RFdat_AIMFAHR_inputs_MayStorm.hdf")

state_0 = np.array([0,0.,0.,0.,0.,0.]) # Initial state vector with Cartesian positions and velocities in m and m/s.
state_0[0] = EarthRadius+300.0e3
state_0[5] = sqrt( GM/norm(state_0[:3]) ) # Simple initial condition for test purposes: a circular orbit with velocity pointing along the +Z direction.

state_0[:3] = [-6556040.07395412,  2081708.80924815,   891365.75101391]
state_0[3:] = [-2324.6552955,  -5130.47304865, -5079.3844195 ]

epoch_0 = datetime(2024, 5, 10, 12, 00, 00, 0, tzinfo=pytz.UTC)
OrbitalPeriod_0 = calculateCircularPeriod(state_0) # Orbital period of the initial circular orbit.

# PROPAGATE THE ORBIT NUMERICALLY.
cp = cProfile.Profile()
cp.enable()


INT_STEP_S = 10.0 # Time step at which the trajectory will be propagated.
MAX_STEP_S = 120.0
NO_ORBITS = 20 # For how manyu orbits to propagate.

epochs = pd.date_range(start=epoch_0,
                                 end=epoch_0+
                                 timedelta(seconds=NO_ORBITS*OrbitalPeriod_0),
                                 freq=pd.DateOffset(seconds=INT_STEP_S)
                                 ).to_pydatetime().tolist()

t = [(dt-epoch_0).total_seconds() for dt in epochs]

## Propoage the function
state = solve_ivp(computeRateOfChangeOfState, [t[0],t[-1]], 
                       state_0, t_eval=t, first_step=INT_STEP_S, 
                       max_step=MAX_STEP_S, method='DOP853', 
                       atol = 1E-9, rtol = 1e-6)

# state = solve_ivp(computeRateOfChangeOfState, [t[0],t[-1]], 
#                        state_0, first_step=INT_STEP_S, 
#                        max_step=MAX_STEP_S, method='DOP853', 
#                        atol = 1E-9, rtol = 1e-6)

if len(epochs) != state.t.shape[0]:
    epochs = [epoch_0 + pd.Timedelta(t) for t in state.t]

s_df = pd.DataFrame({'x':state.y.transpose()[:,0], 
                     'y':state.y.transpose()[:,1],
                     'z':state.y.transpose()[:,3],
                     'alt':norm(state.y.transpose()[:,:3],axis=1)-EarthRadius},
                    index=epochs)

if USE_RFML:
    r_df = s_df.copy()
    r_d = np.array(atm_d)
else:
    m_df = s_df.copy()
    m_d = np.array(atm_d)

cp.disable()
Stats(cp).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)


# PLOT FORMATTING.
ticksFontSize = 8
labelsFontSize = 15
titleFontSize = 15

matplotlib.rc('xtick', labelsize=ticksFontSize) 
matplotlib.rc('ytick', labelsize=ticksFontSize)

# FIGURE THAT SHOWS THE EARTH AND SATELLITE TRAJECTORY.
ax = matplotlib.pyplot.figure().add_subplot(projection='3d')
ax.set_aspect('auto') #TODO change 3D axes aspect ratio to equal, which isn't supported now. Current workaround is set scale_xyz below.
ax.view_init(elev=45., azim=45.)
figRange = 1.5*EarthRadius
ax.set_xlim([-figRange, figRange])
ax.set_ylim([-figRange, figRange])
ax.set_zlim([-figRange, figRange])
ax.auto_scale_xyz([-figRange, figRange], [-figRange, figRange], [-figRange, figRange])

" Plot a sphere that represents the Earth and the coordinate frame. "
N_POINTS = 20 # Number of lattitudes and longitudes used to plot the geoid.
latitudes = np.linspace(0, pi, N_POINTS) # Geocentric latitudes and longitudes where the geoid will be visualised.
longitudes = np.linspace(0, 2*pi, N_POINTS)
Xs = EarthRadius * np.outer(np.cos(latitudes), np.sin(longitudes))
Ys = EarthRadius * np.outer(np.sin(latitudes), np.sin(longitudes))
Zs = EarthRadius * np.outer(np.ones(latitudes.size), np.cos(longitudes))
earthSurface = ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, linewidth=0,
                               antialiased=False, shade=False, alpha=0.5)


" Plot the trajectory. "
ax.plot(state.y[0,:],state.y[1,:],state.y[2,:], c='b', lw=1, ls='dotted')


# FIGURE SHOWING EVOLUTION OF THE POSITION COMPONENTS OVER TIME.
fig3, axarr = matplotlib.pyplot.subplots(4, sharex=True, figsize=(12,8))
axarr[0].grid(linewidth=2)
axarr[1].grid(linewidth=2)
axarr[2].grid(linewidth=2)
axarr[3].grid(linewidth=2)
axarr[0].tick_params(axis='both',reset=False,which='both',length=5,width=1.5)
axarr[1].tick_params(axis='both',reset=False,which='both',length=5,width=1.5)
axarr[2].tick_params(axis='both',reset=False,which='both',length=5,width=1.5)
axarr[3].tick_params(axis='both',reset=False,which='both',length=5,width=1.5)

axarr[3].set_xlabel(r'$Time\ elapsed\ (s)$',fontsize=labelsFontSize)
axarr[1].set_ylabel(r'$Altitude\ (m)$',fontsize=labelsFontSize)
axarr[1].set_ylabel(r'$X\ (m)$',fontsize=labelsFontSize)
axarr[2].set_ylabel(r'$Y\ (m)$',fontsize=labelsFontSize)
axarr[3].set_ylabel(r'$Z\ (m)$',fontsize=labelsFontSize)

axarr[0].plot(epochs, s_df.alt, c='r')
axarr[1].plot(epochs, s_df.x, c='r')
axarr[2].plot(epochs, s_df.y, c='r')
axarr[3].plot(epochs, s_df.z, c='r')


fig3.show()