# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 04:41:04 2024

@author: krmurph1
"""

# old code for integrating
" Actual numerical propagation main loop. "
propagatedStates = zeros( (len(epochsOfInterest),6) ) # State vectors at the  epochs of interest.
# Propagate the state to all the desired epochs statring from state_0.
for i in range(1, len(epochsOfInterest)): 
    propagatedStates[i,:] = RungeKutta4(propagatedStates[i-1], t[i-1],
                                        INTEGRATION_TIME_STEP_S, is_dt, 
                                        computeRateOfChangeOfState)

" Compute quantities derived from the propagated state vectors. "
altitudes = pd.DataFrame({'alt':norm(propagatedStates[:,:3], axis=1) - EarthRadius}) # Altitudes above spherical Earth...
specificEnergies = [ norm(x[3:])*norm(x[3:]) -
                    GM*satelliteMass/norm(x[:3]) for x in propagatedStates] # ...and corresponding specific orbital energies.

altitudes2Body = norm(propagatedStates2Body[:,:3], axis=1) - EarthRadius    

sci_states = solve_ivp(computeRateOfChangeOfState, [t[0],t[-1]], 
                        state_0, t_eval=t, first_step=10, max_step=60)
sci_alt = pd.DataFrame({'alt':norm(sci_states.y.transpose()[:,:3],axis=1)-EarthRadius})