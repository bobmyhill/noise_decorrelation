# Constants
constants = {'G': 6.67408e-11, # m^3/kg/s^2
             'c': 330., # m/s
             'g_0': 9.81,
             'h2s':60.*60.,
             'd2s':23.9345*60.*60.,
             'y2s':365.24*23.9345*60.*60.} # m/s


# Only major tides (displacements > 10mm)
earth_tides = {'K2': 11.96723606*constants['h2s'], # lunisolar semidiurnal
               'M2': 12.4206012*constants['h2s'], # principal lunar semidiurnal
               'N2': 12.65834751*constants['h2s'], # larger lunar elliptic semidiurnal
               'S2': 12.*constants['h2s'], # principal solar semidiurnal
               'K1': 23.93447213*constants['h2s'], # lunar diurnal
               'O1': 25.81933871*constants['h2s'], # lunar diurnal
               'P1': 24.06588766*constants['h2s'], # solar diurnal
               'Mf': 13.661*constants['d2s'],
               'Lunar month': 27.555*constants['d2s'],
               'Solar semi-annual': 0.5*constants['y2s'],
               'Lunar node': 18.613*constants['y2s']}

'''
# More Earth tides with amplitudes (vertical or horizontal) > 10 mm
earth_tides = {'K2': 11.96723606*h2s, # lunisolar semidiurnal
               'M2': 12.4206012*h2s, # principal lunar semidiurnal
               'N2': 12.65834751*h2s, # larger lunar elliptic semidiurnal
               'S2': 12.*h2s, # principal solar semidiurnal
               'K1': 23.93447213*h2s, # lunar diurnal
               'O1': 25.81933871*h2s, # lunar diurnal
               'P1': 24.06588766*h2s, # solar diurnal
               'phi1': 23.804*h2s, # solar diurnal
               'psi1': 23.869*h2s, # solar diurnal
               'S1': 24.*h2s, # solar diurnal
               'Mf': 13.661*d2s,
               'Lunar month': 27.555*d2s,
               'Solar semi-annual': 0.5*y2s,
               'Lunar node': 18.613*y2s,
               'Solar annual': y2s}
'''
