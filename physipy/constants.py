# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define common physical constants.

The constants values are retrieved from scipy.constants, and are separeted in 2 dictionnaries:
 - "scipy_constants" for the most commons constants
 - "scipy_constants_codata" for all the others
A third "constants" dictionnary merges these two.

TODO :
 - create a function to wrap dict creation ?
 - should constants and units be in the same module ?

"""

import scipy.constants as csts

# setup
from .quantity import m, s, kg, A, K, rad, units, mol
from .quantity import make_quantity

Pa = units["Pa"]
J = units['J']
W = units["W"]
N = units["N"]

# Raw mapping for scipy constants
scipy_constants_raw = {
    "c"                      : (csts.c                     , m/s),
    "speed_of_light"         : (csts.speed_of_light        , m/s),   	
    "mu_0"                   : (csts.mu_0                  , m * kg * s**-2 * A**-2), 
    "epsilon_0" 	         : (csts.epsilon_0 	           , A**2 * s**4 * kg**-1 * m**-3), 
    "h" 	                 : (csts.h 	                   , kg * m**2 * s**-1),    
    "Planck"                 : (csts.Planck                , kg * m**2 * s**-1), 
    "hbar" 	                 : (csts.hbar 	               , kg * m**2 * s**-1),    
    "G" 	                 : (csts.G	                   , m**3 * kg**-1 * s**-2),     
    "gravitational_constant" : (csts.gravitational_constant, m**3 * kg**-1 * s**-2),	
    "g"                      : (csts.g                     , m * s**-2),        
    "e" 	                 : (csts.e                     , A * s),    
    "elementary_charge"      : (csts.elementary_charge     , A * s),  	
    "R" 	                 : (csts.R	                   , m**2 * kg * s**-2 * K**-1 * mol**-1),  
    "gas_constant"           : (csts.gas_constant          , m**2 * kg * s**-2 * K**-1 * mol**-1),
    "alpha" 	             : (csts.alpha	               , 1),        
    "fine_structure"         : (csts.fine_structure        , 1),     	
    "N_A" 	                 : (csts.N_A                   , mol**-1), 
    "Avogadro"               : (csts.Avogadro              , mol**-1),   	
    "k" 	                 : (csts.k                     , J * K**-1),  
    "Boltzmann"              : (csts.Boltzmann             , J * K**-1),    	
    "sigma" 	             : (csts.sigma 	               , W * m**-2 * K**-4),  
    "Stefan_Boltzmann"       : (csts.Stefan_Boltzmann      , W * m**-2 * K**-4),              	
    "Wien" 	                 : (csts.Wien	               , m * K),    
    "Rydberg"                : (csts.Rydberg               , m**-1),   
    "m_e" 	                 : (csts.m_e                   , kg),
    "electron_mass" 	     : (csts.electron_mass	       , kg),        
    "m_p" 	                 : (csts.m_p                   , kg),         
    "proton_mass"            : (csts.proton_mass           , kg),  
    "m_n" 	                 : (csts.m_n                   , kg),   
    "neutron_mass"           : (csts.neutron_mass          , kg),          
}

# Raw mapping for scipy constants codata
scipy_constants_codata_raw = {
    # SI prefixes
    'yotta': (csts.yotta, 1), 	
    'zetta': (csts.zetta, 1), 	
    'exa'  : (csts.exa,   1),
    'peta' : (csts.peta,  1),	
    'tera' : (csts.tera,  1),	
    'giga' : (csts.giga,  1),	
    'mega' : (csts.mega,  1),	
    'kilo' : (csts.kilo,  1),	
    'hecto': (csts.hecto, 1), 	
    'deka ': (csts.deka,  1),	
    'deci ': (csts.deci,  1),	
    'centi': (csts.centi, 1), 	
    'milli': (csts.milli, 1), 	
    'micro': (csts.micro, 1), 	
    'nano ': (csts.nano , 1),	
    'pico ': (csts.pico , 1),	
    'femto': (csts.femto, 1), 	
    'atto ': (csts.atto , 1),	
    'zepto': (csts.zepto, 1), 	
     
    # Binary prefix
    'kibi': (csts.kibi, 1),	
    'mebi': (csts.mebi, 1),	
    'gibi': (csts.gibi, 1),	
    'tebi': (csts.tebi, 1),	
    'pebi': (csts.pebi, 1),	
    'exbi': (csts.exbi, 1),	
    'zebi': (csts.zebi, 1),	
    'yobi': (csts.yobi, 1),	
    
    # Mass
    'gram' 	     : (csts.gram       , kg),  
    'metric_ton' : (csts.metric_ton , kg),  	
    'grain' 	 : (csts.grain 	    , kg),   
    'lb' 	     : (csts.lb 	    , kg), 
    'pound' 	 : (csts.pound	    , kg),    
    'blob' 	     : (csts.blob 	    , kg), 
    'slinch'     : (csts.slinch     , kg), 
    'slug' 	     : (csts.slug	    , kg), 
    'oz'         : (csts.oz         , kg), 
    'ounce' 	 : (csts.ounce 	    , kg),   
    'stone' 	 : (csts.stone 	    , kg),   
    'grain' 	 : (csts.grain 	    , kg),   
    'long_ton' 	 : (csts.long_ton 	, kg),
    'short_ton'  : (csts.short_ton  , kg),
    'troy_ounce' : (csts.troy_ounce , kg), 
    'troy_pound' : (csts.troy_pound , kg),    
    'carat'      : (csts.carat      , kg),
    'm_u' 	     : (csts.m_u 	    , kg), 
    'u'	         : (csts.u	        , kg), 
    'atomic_mass': (csts.atomic_mass, kg), 
    
    # Angle
    'deg'      : (csts.degree,    rad),
    'arcmin'   : (csts.arcmin,    rad),
    'arcminute': (csts.arcminute, rad),
    'arcsec'   : (csts.arcsec,    rad),
    'arcsecond': (csts.arcsecond, rad),
    
    # Time
    'minute'     : (csts.minute,      s),
    'hour'       : (csts.hour,        s),
    'day'        : (csts.day,         s),
    'week'       : (csts.week,        s),
    'year'       : (csts.year,        s),
    'Julian_year': (csts.Julian_year, s),
    
    # Length
    'inch'             : (csts.inch,              m),
    'foot'             : (csts.foot,              m),
    'yard'             : (csts.yard,              m),
    'mile'             : (csts.mile,              m),
    'mil'              : (csts.mil,               m),
    'pt'               : (csts.pt,                m),
    'point'            : (csts.point,             m),
    'survey_foot'      : (csts.survey_foot,       m),
    'survey_mile'      : (csts.survey_mile,       m),
    'nautical_mile'    : (csts.nautical_mile,     m),
    'fermi'            : (csts.fermi,             m),
    'angstrom'         : (csts.angstrom,          m),
    'micron'           : (csts.micron,            m),
    'au'               : (csts.au,                m),
    'astronomical_unit': (csts.astronomical_unit, m),
    'light_year'       : (csts.light_year,        m),
    'parsec'           : (csts.parsec,            m),
    
    # Pressure
    'atm'       : (csts.atmosphere, Pa),
    'atmosphere': (csts.atmosphere, Pa),
    'bar'       : (csts.atmosphere, Pa),
    'torr'      : (csts.atmosphere, Pa),
    'mmHg'      : (csts.atmosphere, Pa),
    'psi'       : (csts.atmosphere, Pa),
        
    # Area
    'hectare' : (csts.hectare, m**2),
    'acre'    : (csts.acre,    m**2),
    
    # Volume
    'liter'           : (csts.liter,           m**3),
    'litre'           : (csts.litre,           m**3),
    'gallon'          : (csts.gallon,          m**3),
    'gallon_US'       : (csts.gallon_US,       m**3),
    'gallon_imp'      : (csts.gallon_imp,      m**3),
    'fluid_ounce'     : (csts.fluid_ounce,     m**3),
    'fluid_ounce_US'  : (csts.fluid_ounce_US,  m**3),
    'fluid_ounce_imp' : (csts.fluid_ounce_imp, m**3),
    'bbl'             : (csts.bbl,             m**3),
    'barrel'          : (csts.barrel,          m**3),
        
    #Speed
    'kmh'            : (csts.kmh,            m/s),
    'mph'            : (csts.mph,            m/s),
    'mach'           : (csts.mach,           m/s),
    'speed_of_sound' : (csts.speed_of_sound, m/s),
    'knot'           : (csts.knot,           m/s),
    
    # Temperature
    'zero_Celsius'      : (csts.zero_Celsius,      K),
    'degree_Fahrenheit' : (csts.degree_Fahrenheit, K),
        
    # Energy
    'eV'            : (csts.eV,            J),
    'electron_volt' : (csts.electron_volt, J),
    'calorie'       : (csts.calorie,       J),
    'calorie_th'    : (csts.calorie_th,    J),
    'calorie_IT'    : (csts.calorie_IT,    J),
    'erg'           : (csts.erg,           J),
    'Btu'           : (csts.Btu,           J),
    'Btu_IT'        : (csts.Btu_IT,        J),
    'Btu_th'        : (csts.Btu_th,        J),
    'ton_TNT'       : (csts.ton_TNT,       J),
    
    # Power
    'hp'         : (csts.hp,         W),
    'horsepower' : (csts.horsepower, W),
    
    # Force
    'dyn'            : (csts.dyn,            N),
    'dyne'           : (csts.dyne,           N),
    'lbf'            : (csts.lbf,            N),
    'pound_force'    : (csts.pound_force,    N),
    'kgf'            : (csts.kgf,            N),
    'kilogram_force' : (csts.kilogram_force, N),

}


# scipy constants codata
scipy_constants_codata = {
    key: make_quantity(
        value[0] * value[1],
        symbol=key) for key,
    value in scipy_constants_codata_raw.items()}

# scipy constants
scipy_constants = {
    key: make_quantity(
        value[0] * value[1],
        symbol=key) for key,
    value in scipy_constants_raw.items()}

# constants : concatenation of the two dicts
constants = {**scipy_constants, **scipy_constants_codata}

# clean up
del csts
del units
del make_quantity
del scipy_constants_codata_raw
del scipy_constants_raw
del m, s, kg, A, K, rad, mol
