# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='inviscid_flux' params='s, f, p, v'>
    fpdtype_t invrho = 1.0/s[0], E = s[${nvars - 3}];

    // Compute the velocities
    fpdtype_t rhov[${ndims}];
% for i in range(ndims):
    rhov[${i}] = s[${i + 1}];
    v[${i}] = invrho*rhov[${i}];
% endfor

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*invrho*${pyfr.dot('rhov[{i}]', i=ndims)});

    // Density and energy fluxes
% for i in range(ndims):
    f[${i}][0] = rhov[${i}];
    f[${i}][${nvars - 3}] = (E + p)*v[${i}];
% endfor

    // Momentum fluxes
% for i, j in pyfr.ndrange(ndims, ndims):
    f[${i}][${j + 1}] = rhov[${i}]*v[${j}]${' + p' if i == j else ''};
% endfor

    // Turbulence variable fluxes
    fpdtype_t ku = max(s[${nvars-2}], ${c['min_ku']});
    fpdtype_t eu = max(s[${nvars-1}], ${c['min_eu']});
% for i in range(ndims):
     f[${i}][${nvars-2}] = ${c['tmswitch']}*ku*v[${i}];
     f[${i}][${nvars-1}] = ${c['tmswitch']}*eu*v[${i}];
% endfor

</%pyfr:macro>
