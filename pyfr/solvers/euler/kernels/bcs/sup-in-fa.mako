# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, t'>
    ur[0] = ${c['rho']};
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = (${c['rho']})*(${c[v]});
% endfor
    ur[${nvars - 3}] = ${c['p']}/${c['gamma'] - 1} +
                       0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
    // turbulence variables
    ur[${nvars - 2}] = ${c['ku']};
    ur[${nvars - 1}] = ${c['eu']};
</%pyfr:macro>
