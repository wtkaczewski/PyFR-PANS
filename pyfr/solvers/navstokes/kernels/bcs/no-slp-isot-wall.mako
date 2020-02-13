# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, t'>
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = -ul[${i + 1}] + 2*${c[v]}*ul[0];
% endfor
    ur[${nvars - 3}] = ${c['cpTw']/c['gamma']}*ur[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};    
    ur[${nvars - 2}] = 0.0;
    ur[${nvars - 1}] = ul[${nvars - 1}];
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur, ploc, t'>
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = ${c[v]}*ul[0];
% endfor
    ur[${nvars - 3}] = ${c['cpTw']/c['gamma']}*ur[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
    ur[${nvars - 2}] = 0.0;
    ur[${nvars - 1}] = ${c['wu_wall']};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
