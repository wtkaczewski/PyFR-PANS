# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, t'>
% for i in range(nvars - 1):
    ur[${i}] = ul[${i}];
% endfor
    ur[${nvars - 3}] = ${c['p']}/${c['gamma'] - 1}
                     + 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};

    ur[${nvars - 2}] = ${c['ku']};
    ur[${nvars - 1}] = ${c['eu']};

</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
