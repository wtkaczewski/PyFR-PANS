# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='negdivconfpans' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'
              gradu='in fpdtype_t[${str(ndims)}][${str(nvars)}]'>

fpdtype_t prod = 0.0;
fpdtype_t ku = max(u[${nvars-2}], ${c['min_ku']});
fpdtype_t eu = max(u[${nvars-1}], ${c['min_eu']});

fpdtype_t mu_t = ${c['Cmu']}*ku*ku/eu;
fpdtype_t Ce2s = ${c['Ce1']} + (${c['Ce2']} - ${c['Ce1']})*(${c['fk']/c['fe']} );

% for i in range(ndims):
	% for j in range(ndims):
		prod += gradu[${j}][${i}]*(-mu_t*(gradu[${j}][${i}] + gradu[${i}][${j}]));
		% if (i == j):
			prod += gradu[${j}][${i}]*(${2.0/3.0}*ku);
		% endif
	% endfor
% endfor


% for i, ex in enumerate(srcex):
	// Turbulence sources seperately
	% if i == (nvars-2):
		tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex} + (prod - eu);
	% elif i == (nvars-1):
		tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex} + (${c['fk']} * (${c['Ce1']}*prod*eu/ku - Ce2s*(eu*eu)/ku));
	% else:
    	tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
	% endif
% endfor

</%pyfr:kernel>