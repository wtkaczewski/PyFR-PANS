# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcorupans' ndim='2'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              u='in fpdtype_t[${str(nvars)}]'
              prod='inout fpdtype_t'>

fpdtype_t tmpgradu[${ndims}];


% for j in range(nvars):
	% for i in range(ndims):
	    tmpgradu[${i}] = gradu[${i}][${j}];
	% endfor
	% for i in range(ndims):
	    gradu[${i}][${j}] = rcpdjac*(${' + '.join('smats[{k}][{i}]*tmpgradu[{k}]'
	                                              .format(i=i, k=k)
	                                              for k in range(ndims))});
	% endfor
% endfor


prod = 0.0;
fpdtype_t ku = max(u[${nvars-2}], ${c['min_ku']});
fpdtype_t eu = max(u[${nvars-1}], ${c['min_eu']});
fpdtype_t mu_t = ${c['Cmu']}*ku*ku/eu;

% for i, j in pyfr.ndrange(ndims, ndims):
	prod += gradu[${j}][${i}]*(-mu_t*(gradu[${j}][${i}] + gradu[${i}][${j}]));
	% if (i == j):
		prod += gradu[${j}][${i}]*(${2.0/3.0}*ku);
	% endif
% endfor

</%pyfr:kernel>
