# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcorupans' ndim='2'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              u='in fpdtype_t[${str(nvars)}]'
              ku_src='inout fpdtype_t'
              eu_src='inout fpdtype_t'>


fpdtype_t tmpgradu[${ndims}];

// Get density gradients first
% for i in range(ndims):
    tmpgradu[${i}] = gradu[${i}][${0}];
% endfor	
% for i in range(ndims):
    gradu[${i}][${0}] = rcpdjac*(${' + '.join('smats[{k}][{i}]*tmpgradu[{k}]'
                                              .format(i=i, k=k)
                                              for k in range(ndims))});
% endfor


// Get velocity gradients and TKE production term

fpdtype_t prod = 0.0;
fpdtype_t rcprho = 1/u[0];
fpdtype_t duk_dxj, duj_dxk;
fpdtype_t ku = max(u[${nvars-2}], ${c['min_ku']});
fpdtype_t eu = max(u[${nvars-1}], ${c['min_eu']});
fpdtype_t mu_t = ${c['Cmu']}*ku*ku/eu;
fpdtype_t Ce2s = ${c['Ce1']} + (${c['Ce2']} - ${c['Ce1']})*(${c['fk']/c['fe']} );

% for j in range(0,ndims):
	% for i in range(ndims):
	    tmpgradu[${i}] = gradu[${i}][${j+1}];
	% endfor

	% for i in range(ndims):
	    gradu[${i}][${j+1}] = rcpdjac*(${' + '.join('smats[{k}][{i}]*tmpgradu[{k}]'
	                                              .format(i=i, k=k)
	                                              for k in range(ndims))});
	% endfor

	// Sum production terms only after grad_ij and grad_ji have been evaluated 
	% for k in range(0,j+1):
		duk_dxj = rcprho*(gradu[${j}][${k+1}] - gradu[${j}][0]*u[${k+1}]); // duk_dxj = 1/rho*(drhouk_dxj - drho_dxj*uk)
		duj_dxk = rcprho*(gradu[${k}][${j+1}] - gradu[${k}][0]*u[${j+1}]); // duj_dxk = 1/rho*(drhouj_dxk - drho_dxk*uj)

	    // TKE production term
	    prod += duk_dxj*(-mu_t*(duk_dxj + duj_dxk));
	    % if (k == j):
			prod += duk_dxj*(${2.0/3.0}*ku);
		% else:
	    	prod += duj_dxk*(-mu_t*(duj_dxk + duk_dxj));
	    % endif
	% endfor

% endfor

// Calculate ku and eu source terms
ku_src = prod - eu;
eu_src = (${c['fk']} * (${c['Ce1']}*prod*eu/ku - Ce2s*(eu*eu)/ku));

// Get gradients for energy and turbulence model variables
% for j in range(ndims+1, nvars):
	% for i in range(ndims):
	    tmpgradu[${i}] = gradu[${i}][${j}];
	% endfor
	% for i in range(ndims):
	    gradu[${i}][${j}] = rcpdjac*(${' + '.join('smats[{k}][{i}]*tmpgradu[{k}]'
	                                              .format(i=i, k=k)
	                                              for k in range(ndims))});
		
	% endfor
% endfor


</%pyfr:kernel>
