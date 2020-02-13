# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcorupans' ndim='2'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              u='in fpdtype_t[${str(nvars)}]'
              ku_src='inout fpdtype_t'
              wu_src='inout fpdtype_t'
              t = 'scalar fpdtype_t'>


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


% for j in range(ndims):
	% for i in range(ndims):
	    tmpgradu[${i}] = gradu[${i}][${j+1}];
	% endfor

	% for i in range(ndims):
	    gradu[${i}][${j+1}] = rcpdjac*(${' + '.join('smats[{k}][{i}]*tmpgradu[{k}]'
	                                              .format(i=i, k=k)
	                                              for k in range(ndims))});
	% endfor

% endfor

// Get velocity gradients and TKE production term

fpdtype_t prod = 0.0;
fpdtype_t rho = u[0];
fpdtype_t rcprho = 1/u[0];
fpdtype_t duk_dxj, duj_dxk;

fpdtype_t ku = u[${nvars-2}];
fpdtype_t wu = u[${nvars-1}];

fpdtype_t mu_t = (rho*ku/wu < 0.0) ? 0.0 : rho*ku/wu;
mu_t = ${c['tmswitch']}*(1.0 - exp(-${c['tdvc']}*(t - ${c['tmstarttime']})))*mu_t;
mu_t = (mu_t > ${c['mu']}*${c['max_mutrat']}) ? ${c['mu']}*${c['max_mutrat']} : mu_t;

fpdtype_t betaprime = ${c['alpha']}*${c['betastar']} - ${c['alpha']}*${c['betastar']}/${c['fw']} + ${c['beta']}/${c['fw']} ;

// DEBUGGING CODE
fpdtype_t Sjk = 0.0;
fpdtype_t Tjk = 0.0;
fpdtype_t trc = 0.0;
fpdtype_t dui_dxi;

fpdtype_t ku_temp = (ku < ${c['min_ku']}) ? ${c['min_ku']} : ku;

% for i in range(ndims):
	dui_dxi = rcprho*(gradu[${i}][${i+1}] - gradu[${i}][0]*u[${i+1}]); 
	trc += dui_dxi;
% endfor

% for j,k in pyfr.ndrange(ndims,ndims):
	duk_dxj = rcprho*(gradu[${j}][${k+1}] - gradu[${j}][0]*u[${k+1}]); // duk_dxj = 1/rho*(drhouk_dxj - drho_dxj*uk)
	duj_dxk = rcprho*(gradu[${k}][${j+1}] - gradu[${k}][0]*u[${j+1}]); // duj_dxk = 1/rho*(drhouj_dxk - drho_dxk*uj)

	Sjk = 0.5*(duk_dxj + duj_dxk);
	% if (j == k):
		Tjk = rcprho*mu_t*(2*Sjk - ${2.0/3.0}*trc) - ${2.0/3.0}*ku_temp;
	% else:
		Tjk = rcprho*mu_t*(2*Sjk);
	% endif
	prod += duj_dxk*Tjk; // CHECK THESIS?
% endfor
// END DEBUGGING CODE

fpdtype_t prod_u = ${c['fk']}*prod + ${c['betastar']}*ku*wu*(1.0 - 1.0/${c['fw']});



// Calculate ku and wu source terms

ku_src = ${c['tmswitch']}*(rho*prod_u - ${c['betastar']}*rho*ku*wu);
wu_src = ${c['tmswitch']}*(${c['alpha']}*rho*prod_u*wu/ku - betaprime*wu*wu);



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
