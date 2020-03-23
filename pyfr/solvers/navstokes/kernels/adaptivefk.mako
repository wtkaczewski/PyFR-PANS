# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='adaptivefk' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              rcpdjac='in fpdtype_t[${str(nupts)}]'
              fk='out fpdtype_t'>

// Evaluate modal coefficients
fpdtype_t k_res = ${c['min_ku']}, tmp;

% for i, deg in enumerate(ubdegs):
	% for k in range(1,ndims+1):
    	tmp = ${' + '.join('{jx}*u[{j}][{k}]'.format(j=j, jx=jx, k=k)
                       for j, jx in enumerate(invvdm[i]) if jx != 0)};
		% if deg > 0:
		    k_res += tmp*tmp;
		% endif
	% endfor
% endfor

fpdtype_t djac = 0;
fpdtype_t rho = 0;
fpdtype_t wu = 0;

// Average rho, djac, and wu over element
% for i in range(nupts):
	djac += (1.0/rcpdjac[${i}])/${nupts}; 
	rho  += u[${i}][0]/${nupts};
	wu   += exp(u[${i}][${nvars-1}])/${nupts};
% endfor


// Calculate minimum turbulent length scale = (nu^3/epsilon)^(1/4) = (nu^3/(betastar*k*w))^(1/4)
fpdtype_t dx_max = 1/pow(8.0*djac, 0.33); // 8.0 since base element is 2.0^3 volume
fpdtype_t l_min = pow(((pow(${c['mu']}/rho, 3))/(${c['betastar']}*k_res*wu)), 0.25);

// Calculate variable fk and assume fw = 1/fk
fk = min(1.0, max(${c['min_fk']}, pow(${c['C_PANS']}*(dx_max/l_min), 2.0/3.0)));

</%pyfr:kernel>



