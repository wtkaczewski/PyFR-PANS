# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements(object):
    formulations = ['std', 'dual']

    privarmap = {2: ['rho', 'u', 'v', 'p', 'ku', 'wu'],
                 3: ['rho', 'u', 'v', 'w', 'p', 'ku', 'wu']}

    convarmap = {2: ['rho', 'rhou', 'rhov', 'E', 'ku', 'wu'],
                 3: ['rho', 'rhou', 'rhov', 'rhow', 'E', 'ku', 'wu']}

    dualcoeffs = convarmap

    visvarmap = {
        2: [('density', ['rho']),
            ('velocity', ['u', 'v']),
            ('pressure', ['p']),
            ('K_u', ['ku']),
            ('Omega_u', ['wu'])],
        3: [('density', ['rho']),
            ('velocity', ['u', 'v', 'w']),
            ('pressure', ['p']),
            ('K_u', ['ku']),
            ('Omega_u', ['wu'])]
    }


    @staticmethod
    def pri_to_con(pris, cfg, addvars=False):
        if addvars:
            rho, p = pris[0], pris[-5]
            ku, wu = pris[-4], pris[-3]
            F1, fk = pris[-2], pris[-1]

            # Multiply velocity components by rho
            rhovs = [rho*c for c in pris[1:-5]]

            # Compute the energy
            gamma = cfg.getfloat('constants', 'gamma')
            E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-5])

            return [rho] + rhovs + [E] + [ku,wu] + [F1, fk]
        else:
            rho, p = pris[0], pris[-3]
            ku, wu = pris[-2], pris[-1]

            # Multiply velocity components by rho
            rhovs = [rho*c for c in pris[1:-3]]

            # Compute the energy
            gamma = cfg.getfloat('constants', 'gamma')
            E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-3])

            return [rho] + rhovs + [E] + [ku,wu]
    @staticmethod
    def con_to_pri(cons, cfg, addvars=False):
        if addvars:
            rho, E = cons[0],  cons[-5]
            ku, wu = cons[-4], cons[-3]
            F1, fk = cons[-2], cons[-1]
            # Divide momentum components by rho
            vs = [rhov/rho for rhov in cons[1:-5]]
            # Compute the pressure
            gamma = cfg.getfloat('constants', 'gamma')
            p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))
            return [rho] + vs + [p] + [ku,wu] + [F1, fk]
        else:
            rho, E = cons[0],  cons[-3]
            ku, wu = cons[-2], cons[-1]
            # Divide momentum components by rho
            vs = [rhov/rho for rhov in cons[1:-3]]
            # Compute the pressure
            gamma = cfg.getfloat('constants', 'gamma')
            p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))
            return [rho] + vs + [p] + [ku,wu]





class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, backend, nscalupts, nonce):
        super().set_backend(backend, nscalupts, nonce)

        # Register our flux kernel
        backend.pointwise.register('pyfr.solvers.euler.kernels.tflux')

        # Template parameters for the flux kernel
        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       c=self.cfg.items_as('constants', float))

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts
            )
