# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    # Use the density field for shock sensing
    shockvar = 'rho'
    
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts','scal_upts', 'vect_upts'}

        if 'div-flux' in self.antialias:
            bufs |= {'scal_qpts_cpy'}
        else:
            bufs |= {'scal_upts_cpy'}

        return bufs
    
    def set_backend(self, backend, nscalupts, nonce):
        super().set_backend(backend, nscalupts, nonce)
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')

        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')

        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       shock_capturing=shock_capturing, visc_corr=visc_corr,
                       c=self.cfg.items_as('constants', float))

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts, artvisc=self.artvisc
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts, artvisc=self.artvisc
            )


        # NEW KERNELS FOR PANS ------------------------------------------
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.negdivconfpans')

        fluxaa = 'flux' in self.antialias
        divfluxaa = 'div-flux' in self.antialias

        srctplargs = {
            'ndims':    self.ndims,
            'nvars':    self.nvars,
            'srcex':    self._src_exprs,
            'c'    :    self.cfg.items_as('constants', float)
        }


        if divfluxaa:
            plocqpts = self.ploc_at('qpts') 
            solnqpts = self._scal_qpts_cpy

            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconfpans', tplargs=srctplargs,
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=self.rcpdjac_at('qpts'), ploc=plocqpts, u=solnqpts,
                gradu=self._vect_qpts
            )
        else:
            plocupts = self.ploc_at('upts')
            solnupts = self._scal_upts_cpy


            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconfpans', tplargs=srctplargs,
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts, 
                gradu=self._vect_upts
            )