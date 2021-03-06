# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    # Use the density field for shock sensing
    shockvar = 'rho'
    
    @property
    def _scratch_bufs(self):
        bufs = {'scal_fpts', 'vect_fpts', 'scal_upts', 'vect_upts'}

        if 'div-flux' in self.antialias:
            bufs |= {'scal_qpts_cpy'}
        else:
            bufs |= {'scal_upts_cpy'}

        if 'flux' in self.antialias:
            bufs |= {'scal_qpts', 'vect_qpts'}

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

        # ----- NEW KERNELS FOR PANS -----
        
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.negdivconfpans')
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.gradcorupans')
        
        self.ku_src = self._be.matrix((self.nupts, self.neles), tags={'align'})
        self.wu_src = self._be.matrix((self.nupts, self.neles), tags={'align'})
        self.F1     = self._be.matrix((self.nupts, self.neles), tags={'align'}, extent= nonce + 'F1')

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts, artvisc=self.artvisc,
                F1=self.F1
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts, artvisc=self.artvisc,
                F1=self.F1
            )




        srctplargs = {
            'ndims':    self.ndims,
            'nvars':    self.nvars,
            'srcex':    self._src_exprs,
            'c'    :    self.cfg.items_as('constants', float),
            'geo'   :    self.cfg.get('solver', 'geometry')
        }


        # ----- GRADCORU KERNELS -----
        
        self.kernels['gradcoru_upts'] = lambda: backend.kernel(
            'gradcorupans', tplargs=srctplargs,
             dims=[self.nupts, self.neles], smats=self.smat_at('upts'),
             rcpdjac=self.rcpdjac_at('upts'), gradu=self._vect_upts,
             u=self.scal_upts_inb, ku_src=self.ku_src, wu_src=self.wu_src,
             ploc=self.ploc_at('upts'), F1=self.F1
        )

        # ----- NEGDIVCONF KERNELS -----

        # Possible optimization when scal_upts_inb.active != scal_upts_outb.active -- Generate two negdivconf kernels (upts and upts_cpy) and let rhs() decide which one to call 

        if 'div-flux' in self.antialias:
            plocqpts = self.ploc_at('qpts') 
            solnqpts = self._scal_qpts_cpy

            self.kernels['copy_soln'] = lambda: backend.kernel(
                'copy', self._scal_qpts_cpy, self._scal_qpts
            )

            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconfpans', tplargs=srctplargs,
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=self.rcpdjac_at('qpts'), ploc=plocqpts, u=solnqpts,
                ku_src=self.ku_src, wu_src=self.wu_src
            )

        else:
            plocupts = self.ploc_at('upts')
            solnupts = self._scal_upts_cpy


            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconfpans', tplargs=srctplargs,
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts, 
                ku_src=self.ku_src, wu_src=self.wu_src
            )


    def get_F1_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rmap = self._srtd_face_fpts[fidx][eidx]
        cmap = (eidx,)*nfp

        return (self.F1.mid,)*nfp, rmap, cmap

