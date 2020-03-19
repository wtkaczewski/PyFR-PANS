# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict
import os
import re
import sys 

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where, subclasses
from pyfr.writers import BaseWriter

# add tecplot writer 
# only support serial I/O 
# grammar note: python can add attributes anytime
# add surface geometries
# for surface geometries need additional variables
# cp and cf
# add q-criterion as well as surface output
# add vorticity x y z

class TECWriter(BaseWriter):
    # Supported file types and extensions
    name = 'tec'
    extn = ['.dat', '.plt']

    def __init__(self, args):
        super().__init__(args)

        self.dtype = np.dtype(args.precision).type
        self.divisor = args.divisor or self.cfg.getint('solver', 'order')
        # Solutions need a separate processing pipeline to other data
        if self.dataprefix == 'soln':
            self._pre_proc_fields = self._pre_proc_fields_soln
            self._post_proc_fields = self._post_proc_fields_soln
            self._soln_fields = list(self.elementscls.privarmap[self.ndims])
            self._tec_vars = list(self.elementscls.visvarmap[self.ndims])
        # Otherwise we're dealing with simple scalar data
        else:
            self._pre_proc_fields = self._pre_proc_fields_scal
            self._post_proc_fields = self._post_proc_fields_scal
            self._soln_fields = self.stats.get('data', 'fields').split(',')
            self._tec_vars = [(k, [k]) for k in self._soln_fields]

        # See if we are computing gradients
        if args.gradients:
            self._pre_proc_fields_ref = self._pre_proc_fields
            self._pre_proc_fields = self._pre_proc_fields_grad
            self._post_proc_fields = self._post_proc_fields_grad

            # Update list of solution fields
            self._soln_fields.extend(
                '{0}-{1}'.format(f, d)
                for f in list(self._soln_fields) for d in range(self.ndims)
            )

            # Update the list of tec variables to solution fields
            nf = lambda f: ['{0}-{1}'.format(f, d) for d in range(self.ndims)]
            for var, fields in list(self._tec_vars):
                if len(fields) == 1:
                    self._tec_vars.append(('grad ' + var, nf(fields[0])))
                else:
                    self._tec_vars.extend(
                        ('grad {0} {1}'.format(var, f), nf(f)) for f in fields
                    )
        # add q here


        #check and extract bcs
	#check if we need to compute viscous force
        self._viscous = 'navier-stokes' in self.cfg.get('solver','system')
        self._ac      = 'ac' in self.cfg.get('solver','system')
        self._viscorr = self.cfg.get('solver','viscosity-correction','none')
        self._mu      = self.cfg.get('constants','mu')
        self._constant= self.cfg.items('constants')
        self._order   = self.cfg.get('solver','order')
        
        self._gradients = args.gradients
    
        #if self._viscous:
            # add q here

        # grab the bc
        self._allparts = list(self.mesh)
        self._bcparts  = []
        for i in range(len(self._allparts)):
            if 'bcon_' in self._allparts[i]:
                self._bcparts.append(self._allparts[i])
        
        # interpolation matrices and quadrature weights
        
        if self._viscous:
            self._m4 = m4 = {}
            rcpjact  = {}
        # cheat the solver to output bc surface elements
        # the value does not matter
        if self.ndims == 3:
            self.cfg.set('solver-elements-quad',
                            'soln-pts','gauss-legendre')
            self.cfg.set('solver-elements-tri',
                            'soln-pts','williams-shunn')
        elif self.ndims == 2:
            self.cfg.set('solver-elements-line',
                            'soln-pts','gauss-legendre')
	
    def _pre_proc_fields_soln(self, name, mesh, soln):
        # Convert from conservative to primitive variables
        return np.array(self.elementscls.con_to_pri(soln, self.cfg))

    def _pre_proc_fields_scal(self, name, mesh, soln):
        return soln

    def _post_proc_fields_soln(self, vsoln):
        # Primitive and visualisation variable maps
        privarmap = self.elementscls.privarmap[self.ndims]
        visvarmap = self.elementscls.visvarmap[self.ndims]

        # Prepare the fields
        fields = []
        for fnames, vnames in visvarmap:
            ix = [privarmap.index(vn) for vn in vnames]

            fields.append(vsoln[ix])

        return fields

    def _post_proc_fields_scal(self, vsoln):
        return [vsoln[self._soln_fields.index(v)] for v, _ in self._tec_vars]

    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Call the reference pre-processor
        soln = self._pre_proc_fields_ref(name, mesh, soln)

        # Dimensions
        nvars, nupts = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.swapaxes(0, 1).reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.vstack([soln, gradsoln])

    def _post_proc_fields_grad(self, vsoln):
        # Prepare the fields
        fields = []
        for vname, vfields in self._tec_vars:
            ix = [self._soln_fields.index(vf) for vf in vfields]

            fields.append(vsoln[ix])

        return fields

    def _get_npts_ncells_nnodes(self, mk):
        m_inf = self.mesh_inf[mk]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=m_inf[0])
        subdvcls = subclass_where(BaseShapeSubDiv, name=m_inf[0])

        # Number of vis points
        npts = shapecls.nspts_from_order(self.divisor + 1)*m_inf[1][1]

        # Number of sub cells and nodes
        ncells = len(subdvcls.subcells(self.divisor))*m_inf[1][1]

        # nnodes will be n(mesh cell)*subcells*nodes(per subcell)
        nnodes = len(subdvcls.subnodes(self.divisor))*m_inf[1][1]

        return npts, ncells, nnodes

    def _eval_qcri_vorts(self):
        # evaluate qcriterion and vorticity
        s=1

    def stress_tensor(self,u,du):
        # values are alrady converted to primitive variables
        c = self._constant
        # pressure
        rho,P = u[0],u[-1]
        # Gradient of velocity and viscosity
        gradu, mu = du[:,:, 1:-1,:].swapaxes(1,2), float(c['mu'])
        # Bulk tensor
        bulk = np.eye(self.ndims)[:, :, None, None]*np.trace(gradu)

        if self._viscorr == 'sutherland':
            cpT  = c['gamma']*P/(rho*(c['gamma']-1.0))
            Trat = cpT/c['cpTref']
            mu  *= (c['cpTref']+c['cpTs'])*Trat**1.5/(cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def ac_stress_tensor(self, du):
        # Gradient of velocity and kinematic viscosity
        gradu, nu = du[:, 1:].swapaxes(1,2), self._constants['nu']

        return -nu*(gradu + gradu.swapaxes(0, 1))

    @memoize
    def _get_shape(self, name, nspts):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, self.cfg)

    @memoize
    def _get_std_ele(self, name, nspts):
        return self._get_shape(name, nspts).std_ele(self.divisor)

    @memoize
    def _get_mesh_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    @memoize
    def _get_soln_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)
    
    # for tecplot only support serial output
    def write_out(self):
        name, extn = os.path.splitext(self.outf)
        parts = defaultdict(list)
        for mk, sk in zip(self.mesh_inf, self.soln_inf):
            pfn =  self.outf
            parts[pfn].append((mk, sk))

	# link parts
        # use another file since many variables
        # are only caculated on the surface

        bparts= defaultdict(list)
        for mk in self._bcparts:
            pfn = 'surface_{0}'.format(self.outf)
            bparts[pfn].append(mk)

        print(bparts,self._bcparts)
        write_s_to_fh = lambda s: fh.write(s.encode('utf-8'))
        
        # output the volume data
        for pfn, misil in parts.items():
            with open(pfn, 'wb') as fh:
                # write the title
                write_s_to_fh('TITLE = PYFR_TECPLOT_OUTPUT\n')
		# get the variable list and 
                zonehead  = ' '.join(['"x{0}" '.format(d+1) 
                               for d in range(self.ndims)])
                zonehead += ' '.join(f'"{me2}"'
                               for atr in self._tec_vars for me2 in atr[1])
                write_s_to_fh('VARIABLES = {0}\n'.format(zonehead))
                # now assuming each zone has single type of elements
                off = 0
                for mk, sk in misil:
                    off = self._write_serial_header(fh, mk, off)
		    # write data for each name
                    #self._write_data(fh, mk, sk)

        # output the surface data
        # need to extract the surface data
        if self.ndims == 3:
            for pfn,pnm in bparts.items():
                # extract the p information
                with open(pfn,'wb') as fh:
		    # write the header for bc face
                    write_s_to_fh('TITLE = PYFR_TECSURFACE_OUTPUT\n')
		    # get the variable list and 
                    zonehead  = ' '.join(['"x{0}" '.format(d+1) 
                               for d in range(self.ndims)])
                    zonehead += ' '.join(f'"{me2}"'
                               for atr in self._tec_vars for me2 in atr[1])
                    # add the forces in x y z direction
                    if self._viscous:
                        zonehead  += ' '.join(['" f{0}" '.format(d+1) 
                               for d in range(self.ndims)])
                    write_s_to_fh('VARIABLES = {0}\n'.format(zonehead))
                    for mk in pnm:
                        self._write_bc_data(fh,mk)
	
	# here write another file for surface cp and cf

    def _write_darray(self, array, tecf, dtype):
        # need formatted output
        array = array.astype(dtype)
        outfnt = '%15.7f' if (dtype is np.float32 or dtype is  np.float64) else '%10d'
        for i in range(array.shape[0]):
            np.savetxt(tecf,array[i],fmt=outfnt, delimiter=' ')
            

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_serial_header(self, tecf, mk, off):
        npts, ncells, nnodes = self._get_npts_ncells_nnodes(mk)

        write_s = lambda s: tecf.write(s.encode('utf-8'))
        if self.ndims == 3:
            zonetype = 'FETETRAHEDRON' if 'tet' in mk else 'FEBRICK'
        else: 
            zonetype = 'FEQUADRILATERAL'
        #datapack = 'BLOCK' if 'tet' in mk else 'POINT'
        datapack = 'POINT' 
	# write the zone header
        write_s('Zone T="Interior-mesh-{0}" N={1}, E={2}, '
		'DATAPACKING={3}, ZONETYPE={4}\n'
		.format(mk,npts,ncells,datapack,zonetype))

        return off
    

    def _load_elem_map(self, mesh, prank):
        # get the element map
        # for all the volume elements in the prank
        basismap = {b.name: b for b in subclasses(BaseShape,just_leaf=True) }
        # Look for and load each element type from the mesh
        elemap = OrderedDict()
        for f in mesh:
            m = re.match('spt_(.+?)_{0}'.format(prank),f)
            if m:
                # Element type
                t = m.group(1)
                elemap[t] = self.systemscls.elementscls(basismap[t], mesh[f], self.cfg)
        
        return elemap
        
    def _write_bc_data(self, tecf, mk):
        # mesh of this bc
        # each bc is grouped  
        msh = self.mesh[mk].astype('U4,i4,i1,i1')
        # use process id to group the solution
        prank = re.search('p[0-9]+',mk).group(0)
        
        neles = msh.shape[0]

        if 'sphere' in mk:
            sss = 0

        # corresponding solution
        # all variables to primitive variables
        psoln = [];
        for sk in self.soln:
            if sk[-len(prank):] == prank:
                #psoln.append(self.soln[sk].astype(self.dtype))
                soltmp = self.soln[sk].swapaxes(0,1).astype(self.dtype)
                # get the corresponding mesh
                msh2 = self.mesh[sk.replace('soln','spt')]
                name = self.mesh_inf[sk.replace('soln','spt')][0]
                soltmp = self._pre_proc_fields(name,msh2,soltmp).swapaxes(0,1)
                psoln.append(soltmp)
        
        # corresponding mesh
        pmesh = [];
        for mk2 in self.mesh:
            if mk2[-len(prank):] == prank and 'spt' in mk2:
                pmesh.append(self.mesh[mk2].swapaxes(1,2))

        elemap = self._load_elem_map(self.mesh,prank)
                            
        eidxs = defaultdict(list)
        norms = defaultdict(list)

        if self._viscous:
            m4 = {}
            rcpjact = {}
            smatloc = {}
            pdjac   = {}

        mt2 = {}
        m02 = {}

        for etype, eidx, fidx, flags in msh:
            eles = elemap[etype]
            if (etype, fidx) not in m02:
                bcname = eles.basis.faces[fidx][0]
                subdvcls = subclass_where(BaseShapeSubDiv, name = bcname)
                nodes = subdvcls.subnodes(self.divisor)
                nfpts = len(np.unique(nodes))
                proj   = eles.basis.faces[fidx][1]
                sfpts  = self._get_std_ele(bcname, nfpts)
                vfpts  = _proj_pts(proj,np.array(sfpts)) 
                vvfpts  = [a for a in tuple(map(tuple,vfpts))]

                mt2[etype,fidx] = eles.basis.sbasis.nodal_basis_at(vvfpts).astype(self.dtype)
                m02[etype,fidx] = eles.basis.ubasis.nodal_basis_at(vvfpts)
            
            # this need further test
            if self._viscous and etype not in m4:
                m4[etype] = eles.basis.m4

                # get the smats at the solution points
                smat = eles.smat_at_np('upts').transpose(2,0,1,3)
                # get |J|^-1 at the solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # product to get J^-T
                rcpjact[etype] = smat*rcpdjac

                smatloc[etype] = smat
                pdjac  [etype] = 1.0/rcpdjac
            # Unit physical normals and their magnitudes (including |J|)
            # this need to be modified
            npn = eles.get_norm_pnorms(eidx, fidx)
            mpn = eles.get_mag_pnorms(eidx, fidx)

            eidxs[etype, fidx].append(eidx)
            norms[etype, fidx].append(mpn[:, None]*npn)
	
        if self._viscous:
            self._m4    = m4 
        
        self._m02   = m02

        self._mt2    = mt2

        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
        self._norms = {k: np.array(v) for k, v in norms.items()}

        if self._viscous:
            self._rcpjact = {k: rcpjact[k[0]][..., v]
                                 for k, v in self._eidxs.items()}
            self._smatloc = {k: smatloc[k[0]][..., v]
                                 for k, v in self._eidxs.items()}
            self._pdjac   = {k: pdjac[k[0]][...,v]
                                 for k, v in self._eidxs.items()}
        
        ele_types = list(elemap)	
        # solution matrices indexed by element type
        solns = dict(zip(ele_types,psoln))
        meshs = dict(zip(ele_types,pmesh))
    
        ndims, nvars = self.ndims, self.nvars

        #f = np.zeros(2*ndims if self._viscous else ndims)
        
        for etype, fidx in self._m02:
            # each type write a header

            # Get the interpolation operator
            m02 = self._m02[etype, fidx]
            mt2 = self._mt2[etype, fidx]
            nfpts,  nupts = m02.shape
            nfpts,  nmpts = mt2.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]
            # extract the relevant elements from the geometry
            mmpts = meshs[etype][..., self._eidxs[etype, fidx]]
            
            neles = self._eidxs[etype,fidx].shape[0]

            # Interpolate to the face
            ufpts2 = m02 @ uupts.reshape(nupts, -1)
            if self._gradients:
                ufpts2 = ufpts2.reshape(nfpts, nvars*(1+ndims), -1)
            else:
                ufpts2 = ufpts2.reshape(nfpts, nvars, -1)
            ufpts2 = ufpts2.swapaxes(0, 1)
            
            # Interpolate coordinates to the face
            mfpts2 = mt2 @ mmpts.reshape(nmpts,-1)
            mfpts2 = mfpts2.reshape(nfpts,ndims,-1)
            mfpts2 = mfpts2.swapaxes(0,1)
            # if possible also evaluate the distance
            # for y plus calculation

            # following the same approach for volume elements
            bcname = elemap[etype].basis.faces[fidx][0]
            sfpts = self._get_std_ele(bcname,nfpts)
            nsfpts = len(sfpts)
            
            bcfpts = np.rollaxis(mfpts2,0,3)
            bcsoln = ufpts2

            bcmeshsol = np.concatenate((bcfpts.swapaxes(0,1),bcsoln.swapaxes(0,2)), axis = 2)
            
            subdvcls = subclass_where(BaseShapeSubDiv, name = bcname)
            nodes = subdvcls.subnodes(self.divisor)

            tec_con = np.tile(nodes,(neles,1))
            tec_con += (np.arange(neles)*nsfpts)[:,None]

            d1 = len(subdvcls.subcells(self.divisor))
            d2 = len(nodes)//d1

            tec_con = tec_con.reshape(-1,d1,d2)
            
            ntpts  = bcmeshsol.shape[0]*bcmeshsol.shape[1]
            ncells = neles*d1
            # before proceed need to do some modification for 
            # tet -> extand to 4 points
            if bcname == 'tri':
                vtu_con = np.insert(tec_con,3,tec_con[:,:,2],axis = 2)

            #calculate the total number of nodes and total number of 
            #subcells

            if self._viscous:
                # sphere can be used to verify 
                
                # Get operator and J^-T matrix
                m4 = self._m4[etype]
                rcpjact = self._rcpjact[etype, fidx]

                # Transformed gradient at solution points
                # only use the primitive variables
                # exclude the already calculated gradients if any
                tduupts = m4 @ uupts[:,0:nvars,:].reshape(nupts, -1)
                tduupts = tduupts.reshape(ndims, nupts, nvars, -1)

                # Physical gradient at solution points
                duupts = np.einsum('ijkl,jkml->ikml',rcpjact,tduupts)
                duupts = duupts.reshape(ndims, nupts,-1)

                # Interpolate gradient to visualization points
                dufpts = np.array([m02 @ du for du in duupts])
                dufpts = dufpts.reshape(ndims, nfpts, nvars, -1)

                # get the smat at flux points
                # interpolate smat to surface
                usmat = self._smatloc[etype,fidx]
                usmat = usmat.reshape(ndims*ndims,nupts,-1)

                fsmat = np.array([m02 @ smat for smat in usmat])
                fsmat = fsmat.reshape(ndims,ndims,nfpts,-1)

                normf = np.array(elemap[etype].basis.faces[fidx][2])
                normvpts = np.array([normf,]*nsfpts)
            
                fsmat = np.einsum('ijkl->klij', fsmat)

                pnorm_vfpts = np.einsum('ijkl,il->ijk',fsmat,normvpts)
                mag_pnorm=np.einsum('...i,...i',pnorm_vfpts,pnorm_vfpts)
                mag_pnorm=np.sqrt(mag_pnorm)
                pnorm_vfpts = pnorm_vfpts/mag_pnorm[...,None]

                # Viscous stress
                if self._ac:
                    vis = self.ac_stress_tensor(dufpts)
                else:
                    vis = self.stress_tensor(bcsoln, dufpts)
                
                vis = np.einsum('ijkl->klij', vis)
                # no need to do the quadrature
                forces =  np.einsum('ijkl,ijl->ijk',vis,pnorm_vfpts)
                # verify the force
            
            # append the forces into bcmeshsol
            if self._viscous:
                bcmeshsol = np.concatenate((bcmeshsol,forces.swapaxes(0,1)),axis = 2)

            write_s = lambda s: tecf.write(s.encode('utf-8'))
            write_s('Zone T="{0}-{1}-f{2}" N={3}, E={4}, '
		    'DATAPACKING=POINT, ZONETYPE=FEQUADRILATERAL\n'
		     .format(mk,etype,fidx,ntpts,ncells))
            self._write_darray(bcmeshsol,tecf,self.dtype)
            self._write_darray(tec_con+1,tecf,np.int32)

    def _write_data(self, tecf, mk, sk):
        name = self.mesh_inf[mk][0]
        mesh = self.mesh[mk].astype(self.dtype)
        soln = self.soln[sk].swapaxes(0, 1).astype(self.dtype)

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = self._get_std_ele(name, nspts)
        nsvpts = len(svpts)

        # Generate the operator matrices
        mesh_tec_op = self._get_mesh_op(name, nspts, svpts)
        soln_tec_op = self._get_soln_op(name, nspts, svpts)

        # Calculate node locations of elements
        vpts = mesh_tec_op @ mesh.reshape(nspts, -1)
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Pre-process the solution
        soln = self._pre_proc_fields(name, mesh, soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = soln_tec_op @ soln.reshape(len(soln), -1)
        vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)

        meshsol = np.concatenate((vpts.swapaxes(0,1),
                                  vsoln.swapaxes(0,2)),
                                  axis= 2)

        self._write_darray(meshsol, tecf, self.dtype)

        # Perform the sub division
        subdvcls = subclass_where(BaseShapeSubDiv, name=name)
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare tec cell arrays
        tec_con = np.tile(nodes, (neles, 1))
        tec_con += (np.arange(neles)*nsvpts)[:, None]
        
        d1 = len(subdvcls.subcells(self.divisor))
        d2 = len(nodes)//d1
        tec_con = tec_con.reshape(-1,d1,d2)

        if d2 == 3:
            tec_con = np.insert(tec_con,3,tec_con[:,:,2],axis = 2)
        elif d2 == 5:
            # need more test here
            tec_con = np.insert(tec_con,3,tec_con[:,:,2],axis = 2)
            #tec_con = np.insert(tec_con,3,tec_con[:,:,2],axis = 2)
        elif d2 == 6:
            tec_con = np.insert(tec_con,3,tec_con[:,:,2],axis = 2)
            tec_con = np.insert(tec_con,7,tec_con[:,:,6],axis = 2)

        # Write tec node connectivity
        self._write_darray(tec_con+1, tecf, np.int32)


class BaseShapeSubDiv(object):
    tec_types = dict(tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    tec_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)

    @classmethod
    def subcells(cls, n):
        pass

    @classmethod
    def subcelloffs(cls, n):
        return np.cumsum([cls.tec_nodes[t] for t in cls.subcells(n)])

    @classmethod
    def subcelltypes(cls, n):
        return np.array([cls.tec_types[t] for t in cls.subcells(n)])

    @classmethod
    def subnodes(cls, n):
        pass


class TensorProdShapeSubDiv(BaseShapeSubDiv):
    @classmethod
    def subnodes(cls, n):
        conbase = np.array([0, 1, n + 2, n + 1])

        # Extend quad mapping to hex mapping
        if cls.ndim == 3:
            conbase = np.hstack((conbase, conbase + (1 + n)**2))

        # Calculate offset of each subdivided element's nodes
        nodeoff = np.zeros((n,)*cls.ndim, dtype=np.int)
        for dim, off in enumerate(np.ix_(*(range(n),)*cls.ndim)):
            nodeoff += off*(n + 1)**dim

        # Tile standard element node ordering mapping, then apply offsets
        internal_con = np.tile(conbase, (n**cls.ndim, 1))
        internal_con += nodeoff.T.flatten()[:, None]

        return np.hstack(internal_con)


class QuadShapeSubDiv(TensorProdShapeSubDiv):
    name = 'quad'
    ndim = 2

    @classmethod
    def subcells(cls, n):
        return ['quad']*(n**2)


class HexShapeSubDiv(TensorProdShapeSubDiv):
    name = 'hex'
    ndim = 3

    @classmethod
    def subcells(cls, n):
        return ['hex']*(n**3)


class TriShapeSubDiv(BaseShapeSubDiv):
    name = 'tri'

    @classmethod
    def subcells(cls, n):
        return ['tri']*(n**2)

    @classmethod
    def subnodes(cls, n):
        conlst = []

        for row in range(n, 0, -1):
            # Lower and upper indices
            l = (n - row)*(n + row + 3) // 2
            u = l + row + 1

            # Base offsets
            off = [l, l + 1, u, u + 1, l + 1, u]

            # Generate current row
            subin = np.ravel(np.arange(row - 1)[..., None] + off)
            subex = [ix + row - 1 for ix in off[:3]]

            # Extent list
            conlst.extend([subin, subex])

        return np.hstack(conlst)


class TetShapeSubDiv(BaseShapeSubDiv):
    name = 'tet'

    @classmethod
    def subcells(cls, nsubdiv):
        return ['tet']*(nsubdiv**3)

    @classmethod
    def subnodes(cls, nsubdiv):
        conlst = []
        jump = 0

        for n in range(nsubdiv, 0, -1):
            for row in range(n, 0, -1):
                # Lower and upper indices
                l = (n - row)*(n + row + 3) // 2 + jump
                u = l + row + 1

                # Lower and upper for one row up
                ln = (n + 1)*(n + 2) // 2 + l - n + row
                un = ln + row

                rowm1 = np.arange(row - 1)[..., None]

                # Base offsets
                offs = [(l, l + 1, u, ln), (l + 1, u, ln, ln + 1),
                        (u, u + 1, ln + 1, un), (u, ln, ln + 1, un),
                        (l + 1, u, u+1, ln + 1), (u + 1, ln + 1, un, un + 1)]

                # Current row
                conlst.extend(rowm1 + off for off in offs[:-1])
                conlst.append(rowm1[:-1] + offs[-1])
                conlst.append([ix + row - 1 for ix in offs[0]])

            jump += (n + 1)*(n + 2) // 2

        return np.hstack([np.ravel(c) for c in conlst])


class PriShapeSubDiv(BaseShapeSubDiv):
    name = 'pri'

    @classmethod
    def subcells(cls, n):
        return ['pri']*(n**3)

    @classmethod
    def subnodes(cls, n):
        # Triangle connectivity
        tcon = TriShapeSubDiv.subnodes(n).reshape(-1, 3)

        # Layer these rows of triangles to define prisms
        loff = (n + 1)*(n + 2) // 2
        lcon = [[tcon + i*loff, tcon + (i + 1)*loff] for i in range(n)]

        return np.hstack([np.hstack(l).flat for l in lcon])


class PyrShapeSubDiv(BaseShapeSubDiv):
    name = 'pyr'

    @classmethod
    def subcells(cls, n):
        cells = []

        for i in range(n, 0, -1):
            cells += ['pyr']*(i**2 + (i - 1)**2)
            cells += ['tet']*(2*i*(i - 1))

        return cells

    @classmethod
    def subnodes(cls, nsubdiv):
        lcon = []

        # Quad connectivity
        qcon = [QuadShapeSubDiv.subnodes(n + 1).reshape(-1, 4)
                for n in range(nsubdiv)]

        # Simple functions
        def _row_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*i + j + 1)
                             for i in range(a, n + b)
                             for j in range(n - 1)])

        def _col_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*(i + 1) + j)
                             for i in range(n - 1)
                             for j in range(a, n + b)])

        u = 0
        for n in range(nsubdiv, 0, -1):
            l = u
            u += (n + 1)**2

            lower_quad = qcon[n - 1] + l
            upper_pts = np.arange(n**2) + u

            # First set of pyramids
            lcon.append([lower_quad, upper_pts])

            if n > 1:
                upper_quad = qcon[n - 2] + u
                lower_pts = np.hstack([range(k*(n + 1)+1, (k + 1)*n + k)
                                       for k in range(1, n)]) + l

                # Second set of pyramids
                lcon.append([upper_quad[:, ::-1], lower_pts])

                lower_row = _row_in_quad(n + 1, 1, -1) + l
                lower_col = _col_in_quad(n + 1, 1, -1) + l

                upper_row = _row_in_quad(n) + u
                upper_col = _col_in_quad(n) + u

                # Tetrahedra
                lcon.append([lower_col, upper_row])
                lcon.append([lower_row[:, ::-1], upper_col])

        return np.hstack([np.column_stack(l).flat for l in lcon])

def _proj_pts(projector, pts):
    pts = np.atleast_2d(pts.T)
    return np.vstack(np.broadcast_arrays(*projector(*pts))).T 
