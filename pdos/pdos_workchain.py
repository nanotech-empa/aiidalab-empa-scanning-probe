from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm.nodes.data.array import ArrayData
from aiida.orm import Int, Float, Str, Bool, List
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.orm import Code

from aiida.engine import WorkChain, ToContext, while_
from aiida.engine import submit

from aiida_cp2k.calculations import Cp2kCalculation

from apps.scanning_probe import common

from aiida.plugins import CalculationFactory
OverlapCalculation = CalculationFactory('spm.overlap')

import os
import tempfile
import shutil
import numpy as np
import copy

class PdosWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(PdosWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("slabsys_structure", valid_type=StructureData)
        spec.input("mol_structure", valid_type=StructureData)
        spec.input("pdos_lists", valid_type=List)
        spec.input("wfn_file_path", valid_type=Str, default=lambda: orm.Str(""))
        spec.input("scf_diag",
                   valid_type=orm.Bool,
                   required=False,
                   default=lambda: orm.Bool(False))        
        
        spec.input("dft_params", valid_type=Dict)
        
        spec.input("overlap_code", valid_type=Code)
        spec.input("overlap_params", valid_type=Dict)
        
        spec.outline(
            cls.setup,
            cls.run_scfs,
            cls.run_diags,
            cls.run_overlap,
            cls.finalize,
        )
        
        spec.outputs.dynamic = True
        
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )
    
    def setup(self):
        # set up mol UKS parameters
        
        self.ctx.mol_dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())
        
        self.ctx.mol_dft_params['elpa_switch'] = False # Elpa can cause problems with small systems
        
        if 'uks' in self.ctx.mol_dft_params and self.ctx.mol_dft_params['uks']:
            slab_atoms = self.inputs.slabsys_structure.get_ase()
            mol_atoms = self.inputs.mol_structure.get_ase()
            
            mol_at_tuples = [(e, *np.round(p, 2)) for e, p in zip(
                mol_atoms.get_chemical_symbols(), mol_atoms.positions)]
            
            mol_spin_up = []
            mol_spin_dw = []
            
            for i_up in self.ctx.mol_dft_params['spin_up_guess']:
                at = slab_atoms[i_up]
                at_tup = (at.symbol, *np.round(at.position, 2))
                if at_tup in mol_at_tuples:
                    mol_spin_up.append(mol_at_tuples.index(at_tup))
            
            for i_dw in self.ctx.mol_dft_params['spin_dw_guess']:
                at = slab_atoms[i_dw]
                at_tup = (at.symbol, *np.round(at.position, 2))
                if at_tup in mol_at_tuples:
                    mol_spin_dw.append(mol_at_tuples.index(at_tup))
            
            self.ctx.mol_dft_params['spin_up_guess'] = mol_spin_up
            self.ctx.mol_dft_params['spin_dw_guess'] = mol_spin_dw
            
    
    def run_scfs(self):
        self.report("Running CP2K diagonalization SCF")
        
        emax1 = float(self.inputs.overlap_params.get_dict()['--emax1'])
        nlumo2 = int(self.inputs.overlap_params.get_dict()['--nlumo2'])
        
        self.ctx.n_all_atoms = len(self.inputs.slabsys_structure.sites)
        
        slab_inputs = self.build_slab_cp2k_inputs(
                        self.inputs.slabsys_structure,
                        self.inputs.pdos_lists,
                        self.inputs.cp2k_code,
                        self.inputs.wfn_file_path.value,
                        self.inputs.dft_params.get_dict(),
                        emax1)
        self.report("slab_inputs: "+str(slab_inputs))
        
        slab_future = self.submit(Cp2kCalculation, **slab_inputs)
        self.to_context(slab_scf=slab_future)
        
        mol_inputs = self.build_mol_cp2k_inputs(
                        self.inputs.mol_structure,
                        self.inputs.cp2k_code,
                        self.ctx.mol_dft_params,
                        nlumo2)
        self.report("mol_inputs: "+str(mol_inputs))
        
        mol_future = self.submit(Cp2kCalculation, **mol_inputs)
        self.to_context(mol_scf=mol_future)        
           
    def run_overlap(self):
        
        if not common.check_if_calc_ok(self, self.ctx.slab_scf):
            return self.exit_codes.ERROR_TERMINATION
        
        if not common.check_if_calc_ok(self, self.ctx.mol_scf):
            return self.exit_codes.ERROR_TERMINATION
        
        self.report("Running overlap")
             
        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "overlap"
        inputs['code'] = self.inputs.overlap_code
        inputs['parameters'] = self.inputs.overlap_params
        inputs['parent_slab_folder'] = self.ctx.slab_scf.outputs.remote_folder
        inputs['parent_mol_folder'] = self.ctx.mol_scf.outputs.remote_folder
        
        n_machines = 4 if self.ctx.n_all_atoms < 2000 else 8
        
        inputs['metadata']['options'] = {
            "resources": {"num_machines": n_machines},
            "max_wallclock_seconds": 86400,
        } 
        
        settings = Dict(dict={'additional_retrieve_list': ['overlap.npz']})
        inputs['settings'] = settings
        
        self.report("overlap inputs: " + str(inputs))
        
        future = self.submit(OverlapCalculation, **inputs)
        return ToContext(overlap=future)
    
    def finalize(self):
        self.report("Work chain is finished")
    
    
     # ==========================================================================
    @classmethod
    def build_slab_cp2k_inputs(cls, structure, pdos_lists, code,
                          wfn_file_path, dft_params, emax):

        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "slab_scf"
        inputs['code'] = code
        inputs['file'] = {}
        
        
        atoms = structure.get_ase()  # slow
        n_atoms = len(atoms)
        
        spin_guess = None
        if dft_params['uks']:
            spin_guess = [dft_params['spin_up_guess'], dft_params['spin_dw_guess']]

        geom_f = common.make_geom_file(
            atoms, "geom.xyz", spin_guess
        )

        inputs['file']['geom_coords'] = geom_f

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])
        num_machines = 12
        if n_atoms > 500:
            num_machines = 27
        if n_atoms > 1200:
            num_machines = 48
        if n_atoms > 2400:
            num_machines = 60
        if n_atoms > 3600:
            num_machines = 75
        walltime = 86400
        
        wfn_file = ""
        if wfn_file_path != "":
            wfn_file = os.path.basename(wfn_file_path)
            
        added_mos = np.max([100, int(n_atoms*emax/5.0)])

        inp = cls.get_cp2k_input(dft_params,
                                 cell_abc,
                                 walltime*0.97,
                                 wfn_file,
                                 added_mos,
                                 atoms,
                                 pdos_lists)

        inputs['parameters'] = Dict(dict=inp)

        # settings
        settings = Dict(dict={'additional_retrieve_list': ['*.pdos']})
        inputs['settings'] = settings

        # resources
        inputs['metadata']['options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
            "parser_name": "cp2k_advanced_parser",
        }
        if wfn_file_path != "":
            inputs['metadata']['options']["prepend_text"] = "cp %s ." % wfn_file_path
        
        return inputs
    
    # ==========================================================================
    @classmethod
    def build_mol_cp2k_inputs(cls, structure, code, dft_params, nlumo):

        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "mol_scf"
        inputs['code'] = code
        inputs['file'] = {}
                
        atoms = structure.get_ase()  # slow
        n_atoms = len(atoms)
        
        spin_guess = None
        if dft_params['uks']:
            spin_guess = [dft_params['spin_up_guess'], dft_params['spin_dw_guess']]

        geom_f = common.make_geom_file(
            atoms, "geom.xyz", spin_guess
        )

        inputs['file']['geom_coords'] = geom_f

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])
        num_machines = 12
        if len(atoms) > 200:
            num_machines = 27
        if len(atoms) > 1000:
            num_machines = 48
        walltime = 86400

        inp = cls.get_cp2k_input(dft_params,
                                 cell_abc,
                                 walltime*0.97,
                                 "",
                                 nlumo+2,
                                 atoms)

        inputs['parameters'] = Dict(dict=inp)

        # settings
        #settings = ParameterData(dict={'additional_retrieve_list': ['aiida-RESTART.wfn', 'BASIS_MOLOPT', 'aiida.inp']})
        #inputs['settings'] = settings

        # resources
        inputs['metadata']['options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
            "parser_name": "cp2k_advanced_parser",
        }
        
        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, dft_params, cell_abc, walltime, wfn_file, added_mos, atoms, pdos_lists=None):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'ENERGY',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'MEDIUM',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'FORCE_EVAL': cls.get_force_eval_qs_dft(dft_params, cell_abc,
                                                    wfn_file, added_mos, atoms, pdos_lists),
        }
        
        if dft_params['elpa_switch']:
            inp['GLOBAL']['PREFERRED_DIAG_LIBRARY'] = 'ELPA'
            inp['GLOBAL']['ELPA_KERNEL'] = 'AUTO'
            inp['GLOBAL']['DBCSR'] = {'USE_MPI_ALLOCATOR': '.FALSE.'}

        return inp

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, dft_params, cell_abc, wfn_file, added_mos, atoms, pdos_lists=None):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'RESTART_FILE_NAME': 'aiida-RESTART.wfn',
                'QS': {
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                },
                'MGRID': {
                    'CUTOFF': '%d' % (dft_params['mgrid_cutoff']),
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '1000',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-7',
                    'ADDED_MOS': str(added_mos),
                    'CHOLESKY': 'INVERSE',
                    'DIAGONALIZATION': {
                        '_': '',
                    },
                    'SMEAR': {
                        'METHOD': 'FERMI_DIRAC',
                        'ELECTRONIC_TEMPERATURE': str(dft_params['smear_temperature']),
                    },
                    'MIXING': {
                        'METHOD': 'BROYDEN_MIXING',
                        'ALPHA': '0.1',
                        'BETA': '1.5',
                        'NBROYDEN': '8',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '15',
                        'EPS_SCF': '1.0E-7',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                },
                'XC': {
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                },
                'PRINT': {
                    'V_HARTREE_CUBE': {
                        'FILENAME': 'HART',
                        'STRIDE': '4 4 4',
                    },
                    'E_DENSITY_CUBE': {
                        'FILENAME': 'RHO',
                        'STRIDE': '2 2 2',
                    },
                },
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'geom.xyz',
                    'COORDINATE': 'xyz'
                },
                'KIND': [],
            }
        }
        
        if wfn_file != "":
            force_eval['DFT']['RESTART_FILE_NAME'] = "./%s"%wfn_file
            
        if pdos_lists != None:
            pdos_list_dicts = [{'COMPONENTS': '', 'LIST': e} for e in pdos_lists]
            force_eval['DFT']['PRINT']['PDOS'] = {
                'NLUMO': '-1',
                'LDOS': pdos_list_dicts
            }

        used_kinds = np.unique(atoms.get_chemical_symbols())
        for symbol in used_kinds:
            force_eval['SUBSYS']['KIND'].append({
                '_': symbol,
                'BASIS_SET': common.ATOMIC_KIND_INFO[symbol]['basis'],
                'POTENTIAL': common.ATOMIC_KIND_INFO[symbol]['pseudo'],
            })
        
        if dft_params['uks']:
            force_eval['DFT']['UKS'] = ''
            force_eval['DFT']['MULTIPLICITY'] = dft_params['multiplicity']
            
            spin_up_indexes = dft_params['spin_up_guess']
            spin_dw_indexes = dft_params['spin_dw_guess']
            
            for i_s, spin_indexes in enumerate([spin_up_indexes, spin_dw_indexes]):
                spin_digit = i_s + 1
                a_nel =  1 if i_s == 0 else -1
                b_nel = -1 if i_s == 0 else  1
                used_kinds = np.unique([atoms.get_chemical_symbols()[i_s] for i_s in spin_indexes])
                for symbol in used_kinds:
                    force_eval['SUBSYS']['KIND'].append({
                        '_': symbol+str(spin_digit),
                        'ELEMENT': symbol,
                        'BASIS_SET': common.ATOMIC_KIND_INFO[symbol]['basis'],
                        'POTENTIAL': common.ATOMIC_KIND_INFO[symbol]['pseudo'],
                        'BS': {
                            'ALPHA': {'NEL': a_nel, 'L': 1, 'N': 2},
                            'BETA':  {'NEL': b_nel, 'L': 1, 'N': 2},
                        },
                    })
                    
        return force_eval
