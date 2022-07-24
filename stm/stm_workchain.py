from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm.nodes.data.array import ArrayData
from aiida.orm import Int, Float, Str, Bool
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.orm import Code

from io import StringIO, BytesIO

from aiida.engine import WorkChain, ToContext, while_

from aiida_cp2k.calculations import Cp2kCalculation

from apps.scanning_probe import common

from aiida.plugins import CalculationFactory
StmCalculation = CalculationFactory('spm.stm')

import os
import tempfile
import shutil
import numpy as np

class STMWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(STMWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, default=lambda:Str(""))
        
        spec.input("dft_params", valid_type=Dict)
        
        spec.input("stm_code", valid_type=Code)
        spec.input("stm_params", valid_type=Dict)
        
        spec.outline(
            cls.run_scf_diag,
            cls.run_stm,
            cls.finalize,
        )
        
        spec.outputs.dynamic = True
    
    def run_scf_diag(self):
        self.report("Running CP2K diagonalization SCF")
        
        emax = float(self.inputs.stm_params.get_dict()['--energy_range'][1])
        self.ctx.n_atoms = len(self.inputs.structure.sites)

        inputs = self.build_cp2k_inputs(self.inputs.structure,
                                        self.inputs.cp2k_code,
                                        self.inputs.dft_params.get_dict(),
                                        self.inputs.wfn_file_path.value,
                                        emax)

        self.report("inputs: "+str(inputs))
        future = self.submit(Cp2kCalculation, **inputs)
        return ToContext(scf_diag=future)
   
           
    def run_stm(self):
        self.report("STM calculation")
             
        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "stm"
        inputs['code'] = self.inputs.stm_code
        inputs['parameters'] = self.inputs.stm_params
        inputs['parent_calc_folder'] = self.ctx.scf_diag.outputs.remote_folder
        
        n_machines = 6
        if self.ctx.n_atoms > 1000:
            n_machines = 12
        if self.ctx.n_atoms > 2000:
            n_machines = 18
        if self.ctx.n_atoms > 3000:
            n_machines = 24
        if self.ctx.n_atoms > 4000:
            n_machines = 30
        
        inputs['metadata']['options'] = {
            "resources": {"num_machines": n_machines},
            "max_wallclock_seconds": 36000,
        } 
        
        # Need to make an explicit instance for the node to be stored to aiida
        settings = Dict(dict={'additional_retrieve_list': ['stm.npz']})
        inputs['settings'] = settings
        
        self.report("Inputs: " + str(inputs))
        
        future = self.submit(StmCalculation, **inputs)
        return ToContext(stm=future)
    
    def finalize(self):
        self.report("Work chain is finished")
    
    
     # ==========================================================================
    @classmethod
    def build_cp2k_inputs(cls, structure, code, dft_params, wfn_file_path, emax):

        inputs = {}
        inputs['code'] = code
        inputs['metadata'] = {}
        inputs['file'] = {}
        
        inputs['metadata']['label'] = "scf_diag"

        
        atoms = structure.get_ase()  # slow
        n_atoms = len(atoms)
        
        spin_guess = None
        if dft_params['uks']:
            spin_guess = [dft_params['spin_up_guess'], dft_params['spin_dw_guess']]

        geom_f = common.make_geom_file(
            atoms, "geom.xyz", spin_guess
        )

        inputs['file']['geom_coords'] = geom_f
        
        cell = dft_params['cell']

        # parameters
        cell_abc = "%f  %f  %f" % (cell[0],
                                   cell[1],
                                   cell[2])
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
            "parser_name": 'cp2k_advanced_parser',
        }
        if wfn_file_path != "":
            inputs['metadata']['options']["prepend_text"] = "cp %s ." % wfn_file_path
        
        return inputs
    
    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, dft_params, cell_abc, walltime, wfn_file, added_mos, atoms):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'ENERGY',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'MEDIUM',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'FORCE_EVAL': cls.get_force_eval_qs_dft(dft_params, cell_abc,
                                                    wfn_file, added_mos, atoms),
        }
        
        if dft_params['elpa_switch']:
            inp['GLOBAL']['PREFERRED_DIAG_LIBRARY'] = 'ELPA'
            inp['GLOBAL']['ELPA_KERNEL'] = 'AUTO'
            inp['GLOBAL']['DBCSR'] = {'USE_MPI_ALLOCATOR': '.FALSE.'}

        return inp

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, dft_params, cell_abc, wfn_file, added_mos, atoms):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
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
                    'SCF_GUESS': 'ATOMIC',
                    'EPS_SCF': '1.0E-7',
                    'ADDED_MOS': str(added_mos),
                    'CHOLESKY': 'INVERSE',
                    'DIAGONALIZATION': {
                        '_': '',
                    },
                    'SMEAR': {
                        'METHOD': 'FERMI_DIRAC',
                        'ELECTRONIC_TEMPERATURE': '300',
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
                        'STRIDE': '2 2 2',
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
                    'COORDINATE': 'xyz',
                    'CENTER_COORDINATES': {'_': ''},
                },
                'KIND': [],
            }
        }
        
        if wfn_file != "":
            force_eval['DFT']['RESTART_FILE_NAME'] = "./%s"%wfn_file
            force_eval['DFT']['SCF']['SCF_GUESS'] = 'RESTART'
        
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
                
                magn = 1.0 if i_s == 0 else -1.0
                
                used_kinds = np.unique([atoms.get_chemical_symbols()[i_s] for i_s in spin_indexes])
                for symbol in used_kinds:
                    force_eval['SUBSYS']['KIND'].append({
                        '_': symbol+str(spin_digit),
                        'ELEMENT': symbol,
                        'BASIS_SET': common.ATOMIC_KIND_INFO[symbol]['basis'],
                        'POTENTIAL': common.ATOMIC_KIND_INFO[symbol]['pseudo'],
                        #'BS': {
                        #    'ALPHA': {'NEL': a_nel, 'L': 1, 'N': 2},
                        #    'BETA':  {'NEL': b_nel, 'L': 1, 'N': 2},
                        #},
                        'MAGNETIZATION': magn,
                    })
                
        return force_eval
