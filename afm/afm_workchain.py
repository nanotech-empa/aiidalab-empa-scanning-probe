from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm.nodes.data.array import ArrayData
from aiida.orm import Int, Float, Str, Bool
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.orm import Code

from aiida.engine import WorkChain, ToContext, while_
from aiida.engine import submit

from aiida_cp2k.calculations import Cp2kCalculation

from apps.scanning_probe import common

from aiida.plugins import CalculationFactory
AfmCalculation = CalculationFactory('spm.afm')

import os
import tempfile
import shutil
import numpy as np

class AfmWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(AfmWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("cell", valid_type=ArrayData)
        spec.input("mgrid_cutoff", valid_type=Int, default=lambda: Int(600))
        spec.input("wfn_file_path", valid_type=Str, default=lambda: Str(""))
        spec.input("elpa_switch", valid_type=Bool, default=lambda: Bool(True))
        
        spec.input("afm_pp_code", valid_type=Code)
        spec.input("afm_pp_params", valid_type=Dict)
        
        spec.input("afm_2pp_code", valid_type=Code)
        spec.input("afm_2pp_params", valid_type=Dict)
        
        spec.outline(
            cls.run_scf_diag,
            cls.run_afms,
            cls.finalize,
        )
        
        spec.outputs.dynamic = True
    
    def run_scf_diag(self):
        self.report("Running CP2K diagonalization SCF")

        inputs = self.build_cp2k_inputs(self.inputs.structure,
                                        self.inputs.cell,
                                        self.inputs.cp2k_code,
                                        self.inputs.mgrid_cutoff,
                                        self.inputs.wfn_file_path.value,
                                        self.inputs.elpa_switch)

        self.report("inputs: "+str(inputs))
        future = self.submit(Cp2kCalculation, **inputs)
        return ToContext(scf_diag=future)

    def run_afms(self):
        self.report("Running PP")
        
        afm_pp_inputs = {}
        afm_pp_inputs['metadata'] = {}
        afm_pp_inputs['metadata']['label'] = "afm_pp"
        afm_pp_inputs['code'] = self.inputs.afm_pp_code
        afm_pp_inputs['parameters'] = self.inputs.afm_pp_params
        afm_pp_inputs['parent_calc_folder'] = self.ctx.scf_diag.outputs.remote_folder
        afm_pp_inputs['atomtypes'] = SinglefileData(file="/home/aiida/apps/scanning_probe/afm/atomtypes_pp.ini")
        afm_pp_inputs['metadata']['options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 7200,
        }
        self.report("Afm pp inputs: " + str(afm_pp_inputs))
        afm_pp_future = self.submit(AfmCalculation, **afm_pp_inputs)
        self.to_context(afm_pp=afm_pp_future)
        
        self.report("Running 2PP")
        
        afm_2pp_inputs = {}
        afm_2pp_inputs['metadata'] = {}
        afm_2pp_inputs['metadata']['label'] = "afm_2pp"
        afm_2pp_inputs['code'] = self.inputs.afm_2pp_code
        afm_2pp_inputs['parameters'] = self.inputs.afm_2pp_params
        afm_2pp_inputs['parent_calc_folder'] = self.ctx.scf_diag.outputs.remote_folder
        afm_2pp_inputs['atomtypes'] = SinglefileData(file="/home/aiida/apps/scanning_probe/afm/atomtypes_2pp.ini")
        afm_2pp_inputs['metadata']['options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 7200,
        }
        self.report("Afm 2pp inputs: " + str(afm_2pp_inputs))
        afm_2pp_future = self.submit(AfmCalculation, **afm_2pp_inputs)
        self.to_context(afm_2pp=afm_2pp_future)
   
    def finalize(self):
        self.report("Work chain is finished")
    
    
    # ==========================================================================
    @classmethod
    def build_cp2k_inputs(cls, structure, cell, code,
                          mgrid_cutoff, wfn_file_path, elpa_switch):

        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "scf_diag"
        inputs['code'] = code
        inputs['file'] = {}

        atoms = structure.get_ase()  # slow

        # structure
        tmpdir = tempfile.mkdtemp()
        geom_fn = tmpdir + '/geom.xyz'
        atoms.write(geom_fn)
        geom_f = SinglefileData(file=geom_fn)
        shutil.rmtree(tmpdir)

        inputs['file']['geom_coords'] = geom_f
        
        cell_array = cell.get_array('cell')

        # parameters
        cell_abc = "%f  %f  %f" % (cell_array[0],
                                   cell_array[1],
                                   cell_array[2])
        num_machines = 12
        if len(atoms) > 500:
            num_machines = 27
        walltime = 72000
        
        wfn_file = ""
        if wfn_file_path != "":
            wfn_file = os.path.basename(wfn_file_path)

        inp = cls.get_cp2k_input(cell_abc,
                                 mgrid_cutoff,
                                 walltime*0.97,
                                 wfn_file,
                                 elpa_switch,
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
        }
        if wfn_file_path != "":
            inputs['metadata']['options']["prepend_text"] = "cp %s ." % wfn_file_path
        
        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, cell_abc, mgrid_cutoff, walltime, wfn_file, elpa_switch, atoms):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'ENERGY',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'LOW',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'FORCE_EVAL': cls.get_force_eval_qs_dft(cell_abc,
                                                    mgrid_cutoff, wfn_file, atoms),
        }
        
        if elpa_switch:
            inp['GLOBAL']['PREFERRED_DIAG_LIBRARY'] = 'ELPA'
            inp['GLOBAL']['ELPA_KERNEL'] = 'AUTO'
            inp['GLOBAL']['DBCSR'] = {'USE_MPI_ALLOCATOR': '.FALSE.'}

        return inp

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, mgrid_cutoff, wfn_file, atoms):
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
                    'CUTOFF': '%d' % (mgrid_cutoff),
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '1000',
                    'SCF_GUESS': 'ATOMIC',
                    'EPS_SCF': '1.0E-7',
                    'ADDED_MOS': '800',
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
            force_eval['DFT']['SCF']['SCF_GUESS'] = 'RESTART'

        used_kinds = np.unique(atoms.get_chemical_symbols())
        for symbol in used_kinds:
            force_eval['SUBSYS']['KIND'].append({
                '_': symbol,
                'BASIS_SET': common.ATOMIC_KIND_INFO[symbol]['basis'],
                'POTENTIAL': common.ATOMIC_KIND_INFO[symbol]['pseudo'],
            })

        return force_eval