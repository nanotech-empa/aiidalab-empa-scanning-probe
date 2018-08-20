from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Float, Str, Bool
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

from apps.stm.plugins.evalmorbs import EvalmorbsCalculation
from apps.stm.plugins.stmimage import StmimageCalculation

import tempfile
import shutil
import numpy as np

class STMWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(STMWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("vdw_switch", valid_type=Bool, default=Bool(False))
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
        
        spec.input("eval_orbs_code", valid_type=Code)
        spec.input("eval_orbs_params", valid_type=ParameterData)
        
        spec.input("stm_image_code", valid_type=Code)
        spec.input("stm_image_params", valid_type=ParameterData)
        
        spec.outline(
            cls.run_scf_diag,
            cls.eval_orbs_on_grid,
            cls.make_stm_images
        )
        
        spec.dynamic_output()
    
    def run_scf_diag(self):
        self.report("Running CP2K diagonalization SCF")

        inputs = self.build_cp2k_inputs(self.inputs.structure,
                                        self.inputs.cp2k_code,
                                        self.inputs.mgrid_cutoff,
                                        self.inputs.vdw_switch)

        self.report("inputs: "+str(inputs))
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(scf_diag=Calc(future))
   
    def eval_orbs_on_grid(self):
        self.report("Evaluating Kohn-Sham orbitals on grid")
        
        inputs = {}
        inputs['_label'] = "eval_morbs"
        inputs['code'] = self.inputs.eval_orbs_code
        inputs['parameters'] = self.inputs.eval_orbs_params
        inputs['parent_calc_folder'] = self.ctx.scf_diag.out.remote_folder
        inputs['_options'] = {
            "resources": {"num_machines": 2, "num_mpiprocs_per_machine": 6},
            "max_wallclock_seconds": 7200,
        }
        
        self.report("Inputs: " + str(inputs))
        
        future = submit(EvalmorbsCalculation.process(), **inputs)
        return ToContext(eval_morbs=future)
        
        
           
    def make_stm_images(self):
        self.report("Extrapolating wavefunctions and making STM/STS images")
             
        inputs = {}
        inputs['_label'] = "stm_images"
        inputs['code'] = self.inputs.stm_image_code
        inputs['parameters'] = self.inputs.stm_image_params
        inputs['parent_calc_folder'] = self.ctx.eval_morbs.out.remote_folder
        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 1600,
        }
        
        self.report("Inputs: " + str(inputs))
        
        future = submit(StmimageCalculation.process(), **inputs)
        return ToContext(stm_image=future)
    
    
     # ==========================================================================
    @classmethod
    def build_cp2k_inputs(cls, structure, code,
                          mgrid_cutoff, vdw_switch):

        inputs = {}
        inputs['_label'] = "scf_diag"
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
        
        atoms_z_extent = np.max(atoms.positions[:, 2]) - np.min(atoms.positions[:, 2])
        if atoms.cell[2, 2] < atoms_z_extent+30.0:
            atoms.cell[2, 2] += atoms_z_extent+30.0

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])

        #num_machines = int(np.round(1. + len(atoms)/120.))
        num_machines = 12
        walltime = 3600

        inp = cls.get_cp2k_input(cell_abc,
                                 mgrid_cutoff,
                                 vdw_switch,
                                 walltime*0.97)

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        #settings = ParameterData(dict={'additional_retrieve_list': ['aiida-RESTART.wfn', 'BASIS_MOLOPT', 'aiida.inp']})
        #inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
            "append_text": ur"cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }
        
        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, cell_abc, mgrid_cutoff, vdw_switch, walltime):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'ENERGY',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'LOW'
            },
            'FORCE_EVAL': cls.get_force_eval_qs_dft(cell_abc,
                                                    mgrid_cutoff,
                                                    vdw_switch),
        }

        return inp

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, mgrid_cutoff, vdw_switch):
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
                    'ADDED_MOS': '100',
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
                        'STRIDE': '4 4 4',
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

        if vdw_switch is True:
            force_eval['DFT']['XC']['VDW_POTENTIAL'] = {
                'DISPERSION_FUNCTIONAL': 'PAIR_POTENTIAL',
                'PAIR_POTENTIAL': {
                    'TYPE': 'DFTD3',
                    'CALCULATE_C9_TERM': '.TRUE.',
                    'PARAMETER_FILE_NAME': 'dftd3.dat',
                    'REFERENCE_FUNCTIONAL': 'PBE',
                    'R_CUTOFF': '15',
                }
            }

        force_eval['SUBSYS']['KIND'].append({
            '_': 'Au',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q11'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'C',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q4'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'Br',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q7'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'B',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q3'
        })        
        force_eval['SUBSYS']['KIND'].append({
            '_': 'O',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q6'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'N',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q5'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'H',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q1'
        })

        return force_eval