from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.base import Int, Float, Str, Bool, List
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.remote import RemoteData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

from overlap import OverlapCalculation


import os
import tempfile
import shutil
import numpy as np

class PdosWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(PdosWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("slabsys_structure", valid_type=StructureData)
        spec.input("mol_structure", valid_type=StructureData)
        spec.input("pdos_lists", valid_type=List)
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
        spec.input("wfn_file_path", valid_type=Str, default=Str(""))
        spec.input("elpa_switch", valid_type=Bool, default=Bool(True))
        
        spec.input("overlap_code", valid_type=Code)
        spec.input("overlap_params", valid_type=ParameterData)
        
        spec.outline(
            cls.run_scfs,
            cls.run_overlap
        )
        
        spec.dynamic_output()
    
    def run_scfs(self):
        self.report("Running CP2K diagonalization SCF")

        slab_inputs = self.build_slab_cp2k_inputs(
                        self.inputs.slabsys_structure,
                        self.inputs.pdos_lists,
                        self.inputs.cp2k_code,
                        self.inputs.mgrid_cutoff,
                        self.inputs.wfn_file_path,
                        self.inputs.elpa_switch)
        self.report("slab_inputs: "+str(slab_inputs))
        
        slab_future = submit(Cp2kCalculation.process(), **slab_inputs)
        self.to_context(slab_scf=Calc(slab_future))
        
        mol_inputs = self.build_mol_cp2k_inputs(
                        self.inputs.mol_structure,
                        self.inputs.cp2k_code,
                        self.inputs.mgrid_cutoff,
                        self.inputs.elpa_switch)
        self.report("mol_inputs: "+str(mol_inputs))
        
        mol_future = submit(Cp2kCalculation.process(), **mol_inputs)
        self.to_context(mol_scf=Calc(mol_future))        
           
    def run_overlap(self):
        self.report("Running overlap")
             
        inputs = {}
        inputs['_label'] = "overlap"
        inputs['code'] = self.inputs.overlap_code
        inputs['parameters'] = self.inputs.overlap_params
        inputs['parent_slab_folder'] = self.ctx.slab_scf.out.remote_folder
        inputs['parent_mol_folder'] = self.ctx.mol_scf.out.remote_folder
        inputs['_options'] = {
            "resources": {"num_machines": 4, "num_mpiprocs_per_machine": 12},
            "max_wallclock_seconds": 10600,
        } 
        
        settings = ParameterData(dict={'additional_retrieve_list': ['overlap.npz']})
        inputs['settings'] = settings
        
        self.report("overlap inputs: " + str(inputs))
        
        future = submit(OverlapCalculation.process(), **inputs)
        return ToContext(overlap=future)
    
    
     # ==========================================================================
    @classmethod
    def build_slab_cp2k_inputs(cls, structure, pdos_lists, code,
                          mgrid_cutoff, wfn_file_path, elpa_switch):

        inputs = {}
        inputs['_label'] = "slab_scf"
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

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])
        num_machines = 27
        if len(atoms) > 1500:
            num_machines = 48
        walltime = 72000
        
        wfn_file = ""
        if wfn_file_path != "":
            wfn_file = os.path.basename(wfn_file_path.value)

        inp = cls.get_cp2k_input(cell_abc,
                                 mgrid_cutoff,
                                 walltime*0.97,
                                 wfn_file,
                                 elpa_switch,
                                 pdos_lists)

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.pdos']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
            "append_text": ur"cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }
        if wfn_file_path != "":
            inputs['_options']["prepend_text"] = ur"cp %s ." % wfn_file_path
        
        return inputs
    
    # ==========================================================================
    @classmethod
    def build_mol_cp2k_inputs(cls, structure, code,
                          mgrid_cutoff, elpa_switch):

        inputs = {}
        inputs['_label'] = "mol_scf"
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

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])
        num_machines = 6
        if len(atoms) > 150:
            num_machines = 12
        walltime = 72000

        inp = cls.get_cp2k_input(cell_abc,
                                 mgrid_cutoff,
                                 walltime*0.97,
                                 "",
                                 elpa_switch)

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
    def get_cp2k_input(cls, cell_abc, mgrid_cutoff, walltime, wfn_file, elpa_switch, pdos_lists=None):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'ENERGY',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'LOW',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'FORCE_EVAL': cls.get_force_eval_qs_dft(cell_abc,
                                                    mgrid_cutoff, wfn_file, pdos_lists),
        }
        
        if elpa_switch:
            inp['GLOBAL']['PREFERRED_DIAG_LIBRARY'] = 'ELPA'
            inp['GLOBAL']['ELPA_KERNEL'] = 'AUTO'
            inp['GLOBAL']['DBCSR'] = {'USE_MPI_ALLOCATOR': '.FALSE.'}

        return inp

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, mgrid_cutoff, wfn_file, pdos_lists=None):
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
                    'CUTOFF': '%d' % (mgrid_cutoff),
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '1000',
                    'SCF_GUESS': 'RESTART',
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
                        'STRIDE': '4 4 4',
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

        force_eval['SUBSYS']['KIND'].append({
            '_': 'Au',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q11'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'Ag',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q11'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'Cu',
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
            '_': 'S',
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