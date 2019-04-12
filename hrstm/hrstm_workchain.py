from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.base import Int, Float, Str, Bool
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.remote import RemoteData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

from apps.scanning_probe import common

from afm import AfmCalculation
from hrstm import HrstmCalculation

import os
import tempfile
import shutil
import numpy as np

class HRSTMWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(HRSTMWorkChain, cls).define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("cell", valid_type=ArrayData)
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
        spec.input("wfn_file_path", valid_type=Str, default=Str(""))
        spec.input("elpa_switch", valid_type=Bool, default=Bool(True))

        spec.input("ppm_code", valid_type=Code)
        spec.input("ppm_params", valid_type=ParameterData)

        spec.input("hrstm_code", valid_type=Code)
        spec.input("hrstm_params", valid_type=ParameterData)

        spec.outline(
            cls.run_scf_diag,
            cls.run_ppm,
            cls.run_hrstm,
        )

        spec.dynamic_output()

    # TODO this is seemingly done everywhere, I don't like copy paste though...
    def run_scf_diag(self):
        self.report("Running CP2K diagonalization SCF")

        inputs = self.build_cp2k_inputs(self.inputs.structure,
                                        self.inputs.cell,
                                        self.inputs.cp2k_code,
                                        self.inputs.mgrid_cutoff,
                                        self.inputs.wfn_file_path,
                                        self.inputs.elpa_switch)

        self.report("inputs: "+str(inputs))
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(scf_diag=Calc(future))

    def run_ppm(self):
        self.report("Running PPM")

        inputs = {}
        inputs['_label'] = "hrstm_ppm"
        inputs['code'] = self.inputs.ppm_code
        inputs['parameters'] = self.inputs.ppm_parameters
        inputs['parent_calc_folder'] = self.ctx.scf_diag.out.remote_folder
        # TODO set atom types properly
        inputs['atomtypes'] = SinglefileData(file="/project/apps/scanning_probe/hrstm/atomtypes_2pp.ini")
        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 7200,
        }

        self.report("PPM inputs: " + str(inputs))

        future = submit(AfmCalculation.process(), **inputs)
        return ToContext(ppm=future)

    def run_hrstm(self):
        self.report("Running HR-STM")

        inputs = {}
        # TODO

        future = submit(HrstmCalculation.process(), **inputs)
        return ToContext(hrstm=future)



































