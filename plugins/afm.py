
from aiida.orm.calculation.job import JobCalculation
from aiida.common.utils import classproperty
from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.remote import RemoteData
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError


class AfmCalculation(JobCalculation):

    # --------------------------------------------------------------------------
    def _init_internal_params(self):
        """
        Set parameters of instance
        """
        super(AfmCalculation, self)._init_internal_params()

    # --------------------------------------------------------------------------
    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        This will be manually added to the _use_methods in each subclass
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            "parameters": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters',
               'docstring': "The node that specifies the "
                            "input parameters",
               },
            "parent_calc_folder": {
               'valid_types': RemoteData,
               'additional_parameter': None,
               'linkname': 'parent_calc_folder',
               'docstring': "remote folder containing hartree potential",
               },
            "atomtypes": {
               'valid_types': SinglefileData,
               'additional_parameter': None,
               'linkname': 'atomtypes',
               'docstring': "atomtypes.ini file",
               },
            })
        return retdict

    # --------------------------------------------------------------------------
    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.
        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """
        
        code, params, parent_calc_folder, atomtypes_file = self._verify_inlinks(inputdict)
        
        # ---------------------------------------------------
        # Write params.ini file
        params_fn = tempfolder.get_abs_path("params.ini")
        with open(params_fn, 'w') as f:
            for key, val in params.items():
                line = str(key) + " "
                if isinstance(val, list):
                    line += " ".join(str(v) for v in val)
                else:
                    line += str(val)
                f.write(line + '\n')
        # ---------------------------------------------------
        
        # create code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = False

        # create calc info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]

        # file lists
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = [(atomtypes_file.get_file_abs_path(), 'atomtypes.ini')]
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = ["*/*/*.npy"]

        # symlinks
        if parent_calc_folder is not None:
            comp_uuid = parent_calc_folder.get_computer().uuid
            remote_path = parent_calc_folder.get_remote_path()
            symlink = (comp_uuid, remote_path, "parent_calc_folder")
            calcinfo.remote_symlink_list.append(symlink)
        
        return calcinfo

    # --------------------------------------------------------------------------
    def _verify_inlinks(self, inputdict):
            
        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("No code specified for this calculation")
        
        try:
            params_node = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError("No parameters specified for this calculation")
        if not isinstance(params_node, ParameterData):
            raise InputValidationError("parameters is not of type ParameterData")
        params = params_node.get_dict()
        
        try:
            parent_calc_folder = inputdict.pop(self.get_linkname('parent_calc_folder'))
        except KeyError:
            raise InputValidationError("No parent_calc_folder specified for this calculation")
        if not isinstance(parent_calc_folder, RemoteData):
            raise InputValidationError("parent_calc_folder is not of type RemoteData")
        
        try:
            atomtypes_file = inputdict.pop(self.get_linkname('atomtypes'))
        except KeyError:
            raise InputValidationError("No atomtypes specified for this calculation")
        if not isinstance(atomtypes_file, SinglefileData):
            raise InputValidationError("atomtypes is not of type SinglefileData")
           
        # Here, there should be no more parameters...
        if inputdict:
            raise InputValidationError("The following input data nodes are "
                "unrecognized: {}".format(inputdict.keys()))
            
        return (code, params, parent_calc_folder, atomtypes_file)

# EOF