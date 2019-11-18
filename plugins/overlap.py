
from aiida.engine import CalcJob
from aiida.common.utils import classproperty
from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.common import CalcInfo, CodeInfo
from aiida.common import InputValidationError


class OverlapCalculation(CalcJob):

    # --------------------------------------------------------------------------
    def _init_internal_params(self):
        """
        Set parameters of instance
        """
        super(OverlapCalculation, self)._init_internal_params()

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
            "parent_slab_folder": {
               'valid_types': RemoteData,
               'additional_parameter': None,
               'linkname': 'parent_slab_folder',
               'docstring': "remote folder containing slab_eval results",
               },
            "parent_mol_folder": {
               'valid_types': RemoteData,
               'additional_parameter': None,
               'linkname': 'parent_mol_folder',
               'docstring': "remote folder containing mol_eval results ",
               },
            "settings": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'settings',
               'docstring': "Special settings",
               },
            })
        return retdict

    # --------------------------------------------------------------------------
    def prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.
        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """
        
        ### ------------------------------------------------------
        ###  Input check
        
        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("No code specified for this calculation")
            
        try:
            parameters = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError("No parameters specified for this calculation")
        if not isinstance(parameters, ParameterData):
            raise InputValidationError("parameters is not of type ParameterData")
        
        try:
            parent_slab_folder = inputdict.pop(self.get_linkname('parent_slab_folder'))
        except KeyError:
            raise InputValidationError("No parent_slab_folder specified for this calculation")
        if not isinstance(parent_slab_folder, RemoteData):
            raise InputValidationError("parent_slab_folder is not of type RemoteData")
        try:
            parent_mol_folder = inputdict.pop(self.get_linkname('parent_mol_folder'))
        except KeyError:
            raise InputValidationError("No parent_mol_folder specified for this calculation")
        if not isinstance(parent_mol_folder, RemoteData):
            raise InputValidationError("parent_mol_folder is not of type RemoteData")
        
        try:
            settings = inputdict.pop(self.get_linkname('settings'))
        except KeyError:
            raise InputValidationError("No settings specified for this calculation")
        if not isinstance(settings, ParameterData):
            raise InputValidationError("settings is not of type ParameterData")
        settings_dict = settings.get_dict()
        
        # Here, there should be no more parameters...
        if inputdict:
            raise InputValidationError("The following input data nodes are "
                "unrecognized: {}".format(inputdict.keys()))
        
        ###  End of input check
        ### ------------------------------------------------------
        
        # create code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = code.uuid
        
        cmdline = []
        for key in parameters.dict:
            cmdline += [key]
            if parameters.dict[key] != '':
                if isinstance(parameters.dict[key], list):
                    cmdline += parameters.dict[key]
                else:
                    cmdline += [parameters.dict[key]]
        
        codeinfo.cmdline_params = cmdline

        # create calc info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.codes_info = [codeinfo]

        # file lists
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []

        calcinfo.retrieve_list = settings_dict.pop('additional_retrieve_list', [])

        # symlinks
        if parent_slab_folder is not None:
            comp_uuid = parent_slab_folder.get_computer().uuid
            remote_path = parent_slab_folder.get_remote_path()
            symlink = (comp_uuid, remote_path, "parent_slab_folder")
            calcinfo.remote_symlink_list.append(symlink)
        if parent_mol_folder is not None:
            comp_uuid = parent_mol_folder.get_computer().uuid
            remote_path = parent_mol_folder.get_remote_path()
            symlink = (comp_uuid, remote_path, "parent_mol_folder")
            calcinfo.remote_symlink_list.append(symlink)
        
        return calcinfo

# EOF