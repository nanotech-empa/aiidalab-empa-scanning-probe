
#from aiida.orm.calculation.work import WorkCalculation
#from aiida.engine import CalcJob

from aiida.orm import WorkChainNode
from aiida.orm import load_node

from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import Code, Computer
from aiida.engine import CalcJob

import subprocess

from collections import OrderedDict 

### ----------------------------------------------------------------
### ----------------------------------------------------------------
### ----------------------------------------------------------------
### BS & PP

ATOMIC_KIND_INFO = {
    'H' :{'basis' : 'TZV2P-MOLOPT-GTH'  , 'pseudo' : 'GTH-PBE-q1' },
    'Au':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q11'},
    'Ag':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q11'},
    'Cu':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q11'},
    'Al':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q3' },
    'B' :{'basis' : 'TZV2P-MOLOPT-GTH'  , 'pseudo' : 'GTH-PBE-q1' },
    'Br':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q7' },
    'C' :{'basis' : 'TZV2P-MOLOPT-GTH'  , 'pseudo' : 'GTH-PBE-q4' },
    'Si':{'basis' : 'DZVP-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q4' },
    'Ga':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q13'},
    'N' :{'basis' : 'TZV2P-MOLOPT-GTH'  , 'pseudo' : 'GTH-PBE-q5' },
    'O' :{'basis' : 'TZV2P-MOLOPT-GTH'  , 'pseudo' : 'GTH-PBE-q6' },
    'Pd':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q18'},
    'S' :{'basis' : 'TZV2P-MOLOPT-GTH'  , 'pseudo' : 'GTH-PBE-q6' },
    'Zn':{'basis' : 'DZVP-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q12'},
}

### ----------------------------------------------------------------
### ----------------------------------------------------------------
### ----------------------------------------------------------------
### Preprocessing and viewer links

# This code and the way it's processed is to support
# multiple pre/postprocess versions of the same calculation
workchain_preproc_and_viewer_info = {
    'STMWorkChain': {
        # version : {info}
        0: { 
            'n_calls': 2,
            'viewer_path': "scanning_probe/stm/view_stm.ipynb",
            'retrieved_files': [(1, ["stm.npz"])], # [(step_index, list_of_retr_files), ...]
            'struct_label': 'structure',
        },
    },
    'PdosWorkChain': {
        0: {
            'n_calls': 3,
            'viewer_path': "scanning_probe/pdos/view_pdos.ipynb",
            'retrieved_files': [(0, ["aiida-list1-1.pdos"]), (2, ["overlap.npz"])],
            'struct_label': 'slabsys_structure',
        },
    },
    'AfmWorkChain': {
        0: {
            'n_calls': 3,
            'viewer_path': "scanning_probe/afm/view_afm.ipynb",
            'retrieved_files': [(1, ["df.npy"]), (2, ["df.npy"])],
            'struct_label': 'structure',
        },
    },
    'OrbitalWorkChain': {
        0: {
            'n_calls': 2,
            'viewer_path': "scanning_probe/orb/view_orb.ipynb",
            'retrieved_files': [(1, ["orb.npz"])],
            'struct_label': 'structure',
        },
    },
    'HRSTMWorkChain': {
        0: {
            'n_calls': 3,
            'viewer_path': "scanning_probe/hrstm/view_hrstm.ipynb",
            'retrieved_files': [(1, ["df.npy"]), (2, ['hrstm_meta.npy', 'hrstm.npz'])],
            'struct_label': 'structure',
        },
    },
}


PREPROCESS_VERSION = 1.07

def preprocess_one(workcalc):
    """
    Preprocess one SPM calc
    Supports preprocess of multiple versions
    """
    
    workcalc_name = workcalc.attributes['process_label']
    
    if 'version' in workcalc.extras:
        workcalc_version = workcalc.extras['version']
    else:
        workcalc_version = 0
        
    prepoc_info_dict = workchain_preproc_and_viewer_info[workcalc_name][workcalc_version]
    
    # Check if the calculation was successful
    # ---
    # check if number of calls matches
    if len(workcalc.called) < prepoc_info_dict['n_calls']:
        raise(Exception("Not all calculations started."))
    
    # check if the CP2K calculation finished okay
    cp2k_calc = workcalc.called[-1]
    if not cp2k_calc.is_finished_ok:
        raise(Exception("CP2K calculation didn't finish well."))
    
    # ---
    # check if all specified files are retrieved
    success = True
    for rlps in prepoc_info_dict['retrieved_files']:
        calc_step, retr_list = rlps
        calc = list(reversed(workcalc.called))[calc_step]
        retrieved_files = calc.outputs.retrieved.list_object_names()
        if not all(f in retrieved_files for f in retr_list):
            raise(Exception("Not all files were retrieved."))
    # ---
    
    structure = workcalc.inputs[prepoc_info_dict['struct_label']]
    
    # Add the link to the SPM calc to the structure extras in format STMWorkChain_1: <stm_wc_pk> 
    pk_numbers = [e for e in structure.extras if e.startswith(workcalc_name)]
    pk_numbers = [int(e.split('_')[1]) for e in pk_numbers if e.split('_')[1].isdigit()]
    pks = [e[1] for e in structure.extras.items() if e[0].startswith(workcalc_name)]
    if workcalc.pk in pks:
        return
    nr = 1
    if len(pk_numbers) != 0:
        for nr in range(1, 100):
            if nr in pk_numbers:
                continue
            break
    structure.set_extra('%s_%d_pk'% (workcalc_name, nr), workcalc.pk)
    

def preprocess_spm_calcs(workchain_list = ['STMWorkChain', 'PdosWorkChain', 'AfmWorkChain', 'OrbitalWorkChain']):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={
        'attributes.process_label': {'in': workchain_list},
        'or':[
               {'extras': {'!has_key': 'preprocess_version'}},
               {'extras.preprocess_version': {'<': PREPROCESS_VERSION}},
           ],
    })
    qb.order_by({WorkChainNode:{'ctime':'asc'}})
    
    for m in qb.all():
        n = m[0]
        ## ---------------------------------------------------------------
        ## calculation not finished
        if not n.is_sealed:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        calc_states = [out.get_state() for out in n.outputs]
        if 'WITHSCHEDULER' in calc_states:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        ## ---------------------------------------------------------------
            
        if 'obsolete' not in n.extras:
            n.set_extra('obsolete', False)
        if n.get_extra('obsolete'):
            continue
        
        wc_name = n.attributes['process_label']
        
        try:
            if not all([calc.get_state() == 'FINISHED' for calc in n.outputs]):
                raise(Exception("Not all calculations are 'FINISHED'"))
            
            preprocess_one(n)
            print("Preprocessed PK %d (%s)"%(n.pk, wc_name))
            
            n.set_extra('preprocess_successful', True)
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            
            if 'preprocess_error' in n.extras:
                n.delete_extra('preprocess_error')
            
        except Exception as e:
            n.set_extra('preprocess_successful', False)
            n.set_extra('preprocess_error', str(e))
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            print("Failed to preprocess PK %d (%s): %s"%(n.pk, wc_name, e))

def create_viewer_link_html(structure_extras, apps_path):
    calc_links_str = ""
    for key in sorted(structure_extras.keys()):
        key_sp = key.split('_')        
        if len(key_sp) < 2:
            continue    
        wc_name, nr = key.split('_')[:2]
        if wc_name not in workchain_preproc_and_viewer_info:
            continue
            
        link_name = wc_name.replace('WorkChain', '')
        link_name = link_name.replace('Workchain', '')
        spm_pk = int(structure_extras[key])
        
        spm_node = load_node(spm_pk)
        ver = 0
        if 'version' in spm_node.extras:
            ver = spm_node.extras['version']
        
        viewer_path = workchain_preproc_and_viewer_info[wc_name][ver]['viewer_path']
        
        calc_links_str += "<a target='_blank' href='%s?pk=%s'>%s %s</a><br />" % (
            apps_path + viewer_path, spm_pk, link_name, nr)
        
    return calc_links_str
    
    
### ----------------------------------------------------------------
### ----------------------------------------------------------------
### ----------------------------------------------------------------
### Misc

def get_calc_by_label(workcalc, label):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={'uuid':workcalc.uuid})
    qb.append(CalcJob, with_incoming=WorkChainNode, filters={'label':label})
    assert qb.count() == 1
    calc = qb.first()[0]
    assert(calc.is_finished_ok)
    return calc

def get_slab_calc_info(struct_node):
    html = ""
    try:
        cp2k_calc = struct_node.creator
        opt_workchain = cp2k_calc.caller
        thumbnail = opt_workchain.extras['thumbnail']
        description = opt_workchain.description
        struct_description = opt_workchain.extras['structure_description']
        struct_pk = struct_node.pk
        
        html += '<style>#aiida_results td,th {padding: 5px}</style>' 
        html += '<table border=1 id="geom_info" style="margin:0px;">'
        html += '<tr>'
        html += '<th> Structure description: </th>'
        html += '<td> %s </td>' % struct_description
        html += '<td rowspan="2"><img width="100px" src="data:image/png;base64,%s" title="PK:%d"></td>' % (thumbnail, struct_pk)
        html += '</tr>'
        html += '<tr>'
        html += '<th> Calculation description: </th>'
        html += '<td> %s </td>' % description
        html += '</tr>'
        
        html += '</table>'
        
    except:
        html = ""
    return html

def does_remote_file_exist(hostname, path):
    ssh_cmd="ssh "+hostname+" if [ -f "+path+" ]; then echo 1 ; else echo 0 ; fi"
    f_exists = subprocess.check_output(ssh_cmd.split())
    if f_exists.decode()[0] != '1':
        return False
    return True
    
def find_struct_wf(structure_node, computer):
    # check spm
    extras = structure_node.extras
    for ex_k in extras.keys():
        if ex_k.startswith(('stm', 'pdos', 'afm', 'orb', 'hrstm')):
            spm_workchain = load_node(extras[ex_k])
            
            # if calc was done using UKS, don't reuse WFN
            if not spm_workchain.inputs.dft_params['uks']:
                
                cp2k_scf_calc = get_calc_by_label(spm_workchain, 'scf_diag')
                if cp2k_scf_calc.computer.hostname == computer.hostname:
                    wfn_path = cp2k_scf_calc.outputs.remote_folder.get_remote_path() + "/aiida-RESTART.wfn"
                    # check if it exists
                    file_exists = does_remote_file_exist(computer.hostname, wfn_path)
                    if file_exists:
                        print("Found .wfn from %s"%ex_k)
                        return wfn_path
                    
    # check geo opt
    if structure_node.creator is not None:
        geo_opt_calc = structure_node.creator
        
        # if the geo opt was done using UKS, don't reuse WFN
        if 'UKS' not in dict(geo_opt_calc.inputs['parameters'])['FORCE_EVAL']['DFT']:
        
            geo_comp = geo_opt_calc.computer
            if geo_comp is not None and geo_comp.hostname == computer.hostname:
                wfn_path = geo_opt_calc.outputs.remote_folder.get_remote_path() + "/aiida-RESTART.wfn"
                # check if it exists
                file_exists = does_remote_file_exist(computer.hostname, wfn_path)
                if file_exists:
                    print("Found .wfn from geo_opt")
                    return wfn_path
    
    return ""

def comp_plugin_codes(computer_name, plugin_name):
    qb = QueryBuilder()
    qb.append(Computer, project='name', tag='computer')
    qb.append(Code, project='*', with_computer='computer', filters={
        'attributes.input_plugin': plugin_name,
        'or': [{'extras': {'!has_key': 'hidden'}}, {'extras.hidden': False}]
    })
    qb.order_by({Code: {'id': 'desc'}})
    codes = qb.all()
    sel_codes = []
    for code in codes:
        if code[0] == computer_name:
            sel_codes.append(code[1])
    return sel_codes
