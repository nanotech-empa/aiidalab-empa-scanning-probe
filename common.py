
from aiida.orm import load_node
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm.calculation.work import WorkCalculation
from aiida.orm.calculation.job import JobCalculation

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

workchain_preproc_and_viewer_info = {
    'STMWorkChain': OrderedDict([
        ('stm2', { # If no preprocess matches, error wrt first version is given
            'n_calls': 2,
            'viewer_path': "scanning_probe/stm/view_stm.ipynb",
            'retrieved_files': [(1, ["stm.npz"])], # [(step_index, list_of_retr_files), ...]
            'struct_label': 'structure',
            'req_param': ('stm_params', '--fwhms'), # required parameterdata input and a key in it
        }),
        ('stm1', {
            'n_calls': 2,
            'viewer_path': "scanning_probe/stm/view_stm-v1.ipynb",
            'retrieved_files': [(1, ["stm.npz"])],
            'struct_label': 'structure',
        }),
        ('stm0', {
            'n_calls': 3,
            'viewer_path': "scanning_probe/stm/view_stm-v0.ipynb",
            'retrieved_files': [(2, ["stm_ch.npz"])],
            'struct_label': 'structure',
        })
    ]),
    'PdosWorkChain': OrderedDict([
        ('pdos', {
            'n_calls': 3,
            'viewer_path': "scanning_probe/pdos/view_pdos.ipynb",
            'retrieved_files': [(0, ["aiida-list1-1.pdos"]), (2, ["overlap.npz"])],
            'struct_label': 'slabsys_structure',
        }),
    ]),
    'AfmWorkChain': OrderedDict([
        ('afm', {
            'n_calls': 3,
            'viewer_path': "scanning_probe/afm/view_afm.ipynb",
            'retrieved_files': [(1, ["df.npy"]), (2, ["df.npy"])],
            'struct_label': 'structure',
        }),
    ]),
    'OrbitalWorkChain': OrderedDict([
        ('orb1', {
            'n_calls': 2,
            'viewer_path': "scanning_probe/orb/view_orb.ipynb",
            'retrieved_files': [(1, ["orb.npz"])],
            'struct_label': 'structure',
            'req_param': ('stm_params', '--orb_fwhms'),
        }),
        ('orb0', {
            'n_calls': 2,
            'viewer_path': "scanning_probe/orb/view_orb-v0.ipynb",
            'retrieved_files': [(1, ["orb.npz"])],
            'struct_label': 'structure',
        }),
    ]),
}

PREPROCESS_VERSION = 1.01

def preprocess_one(workcalc):
    """
    Preprocess one SPM calc
    Supports preprocess of multiple versions
    """
    
    workcalc_name = workcalc.get_attrs()['_process_label']
    version_preproc_dict = workchain_preproc_and_viewer_info[workcalc_name]
    
    prefix = None
    reason = None
    
    for prefix_version in version_preproc_dict:
        n_calls = version_preproc_dict[prefix_version]['n_calls']
        retr_list_per_step = version_preproc_dict[prefix_version]['retrieved_files']
        
        # ---
        # check if number of calls matches
        if len(workcalc.get_outputs()) < n_calls:
            if reason is None:
                reason = "Not all calculations started."
            continue
        
        # ---
        # check if all specified files are retrieved
        success = True
        for rlps in retr_list_per_step:
            calc_step, retr_list = rlps
            calc = workcalc.get_outputs()[calc_step]
            retrieved_files = calc.out.retrieved.get_folder_list()
            if not all(f in retrieved_files for f in retr_list):
                if reason is None:
                    reason = "Not all files were retrieved."
                success = False
                break
        if not success:
            continue
            
        # ---
        # check if the required parameter is there
        if 'req_param' in version_preproc_dict[prefix_version]:
            req_param, req_key = version_preproc_dict[prefix_version]['req_param']
            inp_dict = workcalc.get_inputs_dict()
            if not (req_param in inp_dict and req_key in inp_dict[req_param].dict):
                if reason is None:
                    reason = "Required parameter not existing."
                continue
                
        # ---
        # found match!    
        prefix = prefix_version
        break
            
    if prefix is None:
        raise(Exception(reason))
    
    structure = workcalc.get_inputs_dict()[version_preproc_dict[prefix]['struct_label']]
    pk_numbers = [e for e in structure.get_extras() if e.startswith(prefix[:-1])]
    pk_numbers = [int(e.split('_')[1]) for e in pk_numbers if e.split('_')[1].isdigit()]
    pks = [e[1] for e in structure.get_extras().items() if e[0].startswith(prefix[:-1])]
    if workcalc.pk in pks:
        return
    nr = 1
    if len(pk_numbers) != 0:
        for nr in range(1, 100):
            if nr in pk_numbers:
                continue
            break
    structure.set_extra('%s_%d_pk'% (prefix, nr), workcalc.pk)
    

def preprocess_spm_calcs(workchain_list = ['STMWorkChain', 'PdosWorkChain', 'AfmWorkChain']):
    qb = QueryBuilder()
    qb.append(WorkCalculation, filters={
        'attributes._process_label': {'in': workchain_list},
        'or':[
               {'extras': {'!has_key': 'preprocess_version'}},
               {'extras.preprocess_version': {'<': PREPROCESS_VERSION}},
           ],
    })
    qb.order_by({WorkCalculation:{'ctime':'asc'}})
    
    for m in qb.all():
        n = m[0]
        ## ---------------------------------------------------------------
        ## calculation not finished
        if not n.is_sealed:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        calc_states = [out.get_state() for out in n.get_outputs()]
        if 'WITHSCHEDULER' in calc_states:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        ## ---------------------------------------------------------------
            
        if 'obsolete' not in n.get_extras():
            n.set_extra('obsolete', False)
        if n.get_extra('obsolete'):
            continue
        
        wc_name = n.get_attrs()['_process_label']
        
        try:
            if not all([calc.get_state() == 'FINISHED' for calc in n.get_outputs()]):
                raise(Exception("Not all calculations are 'FINISHED'"))
            
            preprocess_one(n)
            print("Preprocessed PK %d (%s)"%(n.pk, wc_name))
            
            n.set_extra('preprocess_successful', True)
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            
            if 'preprocess_error' in n.get_extras():
                n.del_extra('preprocess_error')
            
        except Exception as e:
            n.set_extra('preprocess_successful', False)
            n.set_extra('preprocess_error', str(e))
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            print("Failed to preprocess PK %d (%s): %s"%(n.pk, wc_name, e))

def create_viewer_link_html(structure_extras, apps_path):
    calc_links_str = ""
    for key in sorted(structure_extras.keys()):
        for wc_version_dict in workchain_preproc_and_viewer_info.values():
            for prefix in wc_version_dict.keys():
                if key.split('_')[0] == prefix:
                    nr = key.split('_')[1]
                    pk = structure_extras[key]
                    link_name = ''.join([i for i in prefix if not i.isdigit()])
                    link_name = link_name.upper()
                    calc_links_str += "<a target='_blank' href='%s?pk=%s'>%s %s</a><br />" % (
                        apps_path + wc_version_dict[prefix]['viewer_path'], pk, link_name, nr)
    return calc_links_str
    
    
### ----------------------------------------------------------------
### ----------------------------------------------------------------
### ----------------------------------------------------------------
### Misc

def get_calc_by_label(workcalc, label):
    qb = QueryBuilder()
    qb.append(WorkCalculation, filters={'uuid':workcalc.uuid})
    qb.append(JobCalculation, output_of=WorkCalculation, filters={'label':label})
    assert qb.count() == 1
    calc = qb.first()[0]
    assert(calc.get_state() == 'FINISHED')
    return calc

def get_slab_calc_info(workcalc):
    html = ""
    try:
        cp2k_calc = workcalc.inp.structure.get_inputs()[0]
        opt_workcalc = cp2k_calc.get_inputs_dict()['CALL']
        thumbnail = opt_workcalc.get_extra('thumbnail')
        description = opt_workcalc.description
        struct_description = opt_workcalc.get_extra('structure_description')
        struct_pk = workcalc.inp.structure.pk
        
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