
from aiida.orm import load_node
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm.calculation.work import WorkCalculation
from aiida.orm.calculation.job import JobCalculation

path_to_stm_viewer = "scanning_probe/stm/view_stm.ipynb"
path_to_stm0_viewer = "scanning_probe/stm/view_stm-old.ipynb"
path_to_pdos_viewer = "scanning_probe/pdos/view_pdos.ipynb"
path_to_afm_viewer = "scanning_probe/afm/view_afm.ipynb"

path_to_orb_viewer = "scanning_probe/orb/view_orb.ipynb"

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
}

### ----------------------------------------------------------------
### Preprocessing

PREPROCESS_VERSION = 0.89

def preprocess_spm_calcs():
    qb = QueryBuilder()
    qb.append(WorkCalculation, filters={
        'attributes._process_label': {'in': ['STMWorkChain', 'PdosWorkChain', 'AfmWorkChain']},
        'or':[
               {'extras': {'!has_key': 'preprocess_version'}},
               {'extras.preprocess_version': {'<': PREPROCESS_VERSION}},
           ],
    })
    qb.order_by({WorkCalculation:{'ctime':'desc'}})
    
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
            
        try:
            if not all([calc.get_state() == 'FINISHED' for calc in n.get_outputs()]):
                raise(Exception("Not all calculations are 'FINISHED'"))
            
            if n.get_attrs()['_process_label'] == 'STMWorkChain':
                preprocess_one_stm(n)
                print("Preprocessed PK %d (STM)"%n.pk)
            elif n.get_attrs()['_process_label'] == 'PdosWorkChain':
                preprocess_one_pdos(n)
                print("Preprocessed PK %d (PDOS)"%n.pk)
            elif n.get_attrs()['_process_label'] == 'AfmWorkChain':
                preprocess_one_afm(n)
                print("Preprocessed PK %d (AFM)"%n.pk)
                
            n.set_extra('preprocess_successful', True)
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            
            if 'preprocess_error' in n.get_extras():
                n.del_extra('preprocess_error')
            
        except Exception as e:
            n.set_extra('preprocess_successful', False)
            n.set_extra('preprocess_error', str(e))
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            print("Failed to preprocess PK %d: %s"%(n.pk, e))

def preprocess_one_stm(workcalc):
    
    if len(workcalc.get_outputs()) < 2:
        raise(Exception("stm never started."))
        
    stm_calc = workcalc.get_outputs()[-1]
    
    retrieved_files = stm_calc.out.retrieved.get_folder_list()
    if "stm.npz" not in retrieved_files and "stm_ch.npz" not in retrieved_files:
         raise(Exception("stm.npz or stm_ch.npz was not retrieved"))
    
    structure = workcalc.inp.structure
    stm_numbers = [e for e in structure.get_extras() if e.startswith('stm')]
    stm_numbers = [int(e.split('_')[1]) for e in stm_numbers if e.split('_')[1].isdigit()]
    stm_pks = [e[1] for e in structure.get_extras().items() if e[0].startswith('stm')]
    if workcalc.pk in stm_pks:
        return
    stm_nr = 1
    if len(stm_numbers) != 0:
        for stm_nr in range(1, 100):
            if stm_nr in stm_numbers:
                continue
            break
    if "stm_ch.npz" in retrieved_files:
        # Old version
        structure.set_extra('stm0_%d_pk'%stm_nr, workcalc.pk)
    else:
        structure.set_extra('stm_%d_pk'%stm_nr, workcalc.pk)
        
    
def preprocess_one_pdos(workcalc):
    
    if len(workcalc.get_outputs()) < 3:
        raise(Exception("overlap never started."))
    
    slab_scf = workcalc.get_outputs()[0]
    overlap = workcalc.get_outputs()[-1]
    
    if "aiida-list1-1.pdos" not in slab_scf.out.retrieved.get_folder_list():
         raise(Exception("aiida-list1-1.pdos was not retrieved!"))
            
    if "overlap.npz" not in overlap.out.retrieved.get_folder_list():
         raise(Exception("overlap.npz was not retrieved!"))
    
    structure = workcalc.inp.slabsys_structure
    pdos_numbers = [e for e in structure.get_extras() if e.startswith('pdos')]
    pdos_numbers = [int(e.split('_')[1]) for e in pdos_numbers if e.split('_')[1].isdigit()]
    pdos_pks = [e[1] for e in structure.get_extras().items() if e[0].startswith('pdos')]
    if workcalc.pk in pdos_pks:
        return
    nr = 1
    if len(pdos_numbers) != 0:
        for nr in range(1, 100):
            if nr in pdos_numbers:
                continue
            break
    structure.set_extra('pdos_%d_pk'%nr, workcalc.pk)
    
def preprocess_one_afm(workcalc):
    
    if len(workcalc.get_outputs()) < 3:
        raise(Exception("afm never started."))
    
    afm_pp = workcalc.get_outputs()[1]
    afm_2pp = workcalc.get_outputs()[2]
    
    if "df.npy" not in afm_pp.out.retrieved.get_folder_list():
         raise(Exception("df.npy was not retrieved!"))
            
    if "df.npy" not in afm_2pp.out.retrieved.get_folder_list():
         raise(Exception("df.npy was not retrieved!"))
            
    structure = workcalc.inp.structure
    afm_numbers = [e for e in structure.get_extras() if e.startswith('afm')]
    afm_numbers = [int(e.split('_')[1]) for e in afm_numbers if e.split('_')[1].isdigit()]
    afm_pks = [e[1] for e in structure.get_extras().items() if e[0].startswith('afm')]
    if workcalc.pk in afm_pks:
        return
    nr = 1
    if len(afm_numbers) != 0:
        for nr in range(1, 100):
            if nr in afm_numbers:
                continue
            break
    structure.set_extra('afm_%d_pk'%nr, workcalc.pk)
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