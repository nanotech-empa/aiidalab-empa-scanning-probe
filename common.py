
from aiida.orm import load_node
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm.calculation.work import WorkCalculation
from aiida.orm.calculation.job import JobCalculation


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