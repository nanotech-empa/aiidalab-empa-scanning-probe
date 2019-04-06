import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    #http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">General</th>
        <th style="width:50px" rowspan=2></th>
        <th style="text-align:center">STM</th>
        <th style="width:50px" rowspan=2></th>
        <th style="text-align:center">ORB</th>
        <th style="width:50px" rowspan=2></th>
        <th style="text-align:center">PDOS</th>
        <th style="width:50px" rowspan=2></th>
        <th style="text-align:center">AFM</th>
    </tr>
    
    <tr>
        <td valign="top"><ul>
            <li><a href="{appbase}/setup_codes.ipynb" target="_blank">Setup codes</a>
            <li><a href="{appbase}/manage_calcs.ipynb" target="_blank">Manage calculations</a>
        </ul></td>
        
        <td valign="top"><ul>
            <li><a href="{appbase}/stm/submit_stm.ipynb" target="_blank">Submit STM</a>
            <li><a href="{appbase}/stm/view_stm.ipynb" target="_blank">View STM</a>
        </ul></td>
        
        <td valign="top"><ul>
            <li><a href="{appbase}/orb/submit_orb.ipynb" target="_blank">Submit ORB</a>
            <li><a href="{appbase}/orb/view_orb.ipynb" target="_blank">View ORB</a>
        </ul></td>

        <td valign="top"><ul>
            <li><a href="{appbase}/pdos/submit_pdos.ipynb" target="_blank">Submit PDOS</a>
            <li><a href="{appbase}/pdos/view_pdos.ipynb" target="_blank">View PDOS</a>
        </ul></td>
        
        <td valign="top"><ul>
            <li><a href="{appbase}/afm/submit_afm.ipynb" target="_blank">Submit AFM</a>
            <li><a href="{appbase}/afm/view_afm.ipynb" target="_blank">View AFM</a>
        </ul></td>
    </tr>
    </table>
"""
    
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
    
#EOF
