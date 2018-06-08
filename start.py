import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    #http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">STM</th>
    </tr>
    <tr>
    <td valign="top"><ul>
    <li><a href="{appbase}/submit_stm.ipynb" target="_blank">Submit STM</a>
    </ul></td>
    </tr>
"""
    
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
    
#EOF
