#!/bin/bash 
 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )" 
 
python $DIR/generateLJFF.py -i geom.xyz 
python $DIR/generateElFF.py -i parent_calc/aiida-HART-v_hartree-1_0.cube
python $DIR/relaxed_scan.py
python $DIR/plot_results.py --df --cbar --save_df
