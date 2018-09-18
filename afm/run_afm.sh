#!/bin/bash 

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

HARTREE="parent_calc_folder/aiida-HART-v_hartree-1_0.cube"

NX=$(sed '4q;d' $HARTREE | awk '{print $1;}')
NY=$(sed '5q;d' $HARTREE | awk '{print $1;}')
NZ=$(sed '6q;d' $HARTREE | awk '{print $1;}')

echo "gridN $NX $NY $NZ" >> params.ini

python $DIR/generateLJFF.py -i parent_calc_folder/geom.xyz
python $DIR/generateElFF.py -i $HARTREE
python $DIR/relaxed_scan.py
python $DIR/plot_results.py --df --cbar --save_df
