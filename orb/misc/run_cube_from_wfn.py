# Path to the cube-kit
DIR="."

# Path to cube_from_wfn.py
# Leave empty if already in $PATH
SCRIPT_PATH=""

mkdir $DIR/cubes

$SCRIPT_PATH cube_from_wfn.py \
  --cp2k_input_file $DIR/aiida.inp \
  --basis_set_file $DIR/BASIS_MOLOPT \
  --xyz_file $DIR/geom.xyz \
  --wfn_file $DIR/aiida-RESTART.wfn \
  --output_dir $DIR/cubes/ \
  --n_homo 5 \
  --n_lumo 5 \
  --dx 0.2 \
#  --gen_rho \