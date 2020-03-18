#!/bin/bash
INP_LIST_FILE="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/runs_sh_and_list_files/molmap_1_pdb_debug/list_3A_molmap_debug.txt"
INP_PDB_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/"
OUT_MRC_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/runs_sh_and_list_files/molmap_1_pdb_debug/molmap_mrc_3A/"


CHIMERA_CREATE_SCRIPT="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/chimerascripts/create_all_mrc.py"
chimera-1.13 --nogui  $CHIMERA_CREATE_SCRIPT $INP_LIST_FILE $INP_PDB_FOLDER $OUT_MRC_FOLDER
