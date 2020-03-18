INP_LIST_FILE="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/list_3A_before_rotation.txt"
OUT_LIST_FILE="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/list_3A_after_rotation.txt"
INP_PDB_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/"
INP_MRC_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/"
OUT_MRC_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931mrc_after_rotation/"
N_ANGLES="10"


#ANALYSE FILE
CHIMERA_ROTATE_SCRIPT="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/chimerascripts/rotate_maps_and_pdbs.py"
chimera-1.13 --nogui  $CHIMERA_ROTATE_SCRIPT $INP_LIST_FILE $OUT_LIST_FILE $INP_PDB_FOLDER $INP_MRC_FOLDER $OUT_MRC_FOLDER $N_ANGLES

CHIMERA_CREATE_SCRIPT="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/chimerascripts/create_db_corners_n_m.py"
