#!/bin/bash

FTP_SERVER='ftp.ebi.ac.uk'
MAPS_FOLDER='../data/cryoEM/raw_data'


cd $MAPS_FOLDER
wget ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-9572/map/emd_9572.map.gz


