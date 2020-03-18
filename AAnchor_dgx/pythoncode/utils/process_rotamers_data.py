
def get_mrc_file_name(pdb_id,res,apix):
	 return pdb_id+'_res'+str(int(res*10))+'apix'+str(int(apix*10))+'.mrc'

def get_pdb_id(file_name_string):
        return file_name_string[0:4]


def _read_rotamers_data_line(line_words):
    resType = line_words[0]

    try:
        OMEGA = float(line_words[5])
    except:
        OMEGA = -999;
    try:
        PHI = float(line_words[6])
    except:
        PHI = -999;
    try:
        PSI = float(line_words[7])
    except:
        PSI = -999;
    try:
        CHI1 = float(line_words[8])
    except:
        CHI1 = -999;
    try:
        CHI2 = float(line_words[9])
    except:
        CHI2 = -999;
    try:
        CHI3 = float(line_words[10])
    except:
        CHI3 = -999;
    try:
        CHI4 = float(line_words[11])
    except:
        CHI4 = -999;

    resdata = {}
    resdata["Type"] =  resType
    resdata["OMEGA"] =  OMEGA
    resdata["PHI"] =  PHI
    resdata["PSI"] =  PSI
    resdata["CHI1"] =  CHI1
    resdata["CHI2"] =  CHI2
    resdata["CHI3"] =  CHI3
    resdata["CHI4"] =  CHI4

    return resdata


def read_rotamers_data_text_file(listFileName):
    if listFileName == None:
        return {}
    #get list of files
    ff = open(listFileName)

    #proccess rotamers text file
    rotamersData ={}
    k=0
    for one_line in ff.readlines():
        if one_line[0] == '#':
            continue

        line_words = one_line.split()

        alternative_location = line_words[4]
        if alternative_location == "A":
            continue

        try:
            resnum = int(line_words[3])
        except:
            continue

        pdbID = line_words[1]
        k=k+1
        if k % 100000 ==0:
            print (k , " Rows Readed for list file")
        dict_of_chains = rotamersData.get(pdbID, {})
        chain_ID = line_words[2]
        dict_of_resnums = dict_of_chains.get(chain_ID, {})

        res_data = _read_rotamers_data_line(line_words)
        #for debug
        res_data["line_num_in_file"] = k-1

        dict_of_resnums[resnum] = res_data
        dict_of_chains[chain_ID] = dict_of_resnums
        rotamersData[pdbID] = dict_of_chains


    return rotamersData
