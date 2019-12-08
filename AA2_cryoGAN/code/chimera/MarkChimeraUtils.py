import chimera
import numpy
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import active_volume
from numpy import zeros, array, dot, linalg

def getRotMatr2Res(res1,res2):
    x1,y1,z1 = getCB_CA_N_frame(res1)
    Rot1 = numpy.matrix([[x1[0], y1[0],z1[0]],
                         [x1[1], y1[1],z1[1]],
                         [x1[2], y1[2],z1[2]]])

    x2,y2,z2 = getCB_CA_N_frame(res2)
    Rot2 = numpy.matrix([[x2[0], y2[0],z2[0]],
                         [x2[1], y2[1],z2[1]],
                         [x2[2], y2[2],z2[2]]])



    Rot = numpy.transpose(Rot2)*Rot1
    return Rot, numpy.transpose(Rot1),numpy.transpose(Rot2)

def getCB_CA_N_frame(res1):

    #first res
    CA_atom = res1.findAtom('CA')
    CB_atom = res1.findAtom('CB')
    N_atom = res1.findAtom('N')

    x_axes = CB_atom.coord() - CA_atom.coord()
    y1_axes = N_atom.coord() - CA_atom.coord()


    z_axes = numpy.cross(x_axes,y1_axes)
    y_axes = -numpy.cross(x_axes,z_axes)
    x_axes = array(x_axes)
    x_axes = x_axes/numpy.linalg.norm(x_axes)
    y_axes = y_axes/numpy.linalg.norm(y_axes)
    z_axes = z_axes/numpy.linalg.norm(z_axes)

    #Rot = [x_axes'; y_axes';z_axes']
    return x_axes,y_axes,z_axes


def getResByNumInChain(prot, chainName, Num):
    for res in prot.residues:
        if res.id.chainId == chainName and res.id.position == Num:
            return res
    return []

def atomsList2spec(atomsList):
    spec = ""
    for atom in atomsList:
        mod = str(atom.molecule.id);
        sub_mod = str(atom.molecule.subid);
        chainId = atom.residue.id.chainId;
        resNum = str(atom.residue.id.position);
        atomType = atom.name
        altLoc = atom.altLoc

        spec = spec + ' ' +'#' +mod +'.' + sub_mod + ':' + resNum + '.' + chainId + '@' + atomType + '.' + altLoc

    return spec

def drange(start, stop, step):
    r = start
    while r < stop:
     	yield r
     	r += step


def get_obj_by_id(req_id):
    for obj in  chimera.openModels.list():
        if obj.id == req_id:
            return obj
    return None
