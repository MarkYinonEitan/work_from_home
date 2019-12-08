import chimera
from chimera import specifier
import VolumeViewer
from Matrix import euler_xform
from chimera import runCommand
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import conj
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '../../../code/chimera/'
utils_path= '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera/'
sys.path.append(utils_path)

print(__file__)
import EM_utils
reload(EM_utils)
from EM_utils import get_box, calc_box_grid


data_folder = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/'
res_folder = data_folder+'look_for_prove/'

#real_map_file = data_folder + 'emd_0505_1A.mrc'
real_map_file = data_folder + '6n18_gan_mov.mrc'
mol_map_file =  data_folder + '6nt8_molmap_1A.mrc'
gan_map_file =  data_folder + '6n18_gan_mov.mrc'
pdb_file =  data_folder + '6nt8.pdb'
dr=1
central_pos = np.array([114.333, 96.67, 99.73])
step_df =(dr,dr,dr)# step is default to (1,1,1)
small_size = 30
large_size = 100

def get_all_data():
    #load maps and pdb, create molmap simulation
    return real_id, gan_id, mm_id

def fft_corr(large_map, small_map):
	a1_f = conj(fftn(small_map,s=large_map.shape))
	a2_f = fftn(large_map,s=large_map.shape)
	a3_f = a1_f*a2_f
	corr_map = np.abs(ifftn(a3_f))
        print("DEVUG")
        print(small_map[3,3,3])

	corr_map[:] = corr_map[:]/np.std(corr_map[:])*0.5
	corr_map[:] = corr_map[:] - np.mean(corr_map[:])+0.5
        opt_place_temp = np.where(corr_map == np.amax(corr_map))
        print("DEVUG")
        print(corr_map[3,3,3])
        print(opt_place_temp[0][0])

        opt_place = np.array([opt_place_temp[0][0],opt_place_temp[1][0],opt_place_temp[2][0]])
	return opt_place, corr_map



def cut_box(v, center, size,apix=1):

    box_xyz_grid = calc_box_grid(center, size, apix=1)

    mx = get_box(v,box_xyz_grid)

    mx[:] = mx[:]/np.std(mx[:])*0.3
    mx[:] = mx[:] - np.mean(mx[:])+1

    return mx

def get_n_peaks(ary3d, n):

    x_ind=[]
    y_ind=[]
    z_ind=[]
    thrs=np.sort(ary3d.flatten())[::-1]
    k=0
    for t in thrs:
        inds = np.where(ary3d==t)
        x = inds[0][0]
        y = inds[1][0]
        z = inds[2][0]

        mx_val = np.max(ary3d[x-2:x+2,y-2:y+2,z-2:z+2].flatten())
        if t==mx_val:
            x_ind.append(x)
            y_ind.append(y)
            z_ind.append(z)
            k=k+1
        if k == n:
            break

    return x_ind, y_ind, z_ind



#def compare_1_cut_map(real_id,gan_id, mm_id,center_point, dx)




#first step - cut map and swoh
#second step - correlate with the real
#third step - insert interpolation
#4 correlate with the real (first cut, the intep)
#5 correlatw with mol map and gan
#6 put all on one graph


def create_corr_graph(num, gan_map_file,real_map_file,mol_map_file, fold_to_save,large_map_center = None,small_map_cent =None):
    #load
    central_pos = small_map_cent
    runCommand('close all')
    gan_to_cut = VolumeViewer.volume.open_volume_file(gan_map_file)[0]
    real_to_cut = VolumeViewer.volume.open_volume_file(real_map_file)[0]
    mol_to_cut = VolumeViewer.volume.open_volume_file(mol_map_file)[0]
    real_map = VolumeViewer.volume.open_volume_file(real_map_file)[0]

    #cut
    gan_small = cut_box(gan_to_cut, central_pos, small_size)
    print("DEBUG gan")
    print(gan_small[0,0,0])
    real_small = cut_box(real_to_cut, central_pos, small_size)
    mol_small = cut_box(mol_to_cut, central_pos, small_size)
    print("DEBUG mol_small")
    print(mol_small[0,0,0])
    mx_large = cut_box(real_map, large_map_center, large_size)

    #fit and calc optimal position
    opt_place_r, corr_map_r = fft_corr(mx_large, real_small)
    opt_place_m, corr_map_m = fft_corr(mx_large, mol_small)
    opt_place_g, corr_map_g = fft_corr(mx_large, gan_small)

    #plot correlation
    x_r = np.arange(mx_large.shape[0])
    y_r = np.arange(mx_large.shape[1])
    z_r = np.arange(mx_large.shape[2])

    plt.figure(figsize=(27,9 ))
    fg, (ax1,ax2,ax3) = plt.subplots(3)
    fg.suptitle(str(central_pos))

    ax1.plot(x_r, corr_map_r[:,opt_place_r[1],opt_place_r[2]],label = 'Real')
    ax1.plot(x_r, corr_map_m[:,opt_place_r[1],opt_place_r[2]],label = 'pdb2mrc')
    ax1.plot(x_r, corr_map_g[:,opt_place_r[1],opt_place_r[2]],label = 'cryoGan')
    ax1.plot([x_r[opt_place_r[0]]], [corr_map_r[opt_place_r[0],opt_place_r[1],opt_place_r[2]]],'k*',markersize=12,label = 'correct Position')
    ax1.legend()
    ax1.set_title('Shift Along X axis')
#    ax1.xlabel('x shift [Angstrem]')
#    ax1.ylabel('Correlation')

    ax2.plot(y_r, corr_map_r[opt_place_r[0],:,opt_place_r[2]],label = 'Real')
    ax2.plot(y_r, corr_map_m[opt_place_r[0],:,opt_place_r[2]],label = 'pdb2mrc')
    ax2.plot(y_r, corr_map_g[opt_place_r[0],:,opt_place_r[2]],label = 'cryoGan')
    ax2.plot([y_r[opt_place_r[1]]], [corr_map_r[opt_place_r[0],opt_place_r[1],opt_place_r[2]]],'k*',markersize=12,label = 'correct Position')
#    ax2.legend()
    ax2.set_title('Shift Along Y axis')
#    ax2.xlabel('y shift [Angstrem]')
#    ax2.ylabel('Correlation')

    ax3.plot(z_r, corr_map_r[opt_place_r[0],opt_place_r[1],:],label = 'Real')
    ax3.plot(z_r, corr_map_m[opt_place_r[0],opt_place_r[1],:],label = 'pdb2mrc')
    ax3.plot(z_r, corr_map_g[opt_place_r[0],opt_place_r[1],:],label = 'cryoGan')
    ax3.plot([z_r[opt_place_r[2]]], [corr_map_r[opt_place_r[0],opt_place_r[1],opt_place_r[2]]],'k*',markersize=12,label = 'correct Position')
#    ax3.legend()
    ax3.set_title('Shift Along Z axis')
#    ax3.xlabel('z -shift [Angstrem]')
#    ax3.ylabel('Correlation')

    plt.savefig(fold_to_save +'/'+'xyz'+str(num))

    # x_peak_r, y_peak_r, z_peak_r = get_n_peaks(corr_map_r, 1)
    # x_peak_m, y_peak_m, z_peak_m = get_n_peaks(corr_map_m, 10)
    # x_peak_g, y_peak_g, z_peak_g = get_n_peaks(corr_map_g, 10)
    #
    # dist_m = np.sqrt((x_peak_m-x_peak_r[0])**2+(y_peak_m-y_peak_r[0])**2+(z_peak_m-z_peak_r[0])**2)
    # dist_g = np.sqrt((x_peak_g-x_peak_r[0])**2+(y_peak_g-y_peak_r[0])**2+(z_peak_g-z_peak_r[0])**2)
    #
    # plt.figure(figsize=(9,9 ))
    # plt.plot(range(1,len(dist_m)+1), dist_m,'rs',label = 'pdb2mrc')
    # plt.plot(range(1,len(dist_g)+1), dist_g,'ko',label = 'gan')
    # plt.legend()
    # plt.xlim(-1, 12)
    # plt.ylim(-1, 50)
    #
    # plt.savefig(fold_to_save +'/'+'dist'+str(num))
    #
    # num_m = len(np.where(dist_m[:1]<5)[0])
    # num_g = len(np.where(dist_g[:1]<5)[0])

    num_m=1
    num_g=1

    return num_m, num_g




def look_for_prove():

    prot = chimera.openModels.open(pdb_file,type='PDB')[0]

    central_ps =[]
    for rs in prot.residues:
        ca = rs.findAtom('CA')
        if ca != None:
            central_ps.append(ca.coord())
    n_m_all = []
    n_g_all = []

    num_good_g = 0
    num_good_m = 0

    for k in range(len(central_ps)):
        try:
            n_m,n_g = create_corr_graph(k, gan_map_file,real_map_file,mol_map_file, res_folder,central_ps[k])
            n_m_all.append(n_m)
            n_g_all.append(n_g)

            if n_m>0:
                num_good_m=num_good_m+1
            if n_g>0:
                num_good_g=num_good_g+1

        except:
            print(k)

        plt.figure(figsize=(9,3 ))
        plt.plot( n_m_all,'rs',label = 'pdb2mrc')
        plt.plot(n_g_all,'k*',label = 'gan')
        plt.legend()
        plt.savefig(res_folder+'/'+'summary')

        print('MOL GAN')
        print(num_good_m,num_good_g)

    return

#look_for_prove()
prot = chimera.openModels.open(pdb_file,type='PDB')[0]
lm_cent = prot.residues[60].findAtom('CA').coord()
cm_cent = prot.residues[80].findAtom('CA').coord()

create_corr_graph(0, gan_map_file,real_map_file,mol_map_file, res_folder,large_map_center = lm_cent,small_map_cent =cm_cent)


yi, zi = np.mgrid[0:len(y_r),0:len(z_r)]
corr_map_r2 = corr_map_r*corr_map_r*corr_map_r
corr_map_g2 = corr_map_g*corr_map_g*corr_map_g
corr_map_m2 = corr_map_m*corr_map_m*corr_map_m

opt_place_r[0]=10


yr_l,zr_l = get_n_largets_for_plot(corr_map_r2[opt_place_r[0],:,:], 5)
yg_l,zg_l = get_n_largets_for_plot(corr_map_g2[opt_place_r[0],:,:], 5)
ym_l,zm_l = get_n_largets_for_plot(corr_map_m2[opt_place_r[0],:,:], 5)

plt.figure(figsize=(27,9 ))
fg, (ax1,ax2,ax3) = plt.subplots(3)
ax1.pcolormesh(yi, zi, corr_map_r2[opt_place_r[0],:,:].reshape(zi.shape))
ax1.plot(yr_l,zr_l,'k*')
ax1.axis('off')
ax1.axis('equal')
ax1.set_title('Real Map')

ax2.pcolormesh(yi, zi, corr_map_m2[opt_place_r[0],:,:].reshape(zi.shape))
ax2.plot(ym_l,zm_l,'k*')
ax2.set_title('pdb2mrc simulation')
ax2.axis('off')
ax2.axis('equal')

ax3.pcolormesh(yi, zi, corr_map_g2[opt_place_r[0],:,:].reshape(zi.shape))
ax3.plot(yg_l,zg_l,'k*')

ax3.set_title('cryoGAN simulation')
ax3.axis('off')
ax3.axis('equal')

plt.savefig('cor_yz')
