from glob import glob
import os
import sys
import cPickle
import gzip


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)



import network3
reload(network3)
from network3 import Network 
from network3 import ConvPoolLayer3D,ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU



data_folder = dir_path +'/../../data/'

pdb_file_folder = data_folder + '/rotamersdata/DB_res30apix10/'
#mnist_file_1 ="/Users/macbookpro/Documents/rozSVN\/NNTraining/data/mnist.pkl.gz"
#mnist_file_2 ="/a/home/cc/students/csguests/markroza/Work/NNTraining/data/mnist_expanded.pkl.gz"
#load data

### read all pdb files
pdb_files = glob(pdb_file_folder +'*.pkl.gz')
pdbs_for_test = pdb_files[0:50]
pdbs_for_validation = pdb_files[100:300]
pdbs_for_train = pdb_files[100:2000]


#pdbs_for_test = pdb_files[0:1]
#pdbs_for_validation = pdb_files[0:1]
#pdbs_for_train = pdb_files[0:1]

#print "DEBUG" , pdb_file_folder,pdbs_for_test
#print os.listdir(pdb_file_folder)


test_data= network3.load_data_shared(pdbs_for_test)
valid_data= network3.load_data_shared(pdbs_for_validation)
train_data= network3.load_data_shared(pdbs_for_train)

input_data = train_data[0].eval()


mini_batch_size = 10

#Fully Connected
net = Network([FullyConnectedLayer(n_in=11*11*11, n_out=100,activation_fn=ReLU), SoftmaxLayer(n_in = 100,n_out = 21)], mini_batch_size)
res_per_epoch = net.SGD(train_data,10,mini_batch_size,3, valid_data,test_data,lmbda=0.1)

#Convolulations
net_conv  = Network([ ConvPoolLayer3D(image_shape=(mini_batch_size, 1, 11, 11,11), 
filter_shape=(20, 1, 4, 4,4), poolsize=(2, 2,2),activation_fn=ReLU),
FullyConnectedLayer(n_in=20*4*4*4, n_out=100,activation_fn=ReLU), SoftmaxLayer(n_in=100, n_out=21)], mini_batch_size)
res_per_epoch_conv = net_conv.SGD(train_data,10,mini_batch_size,0.003, valid_data,test_data,lmbda=0.01)

#save data for plot
f_name = data_folder + '/temp/res_for_plot.pkl.gz'
valid_data_y = valid_data[1].eval()
ws = "(description, res_per_epoch_conv,res_per_epoch, valid_data_y)"
f = gzip.open(f_name, "w")
cPickle.dump((ws,res_per_epoch_conv,res_per_epoch, valid_data_y), f)
f.close()



#res = net.run_trained_net(input_data)
