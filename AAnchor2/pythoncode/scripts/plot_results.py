import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



class SoftMaxRes:
    def __init__(self,images_folder,res_per_epoch,input_y,input_x=[],debug_data=[]):
        self.images_folder = images_folder
        self.res_per_epoch = res_per_epoch
        self.input_y = input_y

        #calc useful data
        self.det_per_epoch = numpy.argmax(res_per_epoch,axis=2) # detection per epoch

    def calc_true(self, n):
        true_pos
        false_pos


        return det_rate 
    
    def plot_detection_label(self, n, issave=false):
        #detection data
        det_data = numpy.argmax(res_per_epoch,axis=2)
        ## get results
        detection_rate

        false_positive

        ## plot 

        ## save or show

