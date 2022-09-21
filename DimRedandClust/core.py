###############################################################################
# Import required packages
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.models import model_from_json
import time
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import KDTree
from scipy.spatial import distance
import pickle
###############################################################################

###############################################################################
def prepare_tensorflow():
    
    # Reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    
    # Check tf version
    print("TensorFlow version: {}".format(tf.__version__))
###############################################################################
    
###############################################################################
def extract_GDVs(gdv_file):
    
    # Read GDV file
    df = []
    for line in open(gdv_file, 'r'):
        df.append(line.rstrip())
    
    # Find GDVs
    df_string = ''.join(df)
    pattern = "\[.+?\]"
    gdvs = re.findall(pattern, df_string)
    
    # Clean up extracted GDVs
    for i in range(0, len(gdvs)):
        gdvs[i] = gdvs[i].replace('[','')
        gdvs[i] = gdvs[i].replace(']','')
    
    gdv_final = []
    
    for i in range(0, len(gdvs)):
        
        # Split and rejoin each GDV to remove spaces 
        gdv = gdvs[i].split(' ')
        gdv = [x for x in gdv if x]
        
        # Convert each GDV to an array of integers
        gdv_int = []
        for j in range(0, len(gdv)):
            gdv_int.append(int(gdv[j]))
        gdv_int_array = np.asarray(gdv_int)
        
        # Append all (integer) GDVs to a list
        gdv_final.append(gdv_int_array)
    
    # Convert list of GDVs into an array
    gdv_final_array = np.asarray(gdv_final)  
    
    return gdv_final_array
###############################################################################

###############################################################################
def find_xyz_path(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith(".xyz"):
            xyz_path = os.path.join(data_dir,file)
            return xyz_path, file.split('.xyz')[0]
        
def find_gdv_path_struct(data_dir):
    for file in os.listdir(data_dir):
        if ".gdv" in file:
            if "all" in file:
                gdv_all_p = os.path.join(data_dir,file)
    return gdv_all_p

def find_gdv_path_comp(data_dir):
    for file in os.listdir(data_dir):
        if ".gdv" in file:
            if "same" in file:
                gdv_same_p = os.path.join(data_dir,file)
            if "diff" in file:
                gdv_diff_p = os.path.join(data_dir,file)
    return gdv_same_p, gdv_diff_p

def find_vapor_path(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith("vapor.npy"):
            vapor_p = os.path.join(data_dir, file)
    return vapor_p
###############################################################################

###############################################################################
def identify_vapor(vapor):
    vapor_indices = []
    for i in range(0, len(vapor)):
        # The "vapor.npy" file provides a numerical value for each
        # particle. If that value is "1", that particle is considered a "vapor"
        # particle. A particle is considered a "vapor" particle if that 
        # particle's closest neighbor is greater than or equal to the distance
        # at which interparticle potential is zero.
        if vapor[i] == 1:
            vapor_indices.append(i)
    return vapor_indices
###############################################################################

###############################################################################
def process_gdv_struct(data_dir, unique_clean = False):
    
    # Get GDV paths
    gdv_all_p = find_gdv_path_struct(data_dir)
    
    # Get GDVs
    gdv_struct = extract_GDVs(gdv_all_p)
    
    # Save GDVs
    np.save(os.path.join(data_dir, "gdv_struct_all.npy"), gdv_struct)
    
    if unique_clean == True:
        
        # Identify vapor indices
        vapor_p = find_vapor_path(data_dir)
        vapor = np.load(vapor_p)
        vapor = np.reshape(vapor, (np.shape(vapor)[0]*np.shape(vapor)[1],1))
        vapor_indices = identify_vapor(vapor)
        
        # Remove vapor particles from GDVs
        gdv_struct_clean = np.delete(gdv_struct, vapor_indices, 0)
    
        # Only retain unique GDVs
        gdv_struct_unique = np.unique(gdv_struct_clean, axis = 0)
    
        return gdv_struct_unique
    
    else:
        return gdv_struct
###############################################################################
        
###############################################################################
def process_gdv_comp(data_dir, unique_clean = False):
    
    # Get GDV paths
    gdv_same_p, gdv_diff_p = find_gdv_path_comp(data_dir)
    
    # Get GDVs
    gdv_same = extract_GDVs(gdv_same_p)
    gdv_diff = extract_GDVs(gdv_diff_p)
    
    # Get compositional GDV
    gdv_comp = np.concatenate((gdv_same,gdv_diff), axis = 1)
    
    # Save GDVs
    np.save(os.path.join(data_dir, "gdv_comp_all.npy"), gdv_comp)
    
    if unique_clean == True:
        
        # Identify vapor indices
        vapor_p = find_vapor_path(data_dir)
        vapor = np.load(vapor_p)
        vapor = np.reshape(vapor, (np.shape(vapor)[0]*np.shape(vapor)[1],1))
        vapor_indices = identify_vapor(vapor)
        
        # Remove vapor particles from GDVs
        gdv_comp_clean = np.delete(gdv_comp, vapor_indices, 0)
    
        # Only retain unique GDVs
        gdv_comp_unique = np.unique(gdv_comp_clean, axis = 0)
    
        return gdv_comp_unique
    
    else:
        return gdv_comp
###############################################################################    

###############################################################################
def process_gdvs_train_struct(traj_dirs, mother_dir):
    
    # Collect unique structural GDVs from each indicated XYZ file
    # indicated XYZ file 
    gdv_struct_list = []
    for traj_dir in traj_dirs:
        gdv_struct_temp = process_gdv_struct(traj_dir)
        gdv_struct_list.append(gdv_struct_temp)
        
    gdv_struct = np.vstack(gdv_struct_list)
    
    # Only retain unique GDVs
    gdv_struct_unique = np.unique(gdv_struct, axis = 0)
    
    # Save unique GDVs
    path_struct = os.path.join(mother_dir, "gdv_struct_unique.npy")
    np.save(path_struct, gdv_struct_unique)
    
    # "gdv_struct_unique" and "gdv_comp_unique" contain the "unique"
    # structrual and compositional GDVs (excluding vapor particle GDVs) for
    # all input data. These GDVs will be used to train the autoencoder
    return gdv_struct_unique
###############################################################################
    
###############################################################################
def process_gdvs_train_comp(traj_dirs, mother_dir):
    
    # Collect unique compositional GDVs from each indicated XYZ file 
    gdv_comp_list = []
    for traj_dir in traj_dirs:
        gdv_comp_temp = process_gdv_comp(traj_dir)
        gdv_comp_list.append(gdv_comp_temp)
        
    gdv_comp = np.vstack(gdv_comp_list)
    
    # Only retain unique GDVs
    gdv_comp_unique = np.unique(gdv_comp, axis = 0)
    
    # Save unique GDVs
    path_comp = os.path.join(mother_dir, "gdv_comp_unique.npy")
    np.save(path_comp, gdv_comp_unique)
    
    # "gdv_struct_unique" and "gdv_comp_unique" contain the "unique"
    # structrual and compositional GDVs (excluding vapor particle GDVs) for
    # all input data. These GDVs will be used to train the autoencoder
    return gdv_comp_unique
###############################################################################   

###############################################################################   
def weigh_gdv(gdv):
    
    # Enter GDV weights 
    o = np.array([1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 4, 4, 4, 4, 3, 4, 6, 
              5, 4, 5, 6, 6, 4, 4, 4, 5, 7, 4, 6, 6, 7, 4, 6, 6,
              6, 5, 6, 7, 7, 5, 7, 6, 7, 6, 5, 5, 6, 8, 7, 6, 6, 
              8, 6, 9, 5, 6, 4, 6, 6, 7, 8, 6, 6, 8, 7, 6, 7, 7, 
              8, 5, 6, 6, 4],dtype=np.float)
            
    oo = np.concatenate((o,o), axis = 0)
   
    # Structrual GDVs have 73 entries while compositional ones have 146
    if np.shape(gdv)[1] == 73:
        w = 1. - o / 73.
    elif np.shape(gdv)[1] == 146:
        w = 1. - oo / 73.
    
    # Weigh GDVs
    weighted_gdv_list = []
    for i in range(0, len(gdv)):
        weighted_gdv_list.append(gdv[i,:]*w)
#        weighted_gdv_list.append(gdv[i,:]*w/np.sum(gdv[i,:]*w))
        
    weighted_gdv = np.vstack(weighted_gdv_list)
    
    return weighted_gdv
###############################################################################   

###############################################################################   
def find_minmax(gdv, weighted = False):
    
    if weighted == False:
        weighted_gdv = weigh_gdv(gdv)
    else:
        weighted_gdv = gdv
    
    # Find min and max values of weighted GDV
    min_list = [] # List of minimum values
    max_list = [] # List of maximum values
    
    for i in range(0, np.shape(weighted_gdv)[1]):
        column_min = np.min(weighted_gdv[:,i])
        column_max = np.max(weighted_gdv[:,i])
        if column_max == 0:
            column_max = 1
        min_list.append(column_min)
        max_list.append(column_max)
    
    min_array = np.asarray(min_list)
    max_array = np.asarray(max_list)
    
    return min_array, max_array
###############################################################################   

###############################################################################
def scale_gdv(weighted_gdv, min_array, max_array):
    
    # Scales weighted gdvs from -1 to 1 
    gdv_norm_list = []
    for i in range(0, np.shape(weighted_gdv)[1]):
        gdv_norm_list.append(2*(weighted_gdv[:,i]-
                              min_array[i])/(max_array[i]-min_array[i])-1)
        
    gdv_norm = np.transpose(np.asarray(gdv_norm_list))
    
    return gdv_norm
###############################################################################    

###############################################################################
def normalize_gdv(gdv, min_array, max_array):
    
    # Weigh GDVs
    weighted_gdv = weigh_gdv(gdv)
    
    # Scale weighted gdv
    gdv_norm = scale_gdv(weighted_gdv, min_array, max_array)
    
    return gdv_norm
###############################################################################   

###############################################################################
def train_prep(gdv_unique, train_percentage, mother_dir, model_type):
    
    # Split data into training/validation/testing data sets
    x_train_raw, x_val_test_raw = train_test_split(gdv_unique, 
                                           test_size=(1-train_percentage/100))
    
    x_val_raw, x_test_raw = train_test_split(x_val_test_raw, test_size=0.50)
    
    # Find minimum and maximum values of TRAINING data only
    # (use for normalization)
    min_array, max_array = find_minmax(x_train_raw, weighted = False)
    
    # Normalize GDVs
    x_train = normalize_gdv(x_train_raw, min_array, max_array)
    x_val = normalize_gdv(x_val_raw, min_array, max_array)
    x_test = normalize_gdv(x_test_raw, min_array, max_array)
    
    # Choose label for saving
    # Structrual GDVs have 73 entries while compositional ones have 146
    if model_type == "struct":
        label = "struct"
    elif model_type =="comp":
        label = "comp"
    
    # Save training/validation/testing data and min/max arrays
    x_train_path = os.path.join(mother_dir, "x_train_" + label + ".npy")
    x_val_path = os.path.join(mother_dir, "x_val_" + label + ".npy")
    x_test_path = os.path.join(mother_dir, "x_test_" + label + ".npy")
    min_array_path = os.path.join(mother_dir, "min_array_" + label + ".npy")
    max_array_path = os.path.join(mother_dir, "max_array_" + label + ".npy")
    
    np.save(x_train_path, x_train)
    np.save(x_val_path, x_val)
    np.save(x_test_path, x_test)
    np.save(min_array_path, min_array)
    np.save(max_array_path, max_array)
    
    return x_train, x_val, x_test, min_array, max_array
###############################################################################

###############################################################################
def create_autoencoder(input_dim, n_hidden_nodes, n_bottleneck_nodes, 
                       n_hidden_layers, drop_prob):
    
    # First encoder layer
    encoder_initial = Dense(n_hidden_nodes, input_shape=(input_dim,), 
                            kernel_initializer="glorot_uniform", 
                            activation = "tanh")
    
    # Encoder hidden layers
    encoder_list = []
    for i in range(0,n_hidden_layers-1):
        encoder_list.append(Dense(n_hidden_nodes, activation="tanh", 
                                  kernel_initializer="glorot_uniform"))
    
    # Final layer
    encoder_final = Dense(n_bottleneck_nodes, activation = "linear")
    
    # Create decoder
    # First decoder layer and hidden layers
    decoder_list = []
    for i in range(0,n_hidden_layers):
        decoder_list.append(Dense(n_hidden_nodes, activation="tanh", 
                                  kernel_initializer="glorot_uniform"))
        
    # Final decoder layer
    decoder_final = Dense(input_dim, activation = "linear")
    
    # Combine encoder and decoder to create autoencoder
    autoencoder = Sequential()
    autoencoder.add(encoder_initial)
    autoencoder.add(Dropout(drop_prob))
    for i in range(0, n_hidden_layers-1):
        autoencoder.add(encoder_list[i])
        autoencoder.add(Dropout(drop_prob))
    autoencoder.add(encoder_final)
    for i in range(0, n_hidden_layers):
        autoencoder.add(decoder_list[i])
        autoencoder.add(Dropout(drop_prob))
    autoencoder.add(decoder_final)
    
    return autoencoder       
###############################################################################        

###############################################################################
def train_autoencoders(x_train, x_val, x_test, hidden_nodes, 
                      bottleneck_nodes, hidden_layers, patience, mother_dir):
    
    # Find input dimension
    input_dim = np.shape(x_train)[1]
    
    # Choose label for saving
    # Structrual GDVs have 73 entries while compositional ones have 146
    if input_dim == 73:
        label = "Autoencoder_Struct/"
    elif input_dim == 146:
        label = "Autoencoder_Comp/"
    
    # Standard autoencoder architecture choices
    drop_prob = 0.20 # dropout probability
    learning_rate = 0.001 # learning rate
    n_epoch = 10**4 # number of epochs
    n_batch = 64 # batch size
    es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', verbose = 1, 
                   patience = patience) # patience for early stopping
    
    # Begin loop for neural network architecture choices
    for n_hn in hidden_nodes:
        for n_bn in bottleneck_nodes:
            for n_hl in hidden_layers:
    
                # Create autoencoder
                autoencoder = create_autoencoder(input_dim, n_hn, n_bn, n_hl, 
                                                 drop_prob)
                
                # Choose optimizer
                optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
                
                # Compile autoencoder (and choose loss)
                autoencoder.compile(optimizer=optimizer, 
                                    loss='mean_squared_error')
                
                # Record start time (for autoencoder training)
                start = time.time()
                
                # Train autoencoder
                autoencoder.fit(x_train, x_train,
                            epochs=n_epoch,
                            batch_size=n_batch,
                            shuffle=True, callbacks = [es], 
                            validation_data=(x_val, 
                                             x_val))
                
                # Record end time and total time (for autoencoder training)
                end = time.time()
                total_time = end-start
                
                # Calculate training, validaiton, and testing loss
                train_loss = np.asarray(autoencoder.history.history['loss'])
                val_loss = np.asarray(autoencoder.history.history['val_loss'])
                test_loss = np.mean((autoencoder.predict(x_test)-x_test)**2)
                
                # Extract encoder
                encoder = Sequential()
                encoder.add(autoencoder.layers[0])
                for i in range(1, n_hl+1):
                    encoder.add(autoencoder.layers[2*i])
                
                # Create folder to save models, losses, and training time
                child_dir = (label + str(n_hl) + "_HL_" + str(n_hn) +
                             "_Nodes/" + str(n_bn) + "_OP")
                total_dir = os.path.join(mother_dir, child_dir)
                os.makedirs(total_dir)
                
                # Save autoencoder
                model_json_1 = autoencoder.to_json()
                with open(os.path.join(total_dir,
                                       "Autoencoder.json"), "w") as json_file:
                    json_file.write(model_json_1)
                autoencoder.save_weights(os.path.join(total_dir,
                                                      "Autoencoder.h5"))
                
                # Save encoder
                model_json_2 = encoder.to_json()
                with open(os.path.join(total_dir,
                                       "Encoder.json"), "w") as json_file:
                    json_file.write(model_json_2)
                encoder.save_weights(os.path.join(total_dir,"Encoder.h5"))
                
                # Save loss
                np.save(os.path.join(total_dir,"Val_Loss.npy"), val_loss)
                np.save(os.path.join(total_dir,"Train_Loss.npy"), train_loss)
                np.save(os.path.join(total_dir,"Test_Loss.npy"), test_loss)
                
                # Save total training time
                np.save(os.path.join(total_dir,"total_time.npy"), total_time)
###############################################################################

###############################################################################
def create_elbow(mother_dir, model_type):
    
    # Choose labels for saving/plotting based on input "model type"
    if model_type == "struct":
        model_dir = os.path.join(mother_dir, "Autoencoder_Struct")
        title = "Elbow Plot: Structural Autoencoder"
    elif model_type == "comp":
        model_dir = os.path.join(mother_dir, "Autoencoder_Comp")
        title = "Elbow Plot: Compositional Autoencoder"
    
    # Get labels for lines in elbow plot
    plot_labels = os.listdir((model_dir))
    
    # Collect losses
    y_axis = [[] for x in range(len(plot_labels))]
    for i in range(0, len(plot_labels)):
        bn_labels = np.sort(os.listdir(os.path.join(model_dir, 
                                                    plot_labels[i])))
        for j in range(0, len(bn_labels)):
            loss_path = os.path.join(model_dir, plot_labels[i], bn_labels[j],
                                     "Test_Loss.npy")
            y_axis[i].append(np.load(loss_path))
            
    # Get x-axis values for elbow plot
    x_axis = []
    for i in range(0, len(bn_labels)):
        x_axis.append(int(bn_labels[i].split('_')[0]))
    
    # Create and save elbow plot
    for i in range(0, len(plot_labels)):
        line_style_list = ['solid', 'dashed', 'dotted', 'dashdot']
        plt.plot(x_axis, y_axis[i], 
                 linewidth = 2, 
                 linestyle = line_style_list[int(i % len(line_style_list))],
                 label = plot_labels[i])
    plt.legend(fontsize = 10)
    plt.xlabel('Bottleneck Size', fontsize = 14)
    plt.ylabel('Mean Squared Error', fontsize = 14)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(mother_dir, "Elbow_" + model_type + ".png"), facecolor="w")
    plt.show()
    plt.clf()
###############################################################################

###############################################################################
def get_encoder(n_hidden_nodes, n_bottleneck_nodes, n_hidden_layers, 
                mother_dir, model_type):
    
    # Choose label based on "model type" input
    if model_type == "struct":
        label = label = "Autoencoder_Struct/"
    elif model_type == "comp":
        label = "Autoencoder_Comp/"
    
    # Get encoder paths
    encoder_dir = (label + str(n_hidden_layers) + "_HL_" + str(n_hidden_nodes) +
                 "_Nodes/" + str(n_bottleneck_nodes) + "_OP")
    encoder_dir = os.path.join(mother_dir, encoder_dir)
    
    encoder_path_json = os.path.join(encoder_dir, "Encoder.json")
    encoder_path_h5 = os.path.join(encoder_dir, "Encoder.h5")
    
    # Load encoder
    json_file = open(encoder_path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    encoder = model_from_json(loaded_model_json)
    encoder.load_weights(encoder_path_h5)
    
    return encoder, encoder_dir
###############################################################################

###############################################################################
def reduce_dim(encoder, gdv, min_array, max_array):
    
    # Normalize GDVs
    gdv_norm = normalize_gdv(gdv, min_array, max_array)
    
    # Reduce dimensionalty of GDVs
    lowd = encoder.predict(gdv_norm)
    
    return lowd
###############################################################################

###############################################################################
def reduce_dim_uniquegdvs(encoder, gdv_unique, min_array, max_array, 
                          encoder_dir):
    
    # Reduce dimensionality
    lowd_unique = reduce_dim(encoder, gdv_unique, min_array, max_array)
        
    # Save low-dimensional data in the same path as the encoder
    np.save(os.path.join(encoder_dir, "LowD_unique.npy"), lowd_unique)
        
    return lowd_unique
###############################################################################

###############################################################################
def reduce_dim_allgdvs(encoder, min_array, max_array, traj_dir, encoder_dir):
    
    # Get npy array of all GDV signatures
    # Structrual GDVs have 73 entries while compositional ones have 146
    if len(min_array) == 73:
        filename = "gdv_struct_all.npy"
    elif len(min_array) == 146:
        filename = "gdv_comp_all.npy"
    gdv_all = np.load(os.path.join(traj_dir,filename))
    
    # Reduce dimensionality of all GDV signatures
    lowd_all = reduce_dim(encoder, gdv_all, min_array, max_array)
    
    # Create directory where gdv_all will be saved
    gdv_path = os.path.join(encoder_dir, traj_dir)
    if os.path.isdir(gdv_path) == False:
        os.makedirs(gdv_path)
        
    # Save gdv file in directory
    np.save(os.path.join(gdv_path, "lowd_all.npy"), lowd_all)
    
    return lowd_all
###############################################################################
    
###############################################################################
def calc_linkage(lowd_unique):
    return linkage(lowd_unique, method='ward')
###############################################################################
    
###############################################################################
def identify_target(encoder, target_list, min_array, max_array, encoder_dir,
                    model_type):
    
    lowd_target = {}
    
    if model_type == "struct":
        
        # Target structural GDVs
        FCC = np.asarray([[72, 180, 90, 96, 144, 144, 132, 44, 24, 168, 336, 
                            168, 96, 96, 32, 240, 240, 120, 0, 0, 0, 0, 36, 9,
                            144, 72, 144, 48, 48, 96, 48, 192, 192, 96, 0, 0, 
                            0, 0, 0, 168, 336, 168, 168, 336, 84, 0, 0, 0, 0, 
                            0, 0, 48, 24, 48, 0, 0, 24, 72, 24, 144, 144, 72, 
                            0, 0, 0, 48, 96, 96, 24, 6, 0, 0, 0]])
        
        HCP = np.asarray([[72, 180, 90, 96, 156, 156, 150, 50, 24, 156, 312, 
                            156, 102, 102, 32, 252, 252, 126, 18, 36, 18, 18, 
                            36, 9, 72, 36, 72, 48, 48, 96, 48, 198, 198, 99, 0,
                            12, 12, 24, 12, 162, 324, 162, 162, 312, 78, 12, 
                            12, 12, 24,  0,  0, 36, 18, 36, 18, 12, 30, 90, 30,
                            156, 156, 78,  0,  0,  0, 36, 72, 72, 24, 6, 6, 9, 
                            0]])
        
        BCC = np.asarray([[100, 278, 139, 180, 312, 312, 240, 80, 24, 288, 
                            576, 288, 240, 240, 96, 600, 600, 300, 48, 96, 48,
                            48, 100,  25, 144, 72, 144, 144, 144, 288, 144, 
                            312, 312, 156, 0, 24, 24, 48, 24, 324, 648, 324, 
                            324, 648, 162, 96, 96, 96, 192, 0, 0, 48, 24, 48, 
                            48, 32, 96, 288,  96, 408, 408, 204, 0, 0, 0,  96, 
                            192, 192,  48,  12,  72, 108, 0]])
        
        IrVA = np.asarray([[78,212,106,102,180,180,234,78,28,194,388,194,112,112,32,308,
                  308,154,28,56,28,28,100,25,96,48,96,52,52,104,52,372,372,186,0,
                  16,16,32,16,228,456,228,228,424,106,8,8,8,16,0,0,40,20,40,24,16,
                  36,108,36,180,180,90,0,0,0,40,80,80,28,7,4,6,0]])   
        
        IrVB = np.asarray([[66,150,75,90,132,132,87,29,20,122,244,122,92,92,32,188,188,94,
                   12,24,12,12,8,2,60,30,60,40,40,80,40,88,88,44,10,8,8,16,8,104,
                   208,104,104,216,54,12,12,12,24,0,0,32,16,32,12,8,24,72,24,136,
                   136,68,0,0,0,32,64,64,20,5,8,12,0]])
            
        DCsClA = np.asarray([[90,232,116,156,248,248,168,56,24,224,448,224,196,196,80,436,436,
                   218,32,64,32,32,44,11,88,44,88,104,104,208,104,192,192,96,0,24,24,
                   48,24,224,448,224,224,472,118,72,72,72,144,0,0,40,20,40,36,24,72,
                   216,72,312,312,156,0,0,0,72,144,144,44,11,56,84,0]])
        
        DCsClB = np.asarray([[96,266,133,162,280,280,252,84,20,276,552,276,204,204,80,524,524,
                   262,40,80,40,40,108,27,156,78,156,124,124,248,124,384,384,192,10,12,
                   12,24,12,320,640,320,320,640,160,64,64,64,128,0,0,40,20,40,36,24,88,
                   264,88,348,348,174,0,0,0,80,160,160,32,8,52,78,0]])
            
        # Get low-dimensional representations of target GDVs
        for i in range(0, len(target_list)):
            if target_list[i] == "FCC":
                lowd_target["FCC"] = reduce_dim(encoder, FCC, 
                           min_array, max_array)
            elif target_list[i] == "HCP":
                lowd_target["HCP"] = reduce_dim(encoder, HCP, 
                           min_array, max_array)
            elif target_list[i] == "BCC":
                lowd_target["BCC"] = reduce_dim(encoder, BCC, 
                           min_array, max_array)
            elif target_list[i] == "IrVA":
                lowd_target["IrVA"] = reduce_dim(encoder, IrVA, 
                           min_array, max_array)
            elif target_list[i] == "IrVB":
                lowd_target["IrVB"] = reduce_dim(encoder, IrVB, 
                           min_array, max_array)
            
            elif target_list[i] == "DCsClA":
                lowd_target["DCsClA"] = reduce_dim(encoder, DCsClA, 
                           min_array, max_array)
            
            elif target_list[i] == "DCsClB":
                lowd_target["DCsClB"] = reduce_dim(encoder, DCsClB, 
                           min_array, max_array)
                
            else:
                print ("There is no GDV data for the entry " + 
                       str(target_list[i]))
                      
    
    elif model_type == "comp":
        
        # Target compositional GDVs
        FCC_HCP_IrVB = np.asarray([[8, 12, 6, 0, 0, 0, 12, 4, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 56, 28, 0, 0,
                                0, 168, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 280, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0]])
    
        BCC_DCsClB = np.asarray([[12, 30, 15, 0, 0, 0, 60, 20, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 60, 15, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 16, 56, 28, 0, 0, 0, 168, 
                            56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 280, 
                            70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        IrVA_DCsClA = np.asarray([[10,20,10, 0, 0, 0,30,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0,20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,16,56,28,0,0,0,168,56,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,280,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

        
        # Get low-dimensional representations of target GDVs
        for i in range(0, len(target_list)):
            if target_list[i] == "FCC_HCP_IrVB":
                lowd_target["FCC_HCP_IrVB"] = reduce_dim(encoder, FCC_HCP_IrVB, 
                           min_array, max_array)
            
            elif target_list[i] == "BCC_DCsClB":
                lowd_target["BCC_DCsClB"] = reduce_dim(encoder, BCC_DCsClB, 
                           min_array, max_array)
            
            elif target_list[i] == "IrVA_DCsClA":
                lowd_target["IrVA_DCsClA"] = reduce_dim(encoder, IrVA_DCsClA, 
                           min_array, max_array)
                
            else:
                print ("There is no GDV data for the entry " + 
                       str(target_list[i]))
        
    # Save lowd target
    f1 = open(os.path.join(encoder_dir, "Lowd_Target.pkl"),"wb")
    pickle.dump(lowd_target,f1)
    f1.close()
    
    return lowd_target
###############################################################################                
        
###############################################################################
def choose_cluster_num(encoder, target_list, min_array, max_array, lowd_unique,
                       Z, encoder_dir, model_type):
    
    # Choose plot title based on input "model type"
    if model_type == "struct":
        title = "Structural Characterization: Cluster Number Choice"
        fig_index = 1
    elif model_type == "comp":
        title = "Compositional Characterization: Cluster Number Choice"
        fig_index = 2
    
    # Find low-dimensional representations of target lattice GDVs
    lowd_target = identify_target(encoder, target_list, min_array, 
                                  max_array, encoder_dir, model_type)
    
    # Create dictionary of target indices
    target_indices = {}
    
    # Chooose maximum number of clusters
    max_clust = 500  
    
    # Get kdtree for rapidly looking up nearest neighbors
    kdtree = KDTree(lowd_unique)
        
    # Find indices within "lowd" (i.e., the low dimensional representations of 
    # the unique GDVs) that correspond to target lattices
    # If the exact perfect lattice is not included in the provided data,
    # the closest point to those target lattices will be chosen
    for key in lowd_target:
        sample = lowd_target[key]
        dist, point = kdtree.query(sample, 1)
        target_indices[key] = point[0]
        
    # Save indices    
    for key in target_indices:
        np.save(os.path.join(encoder_dir, key + "_index.npy"), 
                    target_indices[key])
    
    # Initialize dictionary that counts populations of target lattice clusters
    clust_count = {}
    for key in target_indices:
        clust_count[key] = []
    
    # Count populations of target lattice clusters with total changing number
    # of clusters
    for i in range(1, max_clust+1):
        clust_ids = fcluster(Z, i, 'maxclust')
        
        for key in target_indices:
            clust_id = int(clust_ids[target_indices[key]])
            clust_count[key].append(np.count_nonzero(clust_ids == clust_id))
         
    
    # Plot cluster counts as a function of total number of clusters
    plt.figure(fig_index)
    for key in clust_count:
        if key == "FCC" or key == "FCC_HCP_IrVB":
            color = 'r'
        elif key == "HCP":
            color = 'g'
        elif key == "BCC" or key == "BCC_DCsClB":
            color = 'b'
        elif key == "IrVA" or key == "IrVA_DCsClA":
            color = 'm'
        elif key == "IrVB":
            color = 'y'
        elif key == "DCsClA":
            color = 'c'
        elif key == "DCsClB":
            color = 'brown'
            
        plt.plot(np.asarray(range(1,max_clust+1)), clust_count[key], color, 
                 linewidth=2, label= key + " NG Count")

    plt.xlim([1,max_clust])
    plt.ylim([1,500])
    plt.ylabel('Unique Neighborhood Graphs', fontsize=14)
    plt.xlabel('Total Number of Clusters', fontsize=14)
#    plt.axvline(x=180, linewidth = 2, linestyle = '--', color='k', 
#                label = "180 clusters")
    plt.legend(fontsize=10,  bbox_to_anchor=(1.05, 1))
    plt.title(title, fontsize=14)
    plt.tight_layout()
  
    # Save figure
    plt.savefig(os.path.join(encoder_dir, 'Choose_Clusters.png'), facecolor="w")
    
    return lowd_target, target_indices, clust_count
###############################################################################
    
###############################################################################
def create_cluster_tree(Z, dend_labels, nclust, encoder_dir, model_type):
    
    # Initialize cluster tree (i.e., "dendrogram")
    dn_noplot = dendrogram(
                    Z,
                    truncate_mode='lastp',
                    p=nclust,
                    no_plot=True,
                    )
    temp = {dn_noplot["leaves"][ii]: dend_labels[ii] for ii in range(len(dn_noplot["leaves"]))} 
    
    # Define leaf label function for dendrogram
    def llf(xx):
        return "{}".format(temp[xx])
    
    # Plot cluster tree
    plt.figure(figsize = (80,45))
    plt.xlabel('Cluster Labels', fontsize=20)
    plt.ylabel('Ward\'s Distance', fontsize=20)
    
    dn = dendrogram(
                Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=nclust,  # show only the last p merged clusters
                leaf_label_func=llf,
                leaf_rotation=90.,
                leaf_font_size=20.,
                show_contracted=False
                )
    
    # Choose title based on input "model type"
    if model_type == "struct":
        plt.title("Structural Characterization Cluster Tree", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(encoder_dir,"Dendrogram_Structural_" + 
                             str(nclust) +"_Clusters.png"), facecolor="w")
    
    elif model_type == "comp":
        plt.title("Compositional Characterization Cluster Tree", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(encoder_dir,"Dendrogram_Compositional_" + 
                             str(nclust) +"_Clusters.png"), facecolor="w")
    
###############################################################################

###############################################################################
def cluster(lowd_unique, Z, target_indices, nclust, encoder_dir, model_type):
    
    # Cluster and save cluster ids
    clust_ids = fcluster(Z, nclust, 'maxclust')
    np.save(os.path.join(encoder_dir, 'Cluster_IDs_' + str(nclust) + 
                         '_Clusters.npy'), clust_ids)
            
    # Get dendrogram labels
    dend_labels = []
    for i in range(0, nclust):
        target = 0
        for key in target_indices:
            if i + 1 == int(clust_ids[target_indices[key]]):
                dend_labels.append("C" + str(i+1) + " -- " + key)
                
                target = 1
                break
        if target == 0:
            dend_labels.append('') 
            
    # Create cluster tree
    create_cluster_tree(Z, dend_labels, nclust, encoder_dir, model_type)
    
    return clust_ids
###############################################################################

###############################################################################
def assign_clusters(lowd_unique, clust_ids, lowd_all, traj_dir, encoder_dir):
    
    # Find vapor ids
    vapor = np.load(find_vapor_path(traj_dir))
    vapor = np.reshape(vapor, 
                      (np.shape(vapor)[0]*np.shape(vapor)[1],1))
    
    # Assign a "cluster id" to vapor particles
    vapor_label = np.max(clust_ids + 1)
    
    # Get kdtree for rapidly looking up nearest neighbors
    kdtree = KDTree(lowd_unique)
    
    # Match cluster ids to those of all particles
    clust_ids_all = []
    for i in range(0, len(lowd_all)):
        if vapor[i] == 1:
            clust_ids_all.append(vapor_label)
        else:
            sample = lowd_all[i,:]
            dist, point = kdtree.query(sample, 1)
            clust_ids_all.append(clust_ids[point])
        
    # Turn list of cluster ids into an array
    clust_ids_all = np.asarray(clust_ids_all)
    
    # Create directory where cluster ids will be saved
    clust_path = os.path.join(encoder_dir, traj_dir)
    if os.path.isdir(clust_path) == False:
        os.makedirs(clust_path) 
    
    # Save cluster labels
    np.save(os.path.join(clust_path, "Cluster_IDs_" +
                         str(np.max(clust_ids)) +
                         "_Clusters.npy"), clust_ids_all)
    
    
    return clust_ids_all
###############################################################################         

###############################################################################  
def get_target_clusters(clust_ids, target_indices):
    
    # Get cluster IDs of target lattices
    target_clusters = {}
    for key in target_indices:
        target_clusters[key] = int(clust_ids[int(target_indices[key])])
    return target_clusters
###############################################################################

###############################################################################
def get_combined_cluster_ids(target_clust_st, target_clust_co):

    target_comb = {}
    count = 1
    for key_1 in target_clust_st:
        for key_2 in target_clust_co:
            if key_1 in key_2:
                target_comb[key_1] = [target_clust_st[key_1], 
                                      target_clust_co[key_2], 
                                      count, count+1]
                count +=2
    
    return target_comb
###############################################################################

###############################################################################
def assign_clusters_combined(target_comb, clust_ids_all_st_ind, 
                             clust_ids_all_co_ind, traj_dir, encoder_dir):
    
    n_comb_class = 2*len(target_comb) + 1
    
    clust_ids_comb = []
    for i in range(0, len(clust_ids_all_st_ind)):
        clust_id = n_comb_class
        for key in target_comb:
            if target_comb[key][0] == clust_ids_all_st_ind[i] and target_comb[key][1] == clust_ids_all_co_ind[i]:
                clust_id = target_comb[key][2]
                break
            if target_comb[key][0] == clust_ids_all_st_ind[i] and target_comb[key][1] != clust_ids_all_co_ind[i]:
                clust_id = target_comb[key][3]
                break
        clust_ids_comb.append(clust_id)
            
    # Convert to array
    clust_ids_comb = np.asarray(clust_ids_comb)
        
    # Create directory where cluster ids will be saved
    clust_path = os.path.join(encoder_dir, traj_dir)
    if os.path.isdir(clust_path) == False:
        os.makedirs(clust_path) 
    
    # Save cluster labels
    np.save(os.path.join(clust_path, "Combined_Cluster_IDs_" +
                         str(np.max(clust_ids_comb)) +
                         "_Clusters.npy"), clust_ids_comb)
    
    return clust_ids_comb
###############################################################################
    
###############################################################################
def assign_colors(target_comb, mother_dir):
    
    # Create ordered list of colors where list index corresponds to
    # cluster ID. Note that light colors correspond to compositionally-
    # disordered particles and white corresponds to unlabeled particles.
    # This code currently has 15 available colors but more can be added.
    
    colors_available = []
    colors_available.append(['Red', 238, 32, 77])
    colors_available.append(['Light Red', 246.5, 143.5, 166.0])
    colors_available.append(['Green', 28, 172, 120])
    colors_available.append(['Light Green', 141.5, 213.5, 187.5])
    colors_available.append(['Blue', 31, 117, 254])
    colors_available.append(['Light Blue', 143.0, 186.0, 254.5])
    colors_available.append(['Brown', 180, 103, 77])
    colors_available.append(['Light Brown', 217.5, 179.0, 166.0])
    colors_available.append(["Violet", 146,110,174])
    colors_available.append(['Light Violet', 200.5, 182.5, 214.5])
    colors_available.append(["Gray", 149, 145, 140])
    colors_available.append(["Light Gray", 202.0, 200.0, 197.5])
    colors_available.append(["Orange", 255,117,56])
    colors_available.append(["Light Orange", 200.5, 182.5, 214.5])
    
    n_comb_class = 2*len(target_comb)
    
    colors_list = []
    for i in range(0, n_comb_class):
        colors_list.append(colors_available[i])
    
    colors_list.append(["White", 237,237,237])
    
    # Create colors dictionary
    colors_dict = {}
    count = 0
    for key in target_comb:
        colors_dict['CO-' + key] = colors_list[count]
        colors_dict['CD-' + key] = colors_list[count+1]
        count = count + 2

    # Save colors dictionary
    f = open(mother_dir + "/color_assignments.txt","w")

    # write file
    f.write( str(colors_dict) )

    # close file
    f.close()
    
    return colors_list, colors_dict
###############################################################################

###############################################################################    
def read_xyz(xyz_file):

    s = ' '
    
    # Read XYZ file
    xyz = open(xyz_file, "r")
    line_list = []
    for line in xyz:
        line_list.append(line)
    
    # Get number of particles
    num_part = int(line_list[0].split('\n')[0])
    
    # Get number of frames
    num_frame = int(len(line_list)/num_part)
    
    # Get properties list and lattice dimensions
    lattice_dim = line_list[1].split("Properties=")[0]
    prop_list = line_list[1].split("Properties=")[1].split(":")
    
    # Get relative indices and positions
    dt = []
    dt_space = []
    for i in range(0, len(prop_list)):
        if i % 3 == 0:
            dt.append(prop_list[i])
        elif i % 3 == 2:
            dt_space.append(prop_list[i])
    
    # Remove "\n" character
    for i in range(0, len(dt_space)):
        if "\n" in dt_space[i]:
            dt_space[i] = dt_space[i].replace("\n", "")
    
    # Find species index
    spec_ind_header = dt.index('species')
    spec_ind_body = 0
    for i in range(0,spec_ind_header):
        spec_ind_body = spec_ind_body + int(dt_space[i])
    
    # Find position index
    pos_ind_header = dt.index('pos')
    pos_ind_body = 0
    for i in range(0, pos_ind_header):
        pos_ind_body = pos_ind_body + int(dt_space[i])
        
    # See if position is given in 2D or 3D
    dim = int(dt_space[dt.index('pos')])
    
    # Find radius index
    rad_ind_header = dt.index('radius')
    rad_ind_body = 0
    for i in range(0, rad_ind_header):
        rad_ind_body = rad_ind_body + int(dt_space[i])


    # Get indices of lines that contain either (a) number of particles or (b)
    # property information
    nd_list = []
    for i in range(0, len(line_list)):
        if i % int(num_part + 2) == 0:
            nd_list.append(i)
            nd_list.append(i+1)
    rad_list = []
    pos_list = []
    spec_list = []
    
    for i in range(0, len(line_list)):
        line_split = line_list[i].split(s)
        if (i in nd_list) == False:
            rad_list.append(line_split[rad_ind_body])
            spec_list.append(line_split[spec_ind_body])
            pos_list.append(line_split[pos_ind_body:pos_ind_body+dim])
    
    prop_line_pt_1 = lattice_dim
    prop_line_pt_2 = "Properties=species:S:1:pos:R:3:radius:R:1:color:R:3:Transparency:R:1\n"
    prop_line = prop_line_pt_1 + prop_line_pt_2
    
    # Remove "\n" character
    for i in range(0, len(spec_list)):
        if "\n" in rad_list[i]:
            rad_list[i] = rad_list[i].replace("\n", "")
        if "\n" in spec_list[i]:
            spec_list[i] = spec_list[i].replace("\n", "")
        if "\n" in pos_list[0][dim-1]:
            spec_list[i] = spec_list[i].replace("\n", "")
                 
        
    return spec_list, pos_list, rad_list, num_frame, num_part, dim, prop_line        
###############################################################################  

############################################################################### 
def write_xyz(spec_list, pos_list, rad_list, color_list, trans_list, nframe,
              npart, dim, prop_line, out):
    
    if dim == 2:
        f=open(out,'w')
        for i in range (nframe):
            f.write(str(npart) + "\n")
            f.write(prop_line)
            for j in range(npart):
                f.write("{} {} {} {} {} {} {} {}\n".format(spec_list[j+i*npart], 
                        pos_list[j+i*npart][0],
                        pos_list[j+i*npart][1],
                        rad_list[j+i*npart],
                        color_list[j+i*npart][0],
                        color_list[j+i*npart][1],
                        color_list[j+i*npart][2],
                        trans_list[j+i*npart]))
    elif dim == 3:
        f=open(out,'w')
        for i in range (nframe):
            f.write(str(npart) + "\n")
            f.write(prop_line)
            for j in range(npart):
                f.write("{} {} {} {} {} {} {} {} {}\n".format(spec_list[j+i*npart],
                        pos_list[j+i*npart][0],
                        pos_list[j+i*npart][1], 
                        pos_list[j+i*npart][2],
                        rad_list[j+i*npart],
                        color_list[j+i*npart][0],
                        color_list[j+i*npart][1],
                        color_list[j+i*npart][2],
                        trans_list[j+i*npart]))
###############################################################################             
           
###############################################################################
def create_XYZ(clust_ids_all, colors_list, xyz_dir, encoder_dir):
    
    # Find xyz file
    xyz_file, xyz_name = find_xyz_path(xyz_dir)
    
    # Read XYZ file
    spec, pos, rad, nframe, npart, dim, prop = read_xyz(xyz_file)
    
    # Create directory where re-written xyz file will be saved
    new_xyz_path = os.path.join(encoder_dir, xyz_dir)
    if os.path.isdir(new_xyz_path) == False:
        os.makedirs(new_xyz_path)
    
    # Extract scaled RGB values from input colors
    RGB_scaled = []
    for i in range(0, len(colors_list)):
        RGB_scaled.append(np.asarray(colors_list[i][1:4])/255)
    
    # Create new xyz file
    new_filename = (xyz_name + "_" + str(np.max(clust_ids_all)) +
    "_Clusters.xyz")
    new_filename = os.path.join(new_xyz_path, new_filename)
    
    # Assign colors to each GDV
    disc_colors = []
    for i in range(0, len(clust_ids_all)):
        r = str(RGB_scaled[clust_ids_all[i]-1][0])
        g = str(RGB_scaled[clust_ids_all[i]-1][1])
        b = str(RGB_scaled[clust_ids_all[i]-1][2])
        disc_colors.append([r, g, b])
    
    # Assign transparencies to each GDV
    disc_trans = []
    for i in range(0, len(clust_ids_all)):
        if clust_ids_all[i] == len(colors_list):
            disc_trans.append(0.8)
        else:
            disc_trans.append(0)
    
    # Write new discrete XYZ file
    write_xyz(spec, pos, rad, disc_colors, disc_trans, nframe, npart, dim, 
              prop, new_filename)
###############################################################################
