# colloid_char
The provided scripts implement a three-step framework for the state characterization of binary three-dimensional colloidal self-assembly systems. The framework takes .xyz files from molecular simulations as input. These .xyz files (at a minimum) must specify particle species, particle radius, and particle positions. In the first step, branched graphlet decomposition is used to quantify the local topology of each particle in each provided .xyz file in the form of structural and compositional neighborhood graphs. The second step uses a deep autoencoder to reduce the dimensionality of the neighborhood graphs and create structural and compositional low-dimensional spaces. The third step employs agglomerative hierarchical clustering to partition the low-dimensional spaces and assign classifications (e.g., FCC-CuAu, substitutionally defective HCP) to the resulting partitions. The papers explaining these methodologies can be found in the "citation" section of this file.

# Branched Graphlet Decomposition
An example of the branched graphlet decomposition is provided in the folder labeled "BGD_Example". Here, an example .xyz file is processed to form three ".gdv" files (where "gdv" stands for "graphlet degree vector") and one "vapor.npy" file. The ".gdv" files contain different (evaluated) neighborhood graphs. The structural neighborhood graphs are composed of all particles in the neighborhood, while the compositional neighborhood graphs are composed of (i) all particles that are of the same species of the particle of interest and (ii) all particles that are of a different species than the particle of interest. Note that this code is tailored towards binary lattices in which only two types of particles exist (e.g., A- and B-type particles). The structural neighborhood graphs can be found in the "all.gdv" file while the compositional neighborhood graphs can be found in the "same.gdv" and "diff.gdv" files. The "vapor.npy" file indicates particles that are in a vapor state (index = 1 means vapor, all other indices mean non-vapor).

Note that if the user is not interested in particle type, or is using a system that only contains one type of particle, the user only needs to use the structural neighborhood graphs.

# Dimensionality Reduction and Clustering
Code for the dimensionality reduction and clustering framework steps can be found in the folder titled "DimRedandClust". These codes, titled "core.py" and "Train.ipynb" take ".xyz", ".gdv" and "vapor.npy" files as input. An example implementation of the Dimensionality Reduction and Clustering can be found in DimRedandClust/Example.

The Example folder contains the following:

Example_Data_1/2.zip --> The folder labeled "1.0" contains 10 trajectories of the colloidal self-assembly of a binary system of equally-sized particles. The folder labeled "1.05" contains 10 trajectories of the colloidal self-assembly of a binary system of particles with a 5% size disparity. The subfolders refer to different A/B interaction strengths

Train.zip --> This folder contains the results of the characterization framework on these 20 trajectories.

Train_example.ipynb --> This shows Train.py from the main folder filled out for this specific application

core.py --> Identical copy of core.py from main folder (for ease)

# Notes
The method here is an improved version of the method originally reported in "Deep learning for characterizing the self-assembly of three-dimensional colloidal systems" (O'Leary, et al., 2021).

# Help
Please direct all questions to jared.oleary@berkeley.edu.

# Citation
@article{mao2021branched,
  title={Branched graphlet decomposition with deep learning for structural and compositional characterization of three-dimensional binary colloidal superlattices},
  author={Mao, Runfang and O'Leary, Jared and Mesbah, Ali and Mittal, Jeetain},
  journal={Submitted},
  volume={},
  number={},
  pages={},
  year={2022},
  publisher={}
}

@article{o2021deep,
  title={Deep learning for characterizing the self-assembly of three-dimensional colloidal systems},
  author={O’Leary, Jared and Mao, Runfang and Pretti, Evan J and Paulson, Joel A and Mittal, Jeetain and Mesbah, Ali},
  journal={Soft Matter},
  volume={17},
  number={4},
  pages={989--999},
  year={2021},
  publisher={Royal Society of Chemistry}
}
