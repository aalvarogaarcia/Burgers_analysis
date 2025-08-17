# **Numerical Analysis of the Two-Dimensional Burgers' Equation using High-Order Schemes and Large Eddy Simulation (LES) Models**

This repository hosts a computational solver developed in Python, intended for the simulation and analysis of the Burgers' equation within a two-dimensional domain. The fundamental objective of this project is to conduct a comparative performance evaluation of various numerical schemes and sub-grid scale models for Large Eddy Simulation (LES), applied to the resolution of flows exhibiting turbulent characteristics.

[🚀 Core Capabilities](#heading=)

[Numerical Discretization Schemes](#heading=)

[Turbulence Models for LES](#heading=)

[📁 Repository Structure](#heading=)

[⚙️ Solver Usage Guide](#heading=)

[1\. Test Case Generation](#heading=)

[2\. Simulation Execution](#heading=)

[3\. Results Analysis](#heading=)

[Visualization of Physical Fields](#heading=)

[Energy Spectrum Analysis](#heading=)

[Comparison of Temporal Statistics](#heading=)

[🔮 Future Research Directions](#heading=)

## **🚀 Core Capabilities**

The solver integrates a variety of numerical methodologies and physical models, which facilitates a comprehensive and detailed comparative analysis.

### **Numerical Discretization Schemes**

* **Flux Reconstruction (FR)**: A high-order spatial method that employs Lagrange polynomials defined on Gauss-Lobatto points. This scheme is characterized by its high accuracy, although it may exhibit numerical instabilities in the absence of adequate dissipation.  
* **Centered Differences (DC)**: A classic second-order scheme that, from a theoretical perspective, is non-dissipative. It serves as a baseline reference for the comparative evaluation of the accuracy and stability of other methods.

### **Turbulence Models for LES**

* **Implicit Large Eddy Simulation (ILES)**: Corresponds to a simulation without an explicit sub-grid scale model. In this approach, numerical stability depends entirely on the inherent dissipation of the numerical scheme or the physical viscosity of the fluid.  
* **Smagorinsky Model**: Constitutes the canonical sub-grid scale (SGS) model. It calculates an artificial turbulent viscosity based on the magnitude of the strain-rate tensor of the resolved flow.  
* **Vreman Model**: A more recently conceived SGS model, formulated to minimize dissipation in laminar or transitional flow regions. This characteristic allows for a more faithful representation of flows with mixed regimes.

## **📁 Repository Structure**

The project is organized into the following main directories, each with a specific function:

* fr-burgers-2d.py: The main script for executing simulations.  
* data/: A directory containing the input configuration files (inputs/) and where the numerical results will be stored (outputs/).  
* src/: Contains the core of the solver, including modules for mesh generation, implementation of numerical schemes, LES models, and other utility functions.  
* tools/: Hosts a set of tools for the systematic generation of test cases and for the post-processing and analysis of the obtained results.

## **⚙️ Solver Usage Guide**

The operational procedure is structured into three sequential phases: case generation, simulation execution, and results analysis.

### **1\. Test Case Generation**

The generate\_cases.py script allows for the automated creation of configuration files (.txt) for a wide range of simulations, organizing them systematically into subdirectories.  
\# Execute from the project's root directory  
python tools/generate\_cases.py

Executing this command will populate the data/inputs/ directory with subfolders such as no\_les\_stable/, smagorinsky/, and vreman/.

### **2\. Simulation Execution**

The main solver, fr-burgers-2d.py, is designed to process both individual input files and batches of files. The results will be stored in the data/outputs/ directory, replicating the organizational structure of the input files.  
\# Execution of a single configuration file  
python fr-burgers-2d.py data/inputs/vreman/FR\_TG\_Visc0.005.txt

\# Execution of all cases in a directory (requires the use of quotes)  
python fr-burgers-2d.py "data/inputs/smagorinsky/\*.txt"

The program has been endowed with robustness: in the event that a simulation exhibits numerical instability, the solver will save the last stable time step and proceed autonomously to the next case in the sequence.

### **3\. Results Analysis**

The tools/analysis/ directory contains a set of scripts for post-processing the generated data.

#### **Visualization of Physical Fields**

The plot\_results.py script generates a graphical representation (.png) for each results file, illustrating the kinetic energy, vorticity, and velocity field.  
\# Analyze all results from the 'vreman' directory  
python tools/analysis/plot\_results.py "data/outputs/vreman/\*.txt"

#### **Energy Spectrum Analysis**

The compute\_spectrum.py script is responsible for calculating the kinetic energy spectrum, a quantitative tool that is indispensable in the study of turbulence.  
\# Generate the spectrum for a specific result  
python tools/analysis/compute\_spectrum.py data/outputs/vreman/FR\_TG\_Visc0.005.txt

#### **Comparison of Temporal Statistics**

The compute\_statistics.py script is oriented towards model comparison. It processes the results of multiple simulations to generate a unified graph showing the temporal evolution of kinetic energy for each case.  
\# Compare the energy evolution for the three implemented models  
python tools/analysis/compute\_statistics.py "data/outputs/vreman/\*.txt" "data/outputs/smagorinsky/\*.txt" "data/outputs/no\_les\_stable/\*.txt"

## **🔮 Future Research Directions**

This project constitutes a solid foundation for future research and development. The priority lines of work that are contemplated are as follows:

* **Development of a Graphical User Interface (UI)**: The implementation of a web-based user interface is planned, utilizing the existing components in tools/webapp/. The objective is to facilitate the configuration, execution, and visualization of simulations in an interactive manner, thereby improving the solver's accessibility.  
* **Analysis of Forced Turbulence**: The incorporation of a forcing term into the Burgers' equation is foreseen. Such a modification will allow for the study of flows in a statistically stationary state of turbulence, which is ideal for a more rigorous analysis of the energy cascade and the effectiveness of LES models in dissipating the energy injected into the system.
