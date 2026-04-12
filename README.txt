########################################
Additions & updates to alchemy pipeline
########################################

1. Python versions of Assistant and Alchemy.
Assistant updated to download structure factor file and corresponding coordinate file from PDB-REDO.

2. Alloy Script
Reads CCD CIF dictionary and compiles a document containing CCDs of all metallocofactors in the PDB; can be used to update metallocofactor document. metallocofactors_id.txt is updated as of 06/16/2025. called in Analysis to recognize metal-containing cofactors listed in Edstats output.

3. Analysis
a) Python version b) added functionality to recognize metallocofactors and creates an additional output file containing cofactor edstats information. Output: 1 folder per metal containing txt files for each metal, cofactors

4. Autoplot
Python version; can create images for metals found in cofactors as well as ions. Relies on default.mgpic.

5. *new* Bond distance analysis [ needs name ]
called fe_biopython_analysis_dpi_final.py right now.  :(
Output: CSV file; 1 row per metal bond containing bond information, metal information, sigma value
Relies on: metal_distances file; [pdbID]metals folder containing analysis outputs (to add sigma values for metals to output csv file); pdb_rszd.pdb for structure parsing [pdb file with no header]; .pdb file with header for DPI calculation*
* _rszd output from alchemy; may be able to simplify dependancies by parsing structure from .pdb file but I haven't tried that so I dont know if that breaks the parser. 
Finds metal atoms, calculates distance to each neighboring atom, records distance, deviation, z score, protein & metal information, and sigma value. 

########################################
Also included in folder:
########################################
1. results folder
contains CSV outputs for several groups of proteins that alchemy pipeline was run on including: proteins known to be misidentified; Ni proteins, the test I did to make sure pdb redo coordinate files gave the same results as the rscb pdb files, fe bond file I generated, the output for re running alchemy pipeline on re refined 7cup, and lists of the pdb IDs that contain iron/ pdbs with iron and electron data

2. scratchwork.py
not part of the alchemy pipeline but contains scripts and methods for generating the figures used on the poster so I thought it may come in handy.

3. read_metal_pdb.py
to get lists of proteins containing a certain metal I went to metalpdb and searched by a particular metal. this script is what I used to extract a list of pdb IDs from that file that had the metal of interest. 

4. check_progress.py
totally extra; additional script I ran to see how complete different scripts are - load the file pdb IDs and then enter the pdb ID the script of interest is currently on, it will calculate what % through the list the script is. 
