# Alchemy
# To use CCP4 commands in python via subprocess module
import os
import subprocess


# home directory containing python script. files will also download and unzip
# to this directory but that could be changed. 
directory="/home/knamikas/BioXFELproject/alchemyTesting_python/" 
FILE_PATH="/home/knamikas/BioXFELproject/alchemyTesting_python/" 

#get resolution and wavelength

def alchemy(pdbID):
# use subprocess to run mtz dump and put the info in a temporary file (this gets
# overwritten each time the program loops through the code for a different pdb
# identifier. stderr catches errors
    input_file = '/home/knamikas/CCP4I2_PROJECTS/7cup/CCP4_JOBS/job_18/18_7cup_fphiout_prosmart_refmac.mtz'
    output_file = os.path.join(directory, "tempmtz")
    with open(output_file, "w") as out:
        subprocess.run(["mtzdump", "HKLIN", input_file],
                       input="GO\n",
                       text=True,
                       stdout=out,
                       stderr=subprocess.PIPE)
# read out file and extracts individual lines into a list
# low and high resolution limits are line 73, positions 4 and 6. don't forget
# python starts at 0, so these will be called as indices 72, 3, and 5. 
    if pdbID == '7cup':
        lowres=47.41
        highres=2.00
        wavelength=0.97892
    else:
        with open(output_file) as f:
            lines = f.readlines()
        line_73 = lines[72]
        fields = line_73.split()
        lowres = fields[3]
        highres = fields[5]
        print(lowres)
        print(highres)
# extract wavelength from line 43, the first position
        line_43 = lines[42]
        wavelength = line_43.split()[0]
        print(wavelength)

# Find anomalous data if present
    with open(output_file) as f:
        mtzText = f.read()
    if "FAN" in mtzText:
        ANOM = 1
        print("Anom data available as "+wavelength)
    else:
        ANOM = 0
        print("No Anom data available for "+pdbID)

# Calculate Observed Fo map
    print("calculating observed map for "+pdbID)

    output_map = FILE_PATH+pdbID+"_fo.map"
    log_file = FILE_PATH+"tmp_fo.map"

    fftCommand = ["fft", "HKLIN", input_file, "MAPOUT", output_map]
    fftInput = "labi F1=FWT PHI=PHWT\nGRID SAMP=5\n"

    with open(log_file, 'w') as log:
        subprocess.run(fftCommand, input=fftInput, text=True, stdout=log,
                       stderr=subprocess.PIPE)

# Calculate Difference Df map
    print("Calculating difference map for "+pdbID)
    output_dfmap = os.path.join(directory, pdbID+"_df.map")
    logdf_file = os.path.join(directory, "tmp_df.map")

    fftCommand_df = ["fft", "HKLIN", input_file, "MAPOUT", output_dfmap]
    fftInput_df = "labi F1=DELFWT PHI=PHDELWT\nGRID SAMP=5\n"

    with open(logdf_file, 'w') as log:
        subprocess.run(fftCommand_df, input=fftInput_df, text=True, stdout=log,
                       stderr=subprocess.PIPE)

#Anom Difference map
    if ANOM ==1:
        print("calculating Anom difference map for"+pdbID)
        output_anom_map = os.path.join(directory, pdbID+"_am.map")
        log_anom_file = os.path.join(directory, "tmp_am.map")

        fftCommand_anom = ["fft", "HKLIN", input_file, "MAPOUT", output_anom_map]
        fftInput_anom = "labi F1=DELFWT PHI=PHDELWT\nGRID SAMP=5\n"

        with open(log_anom_file, 'w') as log:
            subprocess.run(fftCommand_anom, input=fftInput_anom, text=True, stdout=log,
                           stderr=subprocess.PIPE)
    else:
        pass
# Run Edstats
    res_limits = f'reslo={lowres},reshi={highres}\n'
    command = ['edstats', "XYZIN", '/home/knamikas/CCP4I2_PROJECTS/7cup/CCP4_JOBS/job_18/18_7cup_xyzout_prosmart_refmac.pdb',
               'MAPIN1', '/home/knamikas/CCP4I2_PROJECTS/7cup/CCP4_JOBS/job_18/18_7cup_fphiout_prosmart_refmac.mtz',
               "MAPIN2", '/home/knamikas/CCP4I2_PROJECTS/7cup/CCP4_JOBS/job_18/18_7cup_diffphiout_prosmart_refmac.mtz',
               "XYZOUT", os.path.join(directory, pdbID+"_rszd.pdb"),
               "OUT", os.path.join(directory, pdbID+"_stats.out"),
               "QQDOUT", os.path.join(directory, pdbID+"_qq.out")
               ]
    log_edstats_file = os.path.join(directory, pdbID+"_edstats.log")
    with open(log_edstats_file, 'w') as log:
        subprocess.run(command, input=res_limits, text=True, stdout=log,
                       stderr=subprocess.PIPE)
        


# Run edstats on anomalous map
    if ANOM == 1:
        print("running Edstats on"+pdbID)
        edstats_anom = ['edstats', "XYZIN", os.path.join(directory, pdbID+".pdb"),
                        'MAPIN1', os.path.join(directory, pdbID+"_fo.map"),
                        "MAPIN2", os.path.join(directory, pdbID+"_am.map"),
                        "XYZOUT", os.path.join(directory, pdbID+"_am_rszd.pdb"),
                        "OUT", os.path.join(directory, pdbID+"_stats_am.out"),
                        "QQDOUT",os.path.join(directory, pdbID+"_am_qq.out")
               ]
        log_edstatsam_file = os.path.join(directory, pdbID+"_am_edstats.log")
        with open(log_edstatsam_file, 'w') as log:
            subprocess.run(command, input=res_limits, text=True, stdout=log,
                           stderr=subprocess.PIPE)
        
    


# take in a file, convert to list containing PDB identifier strings

#fileIn = input ("Enter CSV file for PDB identifiers: ")
#fileIn = "alchemyTest.txt"


#with open(fileIn, "r") as file:
#	pdbString = file.read()

# make a list of strings, each string is one pdb ID and clear each entry of any
# extra spaces

#pdbList = [x.strip() for x in pdbString.split(",")]
#pdbList = [s for s in pdbList if s]
#print(pdbList)


# Run through the analysis steps for each identifier
pdbList = ['7cup']
i = 0

while i<len(pdbList):
    #if os.path.exists(FILE_PATH+pdbList[i]+"_0cyc.mtz")==True:
    alchemy(pdbList[i])
    #else:
        #print("No mtz file exists for "+pdbList[i])
    i +=1

