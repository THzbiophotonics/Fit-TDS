call set Path=%Path%;C:\ProgramData\Anaconda2
call set Path=%Path%;C:\ProgramData\Anaconda2\condabin
call set Path=%Path%;C:\ProgramData\Anaconda2\Scripts

call conda activate

call conda install -c anaconda numpy

call conda install -c anaconda mpi4py

call conda install -c anaconda Swig

call conda install -c anaconda MinGW

call conda install -c anaconda tk

call conda install -c mutirri pyopt

call pip install --upgrade pyswarm

pause