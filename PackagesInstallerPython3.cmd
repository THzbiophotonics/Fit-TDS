call set Path=%Path%;C:\ProgramData\Anaconda3
call set Path=%Path%;C:\ProgramData\Anaconda3\condabin
call set Path=%Path%;C:\ProgramData\Anaconda3\Scripts

call conda activate

call conda install -c anaconda numpy

call pip install mpi4py

call conda install -c anaconda Swig

call conda install -c anaconda MinGW

call conda install -c anaconda tk

call cd pyOpt-master

call python setup.py install

call pip install --upgrade pyswarm

pause