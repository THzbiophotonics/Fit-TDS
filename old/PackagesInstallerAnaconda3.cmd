set working_dir=%cd%
cd C:\
set pathconda=''
for /F "delims=" %%I in ('dir /S /B "Anaconda3"') do set pathconda=%pathconda%;%%I
for /F "delims=" %%I in ('dir /S /B "Anaconda3"') do set pathconda=%pathconda%;%%I\condabin
for /F "delims=" %%I in ('dir /S /B "Anaconda3"') do set pathconda=%pathconda%;%%I\Scripts
set Path=%Path%;%pathconda%

cd %working_dir%

echo set Path=%Path%;%pathconda% > FitTDSeasyAnaconda3.cmd
echo call conda activate >> FitTDSeasyAnaconda3.cmd
echo call set/p Nb_proc="How many process do you want to use? " >> FitTDSeasyAnaconda3.cmd
echo call mpiexec -n %%Nb_proc%% python fit@TDS.py >> FitTDSeasyAnaconda3.cmd
echo pause >> FitTDSeasyAnaconda3.cmd

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