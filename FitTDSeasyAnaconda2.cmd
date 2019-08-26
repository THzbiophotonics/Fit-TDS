call set Path=%Path%;C:\ProgramData\Anaconda2
call set Path=%Path%;C:\ProgramData\Anaconda2\condabin
call set Path=%Path%;C:\ProgramData\Anaconda2\Scripts

call conda activate

call set/p Nb_proc="How many process do you want to use? "

call mpiexec -n %Nb_proc% python fit@TDS.py

pause