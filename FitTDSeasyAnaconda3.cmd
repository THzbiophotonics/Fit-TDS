call set Path=%Path%;C:\ProgramData\Anaconda3
call set Path=%Path%;C:\ProgramData\Anaconda3\condabin
call set Path=%Path%;C:\ProgramData\Anaconda3\Scripts

call conda activate

call set/p Nb_proc="How many process do you want to use? "

call mpiexec -n %Nb_proc% python fit@TDS.py

pause