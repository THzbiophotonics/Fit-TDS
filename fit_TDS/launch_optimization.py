import subprocess 
command = "" 
process = subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE) 
output,error = process.communicate() 
print("Output : " + str(output) + "\n Error: " + str(error) + "\n")