# gpAMR


installation (example for NERSC's Perlmutter): 


Go to your system's JupyterHub (for NERSC: https://jupyter.nersc.gov/hub/home) and start a server.

Open a terminal (via ssh or via the hub)

Python3.11 or 3.12 should be availible:

"module load python/3.11-24.1.0"
(check with "python --version")

Download the repo ("git clone https://github.com/MarcusMNoack/gpAMR").

"python -m venv name_of_your_environment"

"source name_of_your_environment/bin/activate"

"pip install -r requirements"


Then add the ipykernel:

python -m ipykernel install --user --name name_of_your_environment --display-name name_of_your_environment


Open two terminals, then:


In the first, open "allocate_GPUs.sh"/"allocate_CPUs.sh", change the account number, and perform other required changes based on the system requirements.
There are a few "allocate_CPUsXXX.sh" examples for a user to look at. This is the allocation the gpAgents will use.

Open "./launch-dask-moduleCPU" or "./launch-dask-moduleGPU" and adjust it based on your system configuration; especially make sure the script activates your Python environment. 

Then, in that terminal, run:

"./allocate_CPUs.sh [number_of_nodes] [number_of_workers]", for instance "./allocate_CPUs 4 16" (for 16 workers on 4 nodes, or use the GPU equivalent).

Then run "./launch-dask-moduleCPU [same_number_of_nodes] [same_number_of_workers]" (or the GPU equivalent) in the same terminal.

This will start the Dask scheduler and the workers. The Jupyter Notebook "gpAMRXXX.ipynb" (or similar) will connect to those resources (don't hit CTRL-C in this terminal). 

Open the Jupyter notebook (in the Jupyter Hub) and choose the right ipykernel (name_of_your_environment) on the top right.

In the unused terminal, go to "./ChomboOut".

Adapt the "jobscript.sh" and run "./launch_chombo.sh".

Now run all cells in the Jupyter Notebook.

Done!
