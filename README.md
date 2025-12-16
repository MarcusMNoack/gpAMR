# gpAMR


installation (example for NERSC's Perlmutter): 

"module load python/3.11"

download the repo ("git clone https://github.com/MarcusMNoack/gpAMR")

"python -m venv name_of_your_environment"

"source name_of_your_environment/bin/activate"

"pip install -r requirements"


Then add the ipykernel:

python -m ipykernel install --user --name name_of_your_environment --display-name name_of_your_environment


Open two terminals on your favorite supercomputing platform (here Perlmutter).

open "allocate_GPUs.sh" and change the account number and perfom other required changes based on the system requirements

Then, in one terminal, run:
"./allocate_GPUs.sh [number_of_nodes] [number_of_workers]"

for instance "./allocate_CPUs 4 16" (for 16 workers on 4 nodes).

then run "./launch-dask-moduleGPU [same_number_of_nodes] [same_number_of_workers]" in the same terminal.

This will start the Dask scheduler and the workers. The Jupyter Notebook "gpAMR.ipynb" (or similar) will connect to those resources. 

Open the Jupyter notebook and choose the right ipykernel (name_of_your_environment).

In the other terminal, go to ./ChomboOut

Adapt the "jobscript.sh"

and run "./launch_chombo.sh".

Now run all cells in the Jupyter Notebook.




