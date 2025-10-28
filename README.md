# gpAMR


installation: 

"module load python/3.11"

download the repo

"python -m venv name_of_your_environment"

"source name_of_your_environment/bin/activate"

"pip install -r requirements"


The add the ipykernel:

python -m ipykernel install --user --name name_of_your_environment --display-name name_of_your_environment


Open two terminals on your favorite supercomputing platform.

open "allocate_GPUs.sh" and change the account number

Then, in one terminal, run:
"./allocate_GPUs.sh [number_of_nodes] [number_of_workers]"

for instance "./allocate_GPUs 4 16".

then run "./launch-dask-moduleGPU [same_number_of_nodes] [same_number_of_workers]" in the same terminal.

This is will start the dask scheduler, and the workers, the Jupyter Notebook "gpAMR.ipynb" (or similar) will connect to. 

Open the jupyter notebook and chose your ipykernel.

In the other terminal, go to ./ChomboOut

and run "./launch_chombo.sh".

Now run all cells in the Jupyter Notebook.




