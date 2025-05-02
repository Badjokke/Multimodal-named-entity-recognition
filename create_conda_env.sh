declare python_version="3.10"
declare env_name="multimodal_named_entity"
conda create -n $env_name python=$python_version 
conda activate $env_name
 pip3 install -r requirements.txt