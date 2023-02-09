# Usage:
# $launch_jupyter <setup-script> <jupyter-port>
# $launch_jupyter setup_repos_kepler.sh 1234

cd ~
cd workspace
source $1
cd ~
jupyter lab --no-browser --port $2
