github_username=$1


echo "======================================================================================"
echo "setup root..."
# Set ROOT by hand
#export ROOTSYS="/opt/root/buildthis"
#export PATH="$ROOTSYS/bin:$PATH"
#export LD_LIBRARY_PATH="$ROOTSYS/lib:$LD_LIBRARY_PATH"
#export PYTHONPATH="/opt/root/buildthis/lib:$PYTHONPATH"
source /setup_root.sh

echo "======================================================================================"
echo "setup gaugi..."
# current=$PWD
# cd git_repos
cd ..
git_path=$PWD
cd $git_path/gaugi && source scripts/setup.sh

echo "======================================================================================"
echo "setup kepler..."
cd $git_path/kepler && source scripts/setup.sh

echo "======================================================================================"
echo "setup jodafons..."
cd $git_path/jodafons && source scripts/setup.sh

echo "======================================================================================"
echo "setup saphyra..."
cd $git_path/saphyra && source scripts/setup.sh

echo "======================================================================================"
echo "setup rootplotlib..."
cd $git_path/rootplotlib && source scripts/setup.sh

echo "======================================================================================"
echo "setup pybeamer..."
cd $git_path/pybeamer && source scripts/setup.sh

echo "======================================================================================"
echo "setup kolmov..."
cd $git_path/kolmov && source scripts/setup.sh

echo "======================================================================================"
echo "all packages can be found at $git_path."
echo "======================================================================================"
cd $current
