for i in {0,1,2}; do python max_wfn_energy.py $i >> energies.txt; done
for i in {5,10,20}; do python max_wfn_energy.py $i >> energies.txt; done
for i in {50,100,300}; do python max_wfn_energy.py $i >> energies.txt; done
for i in {500,800,1000,1400,1800,2500}; do python max_wfn_energy.py $i >> energies.txt; done
for i in {5000,7500,10000}; do python max_wfn_energy.py $i >> energies.txt; done
for i in {100000,1000000}; do python max_wfn_energy.py $i >> energies.txt; done
