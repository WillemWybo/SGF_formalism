Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva

Dependencies:
python 2.7
numpy 1.9.1
scipy 0.15.0
sympy 0.7.6
matplotlib 1.4.2
cython 0.20.1
NEURON 7.3

Add the 'source/' directory to the python path.
Run 'nrnivmodl mech/'

To compile the cython code:
'cd source/cython_code'
'sh install.sh'
add the directory '~/local/lib/python2.7/site-packages' to the python path