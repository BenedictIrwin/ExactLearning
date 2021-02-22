### For the demo/walkthrough pertaining to the most recent journal submission, please see the contents of "ExactLearning.zip"

# ExactLearning
Algorithms to exactly learn analytic functions given numerical data.
Within each directory is an example of fitting a function using the algorithm and the scripts that are run to do this.


Update:
Had a nice improvement by including a closed form suggestion for solved parameters: 

```
*** Assembling Dictionary ***
*** Loading Data ***
*** Initial Guess ***
Params:  [ 4.79593526e-01  1.18914819e+00  1.95630284e-04  5.00033813e-01
 -1.33307505e+00  1.33342100e+00]
Loss:  0.0439870044004525
*** Refined Guess ***
Params:  [ 4.79593396e-01  1.18914822e+00  1.95813609e-04  5.00033765e-01
 -1.33307513e+00  1.33342095e+00]
Loss:  4.604286890498946e-06
Final Loss: 4.604286890498946e-06
Situation Report
p[1]^4 ~ 2
p[3] ~ 1/2
p[3]^2 ~ 1/4
p[0] ~ diff(quot(e,4),recip(5)) : i.e. 0.47959339594500594 ~ 0.47957045711476126
p[1] ~ 4rt(2) : i.e. 1.1891482240441165 ~ 1.189207115002721
p[2] ~ 0 : i.e. 0.00019581360916189755 ~ 0.0
p[3] ~ recip(2) : i.e. 0.5000337654908391 ~ 0.5
p[4] ~ diff(0,quot(4,3)) : i.e. -1.3330751254138067 ~ -1.3333333333333333
p[5] ~ diff(2,quot(2,3)) : i.e. 1.3334209495203304 ~ 1.3333333333333335
```
All of these steps apart from p[0] are actually correct (just a little convoluted) for this solution.


# 1. System requirements:
Requires scientific python3 packages including numpy, scipy
For advanced functions (hypergeometric, Meijer-G it uses mpmath libraries).

Versions the software has been tested on:
Tested with Windows 10 Pro
Tested with numpy 1.19.4
Tested with scipy 1.1.0
Tested with mpmath 1.1.0
Tested with matplotlib 3.0.0

Any required non-standard hardware:
 None

# 2. Installation guide:
Instructions:
'''
pip install numpy
pip install scipy
pip install mpmath
pip install matplotlib
pip install mpl_toolkits
'''
Typical install time on a "normal" desktop computer:

Up to 5 mins

# 3. Demo:
Instructions to run on data:

  > python demo.py

Expected output:
Subject to random variation, the output is as follows:

'''
(base) C:\...\NatureCodeZip>python demo.py
Writing function:  :first:max:c:linear-gamma krqksrcjnq
New record solution! 0.10132724594367028
:first:max:c:linear-gamma
New record solution! 0.09931376404638874
:first:max:c:linear-gamma
New record solution! 0.09926624160764495
:first:max:c:linear-gamma
Writing function:  :first:max:P1:c:c^s:linear-gamma vigwrqnkmn
New record solution! 0.020572485002746624
:first:max:P1:c:c^s:linear-gamma
New record solution! 0.019384079694340074
:first:max:P1:c:c^s:linear-gamma
New record solution! 0.0191125460305366
:first:max:P1:c:c^s:linear-gamma
New record solution! 0.018722150031465823
:first:max:P1:c:c^s:linear-gamma
Writing function:  :first:max:P2:c:c^s:linear-gamma vkgexengcz
New record solution! 5.5403697034886375e-06
:first:max:P2:c:c^s:linear-gamma
Writing function:  :first:max:P1:c:c^s:linear-gamma:neg-linear-gamma zsxwslmziw
Writing function:  :first:max:P2:c:c^s:linear-gamma:neg-linear-gamma jcujykyllv
Best result is:
- Fingerprint:  :first:max:P2:c:c^s:linear-gamma
- Parameters:  [-1.33323776e+00  1.33299577e+00  4.79888438e-01  1.18887566e+00
  2.65380053e-04  5.00171962e-01]
- Loss:  5.5403697034886375e-06
***  p0 : _poly-coeff_  ***
-1.3332377607205095 ~ -4/3  (Δ = 9.557261282377993e-05)
-1.3332377607205095 ~ -19/14  (Δ = 0.023905096422347727)
-1.3332377607205095 ~ -17/13  (Δ = 0.02554545302820177)
-1.3332377607205095 ~ -15/11  (Δ = 0.030398602915854056)
***  ~~~  ***
***  p1 : _poly-coeff_  ***
1.3329957734514948 ~ 4/3  (Δ = 0.0003375598818384784)
1.3329957734514948 ~ 19/14  (Δ = 0.024147083691362425)
1.3329957734514948 ~ 17/13  (Δ = 0.02530346575918707)
1.3329957734514948 ~ 15/11  (Δ = 0.030640590184868755)
***  ~~~  ***
***  p2 : _sqrtnotzero_  ***
0.47988843760466493 ~ sqrt(3/13)  (Δ = 0.0004960238105964909)
0.47988843760466493 ~ sqrt(4/17)  (Δ = 0.005182812468001008)
0.47988843760466493 ~ sqrt(2/9)  (Δ = 0.008483916813633252)
0.47988843760466493 ~ sqrt(3/14)  (Δ = 0.01697838771838922)
***  ~~~  ***
***  p3 : _sqrtnotzero_  ***
1.1888756622102838 ~ sqrt(17/12)  (Δ = 0.0013624092135244847)
1.1888756622102838 ~ sqrt(7/5)  (Δ = 0.005659705590360664)
1.1888756622102838 ~ sqrt(10/7)  (Δ = 0.00635294712410972)
1.1888756622102838 ~ sqrt(18/13)  (Δ = 0.012178851381179578)
***  ~~~  ***
***  p4 : _gamma-rational_  ***
0.0002653800526189825 ~ 0/1  (Δ = 0.0002653800526189825)
0.0002653800526189825 ~ 1/19  (Δ = 0.05236619889474944)
0.0002653800526189825 ~ 1/18  (Δ = 0.05529017550293657)
0.0002653800526189825 ~ 1/17  (Δ = 0.058558149359145724)
***  ~~~  ***
***  p5 : _gamma-rational_  ***
0.5001719623633856 ~ 1/2  (Δ = 0.00017196236338556936)
0.5001719623633856 ~ 10/19  (Δ = 0.026143827110298612)
0.5001719623633856 ~ 9/19  (Δ = 0.026487751837069806)
0.5001719623633856 ~ 9/17  (Δ = 0.02923980234249679)
***  ~~~  ***
'''
 For each fingerprint, the code will write a custom function and store it in the folder Functions/ with a random string.
 The code then applies complex BFGS fits of the function to get the parameters.
 From this process we can deduce the 'best' fingerprint is ":first:max:P2:c:c^s:linear-gamma"
 The 'best' loss jumped from ~ 0.01 to 1e-6 which is a great fit (indicating 'exactness' + noise)
 The format of the fingerprint has a second order polynomial (P2), a constant c, a simple power c^s and a linear gamma term.
 We could write this as: phi(s) = (1 + p0*s + p1*s^2)* p2^2 * (p3^2)^s * Gamma( p4 + p5*s )
 The solution indicates: phi(s) = (1 + (-4/3)*s + (4/3)*s^2)* 0.47988^2 * (1.1888^2)^s * Gamma( 0 + s/2 )
 1.1888 is approximately the 4th root of 2, the current version of the code does not pick this up yet, but it would be simple to add.
 The figure "InterpretingResults.py" makes it clearer how this result is related to the functional form of the wavefunction.
 The actual solution is a Gaussian multipled by the 4th Hermite polynomial, and this is captured by the simple fingerprint.

Expected run time for demo on a "normal" desktop computer:

 Approximately 4 minutes
 Computer ~ 8GB RAM, 64bitx64 Intel Core i3-8300 @ 3.70 GHz
 The code is not optimised for speed at all

