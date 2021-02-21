### For the demo/walkthrough pertaining to the most recent journal submission, please see the contents of "ExactLearning.zip"

# ExactLearning
Algorithms to exactly learn analytic functions given numerical data

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
