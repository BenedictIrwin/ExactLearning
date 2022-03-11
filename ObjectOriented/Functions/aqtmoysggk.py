from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5):
  S = array([array([2.95081119-2.94904001j, 1.35343424-1.29176705j,
       2.53278732+2.27706069j, 2.04602038+2.72783623j,
       2.39849116-3.05213413j, 1.31123052-2.50343707j,
       2.5873944 +2.15262049j, 1.42671483-2.42159639j,
       1.80198041-2.83886466j, 1.58671866-3.08957963j,
       2.4307311 +1.15600471j, 2.6542738 +1.28672655j,
       2.24854043-2.36921773j, 2.38188507-2.82661053j,
       1.90696006-0.42256229j, 2.84803098+0.0365873j ,
       2.21579385+0.06132676j, 2.20386191-0.15655767j,
       2.36899945-2.1067171j , 1.13015051-0.31032728j,
       2.30448346-0.07730548j, 2.10699804+1.97133913j,
       1.71689362-0.72707507j, 1.02618585+0.09937371j,
       2.60061708+2.94826699j, 1.08230447+1.88114927j,
       1.59943594-2.63687092j, 2.9375726 +0.72048652j,
       2.59596605-1.48881913j, 1.35946536-2.16699618j,
       2.97250861+1.68195203j, 2.74250244-1.09944105j,
       2.69726069+0.74313795j, 1.46682041+2.45379106j,
       1.42611406+0.23215721j, 2.34659283-1.00826879j,
       1.77984765+0.16695977j, 1.88305125-1.05104043j,
       2.60744836+1.32069244j, 2.37700841+1.36977471j,
       2.63785055-2.16862299j, 2.07605616-0.11163193j,
       2.15698693-0.23425754j, 1.68227967+0.68590026j,
       1.18352578-2.37646562j, 1.9468539 +2.9423585j ,
       1.60318657+0.84327152j, 2.53623745-2.73895096j,
       2.92586908+0.76263967j, 2.00480703-2.47903608j,
       1.75386299+3.12479216j, 2.32470597-0.09595049j,
       1.9158189 -2.19637263j, 1.76894108-2.77605656j,
       2.10836914+1.55963545j, 2.0667119 +1.09658054j,
       2.40907386-1.3402775j , 1.38314178+0.7813236j ,
       1.86292997+1.18959124j, 2.03227381-0.06187703j,
       2.49990789+2.88244164j, 2.36798448+2.78371663j,
       2.88820302+1.15705136j, 2.07884464+1.22607873j,
       1.9674871 -1.22343611j, 1.78625598-1.74715641j,
       2.27727328-0.59441238j, 2.29461336+1.39747943j,
       2.59136203+0.2829006j , 1.80516396+1.2457308j ,
       2.52619375+0.7426189j , 2.48342292-2.31505588j,
       2.19288954-0.69883254j, 1.99805218-0.30478958j,
       2.61925281-1.78937825j, 2.22261707+2.51281363j,
       1.44362711+0.30458121j, 1.82763103-0.28665j   ,
       1.91574877+2.14429088j, 2.59713797+0.03586762j,
       2.351015  -2.13345594j, 1.10589825+1.72297313j,
       1.03258913+1.16152575j, 1.44354232-2.90143748j,
       1.38281257-2.92605841j, 1.28770071+0.81674066j,
       2.20243946+1.54597967j, 2.00408055+2.0997913j ,
       2.05517923+2.69638536j, 1.36461136-2.43093525j,
       1.85611825-1.87102309j, 2.06843069+2.39594997j,
       1.15355629-1.21726563j, 2.32181324-0.17440361j,
       1.30391627-0.63669072j, 2.12813147+0.90451995j,
       1.72009091-2.94767093j, 1.80860261+0.29494024j,
       2.94546476+2.33249534j, 1.54673195-0.91486827j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  ret += -loggamma(p4 + S*p5)
  return ret
fingerprint = frompyfunc(fp,6,1)
