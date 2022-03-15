from numpy import log
from scipy.special import loggamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10):
  S = array([array([4.34081614+3.00954758j, 2.75990473+1.56845765j,
       3.00066185-2.74851427j, 2.73542323-3.02329337j,
       1.52023299-0.20602289j, 3.5874881 +2.44508186j,
       1.83738685-1.87576265j, 3.05616959+3.03496602j,
       4.54951574-0.26930783j, 4.81900989+1.35833444j,
       1.0448088 -0.76813679j, 2.31194071-0.70055585j,
       2.03666522+2.18072403j, 2.38485919-1.96609575j,
       2.29993909+0.97826578j, 3.59259635+0.23401512j,
       1.00312831+1.90914324j, 3.65852167-2.72851125j,
       2.10509798-1.10014965j, 4.9677765 +2.158307j  ,
       2.32053446+1.90114386j, 1.67815177+1.51906804j,
       2.76807363+1.54710134j, 1.70802958+0.73910981j,
       3.04899685-2.60011152j, 3.41297051+1.82556069j,
       3.38013129+2.09358345j, 2.35473917-0.78814437j,
       4.99885684-2.80812681j, 3.42775332-2.09787855j,
       4.29127677-0.32444394j, 2.31235529-0.56965704j,
       3.41201681-0.32564405j, 3.39198549+0.94795371j,
       4.6838022 -2.04231418j, 1.64672595-2.9875769j ,
       1.27511718-0.37247121j, 1.68494824+1.55948327j,
       3.83147851-1.99916907j, 1.99320314-0.87516443j,
       2.48242055+1.479509j  , 2.5680467 -2.11078251j,
       4.48127204-1.62351317j, 1.10012905+1.65455137j,
       3.10422369+0.41036387j, 4.43460179+0.59698845j,
       1.84991727-1.06893646j, 3.75693998-0.36419687j,
       2.8732386 -1.14623434j, 4.68846537-2.16533155j,
       2.68429854+2.90788322j, 2.8833473 +1.20930246j,
       1.89661063-2.27320474j, 3.85640502+1.40757877j,
       3.61581333-0.14673294j, 1.94700208+1.02255139j,
       4.12176761+0.79083838j, 4.73682104-2.47467706j,
       4.81055583-0.89750703j, 2.16716304+1.93581879j,
       2.20020097+2.91187322j, 3.07466209-2.23422787j,
       3.77429308-0.0692043j , 3.32339947-3.10552062j,
       4.14752498-2.31132563j, 3.85987971+0.22324739j,
       4.36064185+0.21168745j, 3.7540282 -2.96348285j,
       1.22156264+2.15140929j, 1.53731537+2.0903246j ,
       2.14269049-1.07247522j, 1.96013073+0.17411097j,
       4.49859629+1.44369852j, 1.78198649+1.24176091j,
       4.43313521+2.96981874j, 4.9913932 +1.47503118j,
       4.78653341-2.68006026j, 2.25606604+2.22574127j,
       1.02030591-0.41227577j, 2.48517436-0.20383457j,
       2.40917329-0.63300935j, 1.07710229-3.09311224j,
       2.46323074+2.03431312j, 4.4756372 -0.46544082j,
       1.69562107-2.74536972j, 1.52443242+1.26976896j,
       3.26901089-2.69587213j, 3.13408119-2.07373663j,
       1.48940173-0.44034537j, 2.88150607-2.69371582j,
       4.04239789+2.62540952j, 3.60373327+0.98638277j,
       4.31903527-2.91345452j, 4.37829939-2.99361431j,
       4.15421357+2.33836242j, 3.70404392-2.33255152j,
       3.37149668+0.13769451j, 3.33837222-1.31321078j,
       2.72637711-1.01030964j, 3.36976899+1.03512183j])])
  ret = 0
  ret += log(p0**2)
  ret += loggamma(p1 + S*p2)
  ret += loggamma(p3 + S*p4)
  ret += -loggamma(p5 + S*p6)
  ret += -loggamma(p7 + S*p8)
  ret += -loggamma(p9 + S*p10)
  return ret
fingerprint = frompyfunc(fp,11,1)
