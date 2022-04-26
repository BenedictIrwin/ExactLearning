
from scipy.special import digamma
from numpy import array, frompyfunc
def fp(p0,p1,p2):
  S = array([array([2.37522415, 0.80144837]), array([0.15352805, 0.01713704]), array([4.74791379, 5.87222602]), array([ 3.66372182, -3.45216508]), array([ 3.96086308, -1.96612213]), array([3.51748793, 2.51034206]), array([1.11471918, 0.60911261]), array([2.75659647, 5.20600556]), array([ 4.63925035, -2.09292374]), array([2.23739255, 0.91496378]), array([ 0.20556454, -0.12005631]), array([2.52592778, 0.77015849]), array([0.2573065 , 0.02164477]), array([ 0.98913827, -0.11575556]), array([ 3.2032927 , -0.12132303]), array([0.00584091, 0.00380782]), array([0.71305988, 0.49785472]), array([4.62060149, 6.91804601]), array([2.1543113 , 2.80291157]), array([2.77827593, 4.83755804]), array([ 1.77274624, -1.49620284]), array([ 2.6827075 , -2.56192593]), array([ 0.64555001, -0.37447579]), array([4.65435403, 3.13729315]), array([ 2.82082225, -1.35792017]), array([ 4.25128099, -1.2772911 ]), array([ 2.44998945, -0.07435878]), array([1.21613745, 1.66664377]), array([1.39833334, 2.58605731]), array([ 4.438104  , -3.80077003]), array([1.10849274, 1.30766079]), array([ 0.42489938, -0.15097965]), array([1.34517733, 2.24941159]), array([2.87471676, 5.22584343]), array([4.3125318 , 0.74966592]), array([ 3.95032059, -2.13710509]), array([3.59179715, 5.71613606]), array([2.77229747, 3.00433624]), array([ 0.08262614, -0.04756521]), array([1.35340733, 1.96233732]), array([2.1459267 , 0.96418001]), array([3.43400363, 3.89320606]), array([3.75197151, 3.04327615]), array([ 1.3151922 , -0.76497841]), array([ 4.0253571 , -3.04506309]), array([4.04205225, 3.47599384]), array([0.16021185, 0.09814265]), array([4.55824856, 1.75220323]), array([ 0.96544622, -0.42920094]), array([ 1.87434862, -1.71249595]), array([3.96283337, 7.61377783]), array([1.337749 , 0.9740586]), array([ 3.94293541, -2.96975607]), array([3.34165483, 3.51043099]), array([3.08634885, 5.17577388]), array([4.22944371, 3.35216243]), array([ 2.59945709, -0.33526164]), array([1.83359974, 3.16030146]), array([ 4.69053567, -0.08301815]), array([3.18079586, 6.2226699 ]), array([3.00722205, 1.79770641]), array([1.68455884, 1.34893458]), array([1.14917252, 1.92074128]), array([1.82199668, 0.21059074]), array([ 2.51181575, -2.2640155 ]), array([ 2.60369938, -0.45904323]), array([ 2.92986315, -0.28043309]), array([ 2.4394466 , -1.61739683]), array([0.63508584, 0.52739407]), array([0.08637826, 0.09927948]), array([ 1.35109737, -0.49734921]), array([0.4860863 , 0.15524892]), array([1.36827771, 0.53519847]), array([1.43569247, 0.21606643]), array([ 3.40276337, -1.11768284]), array([1.79104262, 1.97106722]), array([2.41308524, 4.24215539]), array([ 4.24872805, -1.2137617 ]), array([0.12316959, 0.13975304]), array([1.79033978, 0.60864483]), array([4.32638023, 0.17886402]), array([3.91549429, 1.72485958]), array([ 0.47082618, -0.26741296]), array([4.60963687, 6.00639852]), array([0.69894023, 0.84539892]), array([ 4.10482869, -0.18992484]), array([4.54818564, 5.44616078]), array([ 0.05237129, -0.04149337]), array([4.26322696, 5.22230104]), array([3.20317122, 3.7262162 ]), array([0.12789476, 0.2280752 ]), array([ 4.79684583, -2.51363678]), array([1.10004886, 0.35101088]), array([2.05503232, 3.11207164]), array([2.40423657, 2.30896986]), array([0.83890784, 0.51745245]), array([ 3.22387488, -3.06984836]), array([0.86067191, 0.6574774 ]), array([ 2.59132152, -0.11302828]), array([3.65340956, 4.01520652])])
  ret = 0
  ret += 0
  ret += digamma(p1 + S)
  ret += -digamma(p2 + S)
  return ret
logderivative = frompyfunc(fp,3,1)
