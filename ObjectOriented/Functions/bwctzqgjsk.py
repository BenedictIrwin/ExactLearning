from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5):
  S = array([array([1.56486235+1.06661515e+00j, 1.22426516-6.59710838e-01j,
       2.18642504-9.90998443e-01j, 1.29028802-1.57648136e-01j,
       2.37543525+5.92785475e-01j, 2.96792159-3.50354982e-01j,
       1.33190968+1.65226842e+00j, 2.7591393 +2.71456104e+00j,
       2.88445498-4.08845929e-01j, 2.20855994+1.71014075e-01j,
       1.03401632+1.55322965e+00j, 1.06623399+5.11284187e-01j,
       2.95961534+2.82041033e+00j, 2.82309146+1.26595337e+00j,
       1.19667076+8.61620184e-02j, 2.66163819+1.75865777e+00j,
       2.64263474+2.11559644e+00j, 1.85562059+6.05347626e-01j,
       1.42504449+2.26429312e+00j, 1.90579715-1.80885376e-01j,
       2.28283791+1.27063002e+00j, 2.62230275+1.05594460e+00j,
       2.88768489+5.33059863e-01j, 2.48755621-1.71197860e+00j,
       1.78566358-2.15581085e+00j, 2.25655365-3.09295918e+00j,
       1.17178366-2.94409866e+00j, 2.22008324-1.09121768e-01j,
       1.59388197+2.68178351e+00j, 1.65145716-1.04494866e+00j,
       2.89288264+1.80334863e+00j, 1.83840368-2.11149187e+00j,
       1.68712522-5.26134947e-01j, 1.70818065-7.11428396e-01j,
       2.16842162+1.36139980e+00j, 1.35921961+1.68481503e+00j,
       2.27576897+6.62801958e-01j, 1.66253147-3.09689285e+00j,
       1.27429142+2.01751374e+00j, 2.55282974+3.28015267e-01j,
       1.14542021+1.73021014e+00j, 2.16522718+1.97175851e+00j,
       1.2261456 -1.82763020e+00j, 1.18969523-1.94146360e+00j,
       2.06163176+2.97859245e+00j, 1.04417301+1.50792174e+00j,
       2.84786744+1.95123958e+00j, 1.01794207-2.93234422e+00j,
       2.89707409-1.95363600e+00j, 1.26022185-1.96656062e+00j,
       1.28265469-1.01624524e+00j, 2.26552644+2.90734371e+00j,
       1.89239319+1.10109571e-01j, 2.98980113-2.73617069e+00j,
       2.38829531+2.77229885e+00j, 2.25417372+7.14411660e-01j,
       2.79976984-2.87088038e+00j, 1.71062921+4.74901327e-01j,
       2.99307992+2.70187545e+00j, 2.53137097+4.03330233e-01j,
       1.9816951 +1.15603845e+00j, 1.8734537 -1.27220854e+00j,
       2.37409184+2.81115296e+00j, 2.11463924+1.28173106e-01j,
       1.14887781+1.39673791e+00j, 2.06970656-8.32223410e-01j,
       2.28091317-2.67062995e+00j, 1.84334427+1.72674791e+00j,
       1.17068228+1.61689458e+00j, 1.66285359-2.82426447e+00j,
       2.73252528-2.10654235e+00j, 2.7428557 +6.13620017e-01j,
       2.87705971-2.85975905e+00j, 1.75421448-2.17874386e+00j,
       1.84998324-3.67633761e-01j, 1.95155101+4.38703559e-01j,
       1.99236706+6.67027796e-01j, 2.32454065-2.18747017e+00j,
       2.15216531-2.30578658e+00j, 2.62678853+2.95608651e+00j,
       2.6132667 +2.95467472e+00j, 2.55834439+2.29392495e+00j,
       2.2303574 +1.30919050e+00j, 2.15101874+2.57346593e+00j,
       2.7176602 +1.99530893e+00j, 1.0691075 +2.49540148e-01j,
       1.65469956+1.15031594e+00j, 2.18981418-2.35440620e+00j,
       1.70827376-3.45216550e-01j, 1.77587771+9.07316190e-01j,
       1.27864594+1.30743638e+00j, 2.88746141+1.65733632e-01j,
       1.8108083 +2.15896085e+00j, 2.62706776+9.56672019e-01j,
       1.57550126+1.18538395e+00j, 2.44738479-1.43412393e+00j,
       2.05072616-2.71158265e+00j, 2.36610289-2.61569645e+00j,
       1.5979679 -1.77765103e+00j, 1.72821554+9.09183629e-01j,
       2.17668734+1.48596521e-01j, 2.60559044-1.35496524e+00j,
       2.129263  +1.20796855e+00j, 2.6507899 -1.36269004e+00j,
       1.14846908+1.98813716e+00j, 1.82963837+6.22912788e-01j,
       1.3926984 +6.70397136e-01j, 1.60874251+3.05384790e+00j,
       1.40314779+1.96334619e+00j, 2.22086435+9.54996743e-01j,
       1.21350142+8.10039457e-01j, 1.29554193-1.72293817e+00j,
       2.48796446-2.75602684e+00j, 2.46061189-2.01644418e+00j,
       1.68241895+4.67212305e-01j, 1.96978707-1.10671706e+00j,
       1.62738323-2.35316185e+00j, 2.20943685-2.04308382e+00j,
       1.33997429-3.06827481e+00j, 2.11940882-2.75055208e+00j,
       1.2250052 -2.49075100e+00j, 2.0489045 +2.31552366e+00j,
       2.31808308+1.84255963e+00j, 1.55368098+1.44085133e+00j,
       1.23661203+1.06388145e+00j, 1.57890814+2.25222609e+00j,
       1.26895918+1.77395329e+00j, 2.7124118 -5.65617399e-01j,
       2.00472237-1.43338476e+00j, 1.111078  +2.96727965e+00j,
       2.43047049-1.92893008e-01j, 1.31090015+6.65325001e-01j,
       2.99632435-6.08766118e-01j, 2.10846783+1.59625377e+00j,
       1.79669798-2.17255597e+00j, 1.7083639 -1.44968007e+00j,
       2.5019759 +1.88221317e+00j, 1.56075545+5.55516776e-02j,
       2.44010053-1.13718187e+00j, 1.05683693-8.73030738e-01j,
       2.51415948-1.13524464e+00j, 1.12500125-1.03172850e+00j,
       1.66744173-2.91116610e-01j, 2.37723181-1.06134713e+00j,
       2.96344001+2.29174198e+00j, 1.55256349+1.85874327e+00j,
       1.69542668+6.54168299e-01j, 2.41406483+2.36299007e+00j,
       2.4618309 +2.42479052e+00j, 2.21768297-2.53724028e+00j,
       1.89473626+1.40416382e+00j, 1.96370287+3.22574188e-01j,
       2.48496501+2.04113957e+00j, 1.92245519-1.70709709e+00j,
       1.18481314+2.08759423e-02j, 2.32454438+2.22079577e+00j,
       1.64312737-2.11490582e+00j, 1.4816051 +1.46923200e+00j,
       2.4490955 -2.25203251e+00j, 1.8282677 +1.37499998e+00j,
       1.06460265-8.68132865e-01j, 1.64237754-1.51757565e+00j,
       1.23137388-1.68340780e-01j, 1.38702139+2.11586444e+00j,
       2.76452061+3.49286599e-01j, 2.05072019-2.55464964e+00j,
       2.06877045-7.29267249e-02j, 1.93312145-7.85223808e-03j,
       1.79502384-2.16687962e+00j, 2.75583352+4.95871635e-01j,
       2.73174033+4.62470115e-02j, 2.1997368 -1.00513233e+00j,
       1.6480371 +1.13498438e+00j, 2.68614723-4.37203747e-01j,
       2.51193784+1.55939995e+00j, 1.01844204+6.40623643e-01j,
       2.94333081-2.25679375e+00j, 2.2472055 +5.62628726e-01j,
       1.99312854-3.94590737e-01j, 2.8721154 +2.50434441e+00j,
       1.67714811+1.06569257e+00j, 2.28328346-7.18488845e-01j,
       1.84193003+1.19176895e+00j, 1.87194833+3.10601141e-01j,
       2.436612  -1.38580424e+00j, 2.11930614+6.55868946e-01j,
       2.54078064-2.99520297e+00j, 2.80919114-1.80830492e+00j,
       2.08322309-1.01067487e+00j, 1.33310882+2.07917614e+00j,
       1.5046551 +5.71116047e-01j, 2.54021469-1.54121864e+00j,
       1.73375276-2.82224017e+00j, 2.18197531-7.33512541e-01j,
       2.35127124-1.74113733e-01j, 1.86187497-1.32606424e+00j,
       2.04977707+6.20722254e-01j, 2.742344  +1.73629039e-01j,
       2.23855005-1.01770227e+00j, 2.01252087+1.08811496e+00j,
       2.18921413+6.47414286e-02j, 1.22329013-9.14878649e-01j,
       1.30501082-1.53213731e+00j, 2.66604271+1.02010569e+00j,
       2.05039611-2.94834504e+00j, 1.12105329-1.00309853e+00j,
       2.46499813+3.50331747e-01j, 2.28732737-1.83147332e+00j,
       1.82911552+2.38145283e+00j, 1.13818318+1.29682799e+00j,
       2.35660848-2.08128377e+00j, 2.91828703-1.27544775e+00j,
       1.68143729+5.57853496e-01j, 2.89375243-2.18786481e+00j,
       2.89747556+2.33311975e+00j, 2.71895684-2.95489535e-01j,
       2.02523106+2.89425412e+00j, 2.54614336-2.01800553e+00j,
       2.30556122+1.36030839e+00j, 1.67240147-1.02131777e+00j,
       2.36141002+1.76847945e+00j, 1.63103412-1.61833296e+00j,
       2.37075393-9.79988570e-01j, 2.1790323 -2.97822796e-01j,
       1.3844279 -9.80690938e-01j, 2.35521005+1.16893420e+00j,
       1.21715618-1.23209473e+00j, 2.14794659+1.64437305e+00j,
       2.62929883+2.75785498e+00j, 1.08133179+1.99568118e+00j,
       2.67963059-2.39232894e+00j, 1.02402535-5.10966613e-01j,
       2.06963841+3.05546755e+00j, 1.71321258-3.11144234e+00j,
       1.53192486-5.43215969e-01j, 2.40567755-2.70343089e+00j,
       2.90535173+4.04542861e-02j, 1.07721923-1.20338935e-01j,
       1.75277083-9.96237408e-01j, 2.22675216+2.27165258e+00j,
       1.56677036-2.57934301e+00j, 2.92214204-2.40963545e-01j,
       1.09090178-2.32690639e+00j, 1.13084971-2.78628753e+00j,
       2.49805394+8.78082583e-01j, 1.11723496-7.99793420e-01j,
       2.81232266+2.43812745e+00j, 2.22107645-2.66104820e+00j,
       1.32472974+2.23591034e+00j, 2.77301073-1.26321838e+00j,
       2.63361565-2.18505736e+00j, 1.88331615-2.56843477e+00j,
       2.49962586-4.67100034e-01j, 1.45527421+2.35644098e+00j,
       2.69201555+2.39509091e+00j, 1.49036729-2.43209106e-01j,
       2.52494052+1.12929400e+00j, 1.00516464+2.13041848e+00j,
       2.14421983+2.21954261e+00j, 1.47606014-1.17093961e-01j,
       2.68647303+7.65374090e-01j, 1.45285155-1.79851578e+00j,
       1.13843071-7.11929223e-01j, 1.98019931-2.39463289e+00j,
       1.20479779+2.17059000e+00j, 2.06725   -1.75485953e+00j,
       2.02737513-1.69460025e+00j, 2.86401593-1.21162481e+00j,
       1.46549263-2.30882364e-01j, 1.82149929+2.15217543e+00j,
       1.92306605+3.08431905e+00j, 1.61259936-2.01166254e+00j,
       2.22330229+1.93116128e+00j, 2.96013303+1.43186759e+00j,
       2.76578733-2.69896829e+00j, 1.88397208-2.38484727e-01j,
       2.57394606-2.43181252e+00j, 1.64397484+2.18157052e+00j,
       2.03882817+2.57526469e+00j, 2.4177415 +3.13136349e+00j,
       1.79048071+1.76216339e+00j, 1.88170615+2.09790764e+00j,
       1.78845475-7.53375362e-01j, 2.22753182+2.06755991e+00j,
       2.4113808 -6.24513692e-03j, 2.33502063+2.96228391e+00j,
       1.74707006+3.10067414e+00j, 1.82060473-1.09153550e+00j,
       1.12778347+5.10644889e-01j, 2.062662  -1.14661086e+00j,
       1.42816584+4.95216807e-01j, 2.68828825+9.41841676e-02j,
       1.0132729 +1.91211066e+00j, 1.52772823-1.87828628e+00j,
       2.91684517-2.38412789e-01j, 1.56295103+1.17898046e+00j,
       2.2546123 -2.64130827e+00j, 1.51687333+1.26976252e+00j,
       2.24895574+2.50869396e+00j, 2.88565642+5.87665288e-01j,
       1.57410242+1.89746159e-01j, 2.35370676-1.49817223e+00j,
       1.3492699 +1.86233548e+00j, 1.91871786-2.50075395e+00j,
       1.13135243-8.11192511e-02j, 1.07485524-2.19859852e-01j,
       2.09729665-6.99536658e-01j, 1.20434835-6.87586352e-01j,
       2.27511397+3.13623357e+00j, 2.87187007-4.46475646e-01j,
       2.18922142+9.35320570e-01j, 2.77739034-2.83627004e+00j,
       1.19894815-2.50361517e-01j, 1.65317867+1.56827945e+00j,
       1.7957946 +1.85853796e+00j, 2.11709234-2.20080528e+00j,
       2.47679454-8.10631742e-01j, 2.64781943+2.60196966e+00j,
       2.001985  -1.30241479e+00j, 2.5309616 -2.57841143e+00j,
       1.31756082+8.68056912e-01j, 1.16020484-2.32196364e+00j,
       2.32999597+2.10215714e+00j, 2.54294066-4.88389967e-01j,
       1.11521961-2.16462779e-01j, 2.28028346+9.53793350e-01j,
       2.20726633+2.23877461e+00j, 1.39232355+8.45876520e-01j,
       1.20441355-2.18476726e+00j, 2.33261795+6.43542560e-01j,
       1.08694274-6.23578967e-01j, 2.19198079-1.99591642e+00j,
       1.173694  +2.81767877e+00j, 2.7001353 +9.07140434e-01j,
       2.51803729+1.57570216e+00j, 1.44653581+1.69787877e-01j,
       1.62534435-1.56961711e-01j, 1.45976883-2.10440135e+00j,
       1.90546336-2.14767459e+00j, 1.35482137-2.74165796e+00j,
       2.76454052+1.45835790e+00j, 2.77813552+2.56240478e+00j,
       2.17330238-2.24492471e+00j, 1.76485029+1.58159992e+00j,
       1.01195789+2.01232036e+00j, 2.24819392+2.68783839e+00j,
       2.44673755-1.37903550e+00j, 1.14333659+2.11860141e+00j,
       2.23959798-1.16456193e+00j, 1.47236254-3.13747499e-01j,
       1.4099558 -1.74550871e+00j, 1.96442059+3.32433058e-01j,
       1.67518613-1.23964284e+00j, 2.259655  +1.67799974e+00j,
       1.76419285-2.57268317e+00j, 2.68822149-2.36130540e+00j,
       2.58809426-2.79085426e+00j, 2.02155699+3.07189406e-01j,
       2.35626122-3.48752984e-01j, 2.00914945+2.31309744e+00j,
       2.52786918+2.62357630e+00j, 2.20100732+1.06004511e+00j,
       2.37496362+2.90729061e-01j, 1.8371935 -2.02582131e+00j,
       1.53254317+1.80706272e+00j, 1.63647922-3.74364085e-01j,
       2.02013596-1.70675068e+00j, 1.7936464 +2.30410409e+00j,
       1.78468976+2.06659217e+00j, 2.51795964+3.53518575e-01j,
       2.15936463-2.32294539e+00j, 1.15170096+2.56477391e+00j,
       2.25146099+1.16729130e+00j, 2.61032681+2.91035406e+00j,
       2.60058088-2.83200000e+00j, 1.19763228-2.71883396e+00j,
       2.46674725+1.90831952e-02j, 1.55673588+1.45160052e+00j,
       1.41308791-2.98533506e+00j, 1.47974694+2.37293545e+00j,
       2.17833011-2.65949235e+00j, 1.48654933+2.02105922e+00j,
       1.20237696+2.60638355e+00j, 2.35521479+1.84313586e+00j,
       1.11449424+6.66800048e-01j, 1.60961191+4.80888676e-01j,
       2.90374397+1.51227345e+00j, 2.70284945+1.71181951e+00j,
       2.93145875+2.97188063e+00j, 1.73813431-2.42487192e+00j,
       2.2684041 +2.45311359e+00j, 2.16491351+1.81757565e+00j,
       1.03107863-7.85235321e-01j, 1.78349596+8.01275156e-01j,
       2.90653897+9.90738510e-01j, 1.97172082+6.67953697e-01j,
       2.27363179-2.12810523e+00j, 2.54248164-3.04746731e+00j,
       1.42753182-2.37484355e+00j, 1.48930843+1.88700206e+00j,
       2.20147104+2.47610918e+00j, 1.94482947-1.07786849e+00j,
       2.80319083+2.88626627e+00j, 1.26731996+2.09562595e+00j,
       1.21697605-7.18657689e-01j, 2.99023495-3.10227877e+00j,
       2.49824105+2.09306986e+00j, 2.03620473-1.12888004e+00j,
       1.05472761+1.49105800e+00j, 2.65881287-3.03145124e-01j,
       1.66137901+9.10348914e-01j, 2.28697482-6.13307778e-01j,
       1.6509408 +1.00077441e+00j, 2.99131468+3.05110001e+00j,
       1.49593422-2.10865442e+00j, 1.33680865+1.53769459e+00j,
       1.01219176+2.43661847e+00j, 1.63200153+2.33476637e+00j,
       2.76360131-2.94993971e-01j, 2.42187848+8.25937002e-01j,
       2.31393773-1.95764768e+00j, 1.66642406-7.31435269e-02j,
       1.6745775 -2.36937764e-01j, 1.37780645-2.75189384e+00j,
       1.30000443+9.65548029e-01j, 1.828477  +1.24695436e+00j,
       2.60004885-2.18976708e+00j, 2.466682  +1.81128521e+00j,
       2.88160934-2.45224459e+00j, 2.76496411-6.50526415e-01j,
       2.21030021+5.01116134e-01j, 2.51918638+1.51436113e+00j,
       1.19496292+2.34759313e+00j, 1.34102538-3.09055374e-01j,
       2.48011933+9.80643406e-01j, 1.98081261-2.61949827e+00j,
       2.83788752+2.94794663e+00j, 1.01782272+2.35821937e+00j,
       1.3754929 +2.82631154e+00j, 2.34224411-2.70561529e+00j,
       2.69468664+2.27485016e+00j, 2.09319492-2.77405221e+00j,
       1.09848966+1.09452785e+00j, 2.89673786+1.89662974e+00j,
       1.13705659-1.97411994e+00j, 2.13362259+4.63958208e-01j,
       1.68384158-2.65443352e+00j, 2.14951976-2.19104770e+00j,
       2.85680534-1.48528783e+00j, 2.34637152+2.81580372e+00j,
       1.02576605+5.79480815e-01j, 2.91563763-2.99034476e+00j,
       1.26196542+1.19382764e+00j, 1.25203217+2.03481742e+00j,
       2.28696874-2.70507500e+00j, 1.25637455-1.19879407e+00j,
       1.97574477+2.55966318e+00j, 2.94146278-2.75734096e+00j,
       2.23904814-2.04559879e-01j, 2.64429342-1.21364800e+00j,
       2.98492357+1.37174439e+00j, 2.23171272-2.11188424e+00j,
       1.08716359+2.43352253e+00j, 2.04758239+2.59930847e+00j,
       1.02810291-1.25456572e+00j, 2.26151199-1.52882834e+00j,
       2.5950306 +8.58130786e-01j, 2.19816239+1.65832921e+00j,
       1.45359671-7.28578184e-01j, 2.12357645-1.78927894e+00j,
       2.51083484+2.23021235e+00j, 2.89544989-1.89707171e+00j,
       2.71070223+2.10466832e+00j, 2.51611074-2.49245856e-01j,
       1.81191732+2.99518836e+00j, 1.06659458+3.10975478e+00j,
       1.3847644 -3.07136200e+00j, 1.90746769+1.10768661e+00j,
       2.72635087+2.80076945e+00j, 2.98691863+1.94273507e+00j,
       2.07546769-2.40295349e+00j, 1.41996143+9.20695399e-01j,
       2.07351897+2.68953590e+00j, 1.4171109 +2.60479002e-01j,
       1.26216115-1.85782333e+00j, 1.77371841-1.47171581e+00j,
       2.20046948+2.08846102e+00j, 2.64027504+1.30532408e+00j,
       2.35333926+8.63361254e-01j, 1.66567652+2.65267405e+00j,
       1.69520835+2.31337186e+00j, 1.41399297+4.69612813e-01j,
       1.79963936-3.48059462e-01j, 1.0574541 -1.09477157e+00j,
       2.57825222+2.03830991e+00j, 1.00065967+1.89150763e+00j,
       2.97929642+1.15440418e+00j, 2.81734218+4.49255966e-02j,
       1.44429435-3.37874489e-01j, 2.83918489+2.77687411e+00j,
       1.19848129+7.97940029e-01j, 2.49463117-1.48107182e+00j,
       1.34965403-2.36287438e+00j, 2.96968331+5.09595207e-01j,
       2.85474342+2.98461150e+00j, 1.17614017-2.16997917e+00j,
       2.08356589-1.11334048e+00j, 2.4553207 -2.47647216e+00j,
       2.5200448 -2.35231629e+00j, 2.17383317+1.12948263e+00j,
       1.30351743+3.68329166e-01j, 1.59839713-2.37396612e+00j,
       2.88140434-1.03074625e+00j, 2.95329376-1.47370111e+00j,
       1.44373033+4.30351558e-01j, 1.24956481-2.16012591e+00j,
       2.15956479-4.02731018e-01j, 2.41668288-6.00909985e-02j,
       1.9462089 -2.26558823e+00j, 2.28394699+7.92143697e-02j,
       2.54733553-2.43878263e+00j, 1.86031675+1.93066430e+00j,
       2.2801689 +2.61409645e+00j, 1.85254899-2.30352738e+00j,
       1.36180875+2.54427502e-02j, 2.04032409+1.86314351e+00j,
       2.90731709+2.28207509e+00j, 1.36742134-4.88566848e-01j,
       2.84570576+1.11512761e+00j, 2.40913772+2.79277787e+00j,
       2.09838074+1.80148216e+00j, 2.07191607+1.72982227e+00j,
       1.66916097+1.40467965e+00j, 2.79595091+2.72246124e+00j,
       2.89701628-1.68767811e-01j, 2.57178037+2.10855114e+00j,
       2.55955256-2.62889437e+00j, 2.32530003+1.35859377e+00j,
       2.04701901-6.79199794e-02j, 1.36033109-7.75366643e-01j,
       2.03916198-2.02487828e+00j, 1.76461737-1.92652107e+00j,
       1.22103117+8.07238229e-01j, 2.78105408+8.90882493e-02j,
       2.4520985 +1.87087818e+00j, 2.37037319-2.75800984e+00j,
       2.0997704 -1.73979764e+00j, 1.69266125-1.11325918e+00j,
       1.6133348 -2.61556168e+00j, 1.36568545+2.78339256e+00j,
       1.06513147+1.68502110e+00j, 1.91250782-2.12591667e+00j,
       1.60407005-3.70967258e-01j, 2.38743933+2.96014944e+00j,
       1.88460664-1.88479123e+00j, 2.2763024 +1.05984764e+00j,
       1.8757829 +3.10189840e+00j, 2.0522273 -3.09697920e+00j,
       2.50824987-1.87883782e+00j, 2.87146609+7.20074808e-01j,
       1.30057807-9.80586434e-01j, 1.62303834+2.72408651e+00j,
       1.9780317 +7.30739828e-01j, 1.36457872-1.22766340e+00j,
       2.08544845-8.65689732e-01j, 1.21781318-7.83828628e-01j,
       1.62387577+1.09536198e+00j, 2.69626655-1.32857144e+00j,
       1.42586281-2.64392353e+00j, 2.08562004+2.95270228e+00j,
       1.88011352-5.56016691e-02j, 1.67402549+1.97504103e+00j,
       1.67646916+2.24618721e+00j, 1.23167354+7.14547439e-01j,
       2.82828938+2.52385957e+00j, 1.49395203+1.95036393e+00j,
       1.08660404+2.12672291e+00j, 1.07452986+8.26247304e-01j,
       2.81205548+1.22964623e+00j, 1.4973479 -4.36964143e-01j,
       2.89124019-2.93277018e+00j, 2.37709177+2.97217418e+00j,
       1.38607518-2.34678466e+00j, 1.81111118-1.80276251e-01j,
       1.46950238-1.02492592e+00j, 1.77608954+2.61091060e+00j,
       2.37774214+2.04257707e+00j, 1.34362962+3.08099615e+00j,
       2.00677266+1.03517696e+00j, 1.58136566-2.06093782e+00j,
       1.19351697+5.21382601e-02j, 1.98175218+1.14039477e+00j,
       2.87392531-1.45010139e-01j, 2.21637155+3.00669288e+00j,
       2.92454332+2.66868472e+00j, 1.23352946-3.09466948e+00j,
       2.57614182-1.32843079e+00j, 2.12991831-2.59825366e+00j,
       2.99128825-2.41189460e+00j, 2.57321796+2.08571160e-01j,
       2.50342939+2.98295072e+00j, 1.3139784 +4.39645311e-01j,
       1.13270725-1.32551943e+00j, 1.40397619-3.08293519e+00j,
       2.2563283 -1.01640785e+00j, 1.18544358+1.30304987e+00j,
       2.88334319+2.21887532e+00j, 2.33628908+1.56200960e+00j,
       1.13435798+2.57907271e+00j, 1.44816946+2.33384853e+00j,
       1.72849345+5.65764784e-02j, 2.10502153-1.86561910e+00j,
       2.90748782-3.13955223e+00j, 1.49625544+6.43184646e-01j,
       2.26756464-1.53698285e+00j, 1.39698359-1.43459884e+00j,
       2.90546054-2.05628527e+00j, 1.32596562+3.20420170e-01j,
       1.70397591-2.01926910e+00j, 2.7831726 +9.25302706e-01j,
       1.42306207+2.27604752e+00j, 1.45318062-1.62922620e-01j,
       1.69398489-8.55396852e-01j, 1.13619039+1.17871359e+00j,
       1.97488601+2.90371210e+00j, 2.81955154+1.47472769e+00j,
       1.05191673-1.78448635e+00j, 1.26008994-1.24582779e+00j,
       1.13547052-2.74165333e+00j, 2.86645289+1.72183808e-03j,
       1.48877298+1.07393840e+00j, 1.61908398-1.31365581e+00j,
       1.21690212+5.14004084e-01j, 2.43408603+6.01535818e-01j,
       2.02274335+1.63841197e+00j, 1.40302661+2.35261739e+00j,
       1.73327911-2.64199052e+00j, 2.26309232-1.16962960e+00j,
       2.67927535-2.86329563e+00j, 1.29257064-2.07041641e+00j,
       1.0739665 +1.13656825e-01j, 1.91760784+1.17131941e+00j,
       2.13247422-1.17051895e+00j, 2.24221931-2.28117465e-01j,
       2.69028987-7.88948306e-01j, 2.60901765-2.86144013e+00j,
       2.83405602-8.85994200e-01j, 2.25804651+3.07323645e+00j,
       1.82341807+2.36189454e+00j, 2.25358312+2.20691150e+00j,
       1.82618152-1.54902762e+00j, 2.89009659+3.13093847e+00j,
       1.22090074-1.14994076e+00j, 2.04212276+2.69330535e+00j,
       2.62948664+1.01769607e+00j, 2.67558144-2.09446052e+00j,
       2.26958134-1.71216291e+00j, 2.69440173+7.90814698e-01j,
       1.1860092 +2.03406697e+00j, 1.60582142-2.86481034e+00j,
       2.1364336 +2.23109916e-01j, 1.26246198+5.08201511e-01j,
       1.14528939-8.04918527e-01j, 2.53815419+1.52410728e+00j,
       1.06507268-7.24233452e-01j, 2.37862509-2.99571768e+00j,
       2.20156204+2.19102029e+00j, 2.4582444 +1.22123325e+00j,
       2.59028631+7.30436695e-01j, 1.2857283 -2.22350537e+00j,
       2.15634553+1.84501464e+00j, 1.81561608+3.09290649e+00j,
       2.6149814 -2.01546258e+00j, 1.254603  -8.65067723e-01j,
       2.73539077-1.45920728e+00j, 1.55083071+2.54550412e+00j,
       1.91584644-2.04532819e+00j, 2.6928513 -1.31622341e+00j,
       1.95386644-1.36569982e+00j, 1.04764228-1.76979703e+00j,
       1.9242004 -6.36696562e-01j, 2.75026495+5.11999568e-01j,
       2.23181615+6.91123966e-02j, 1.8927909 +3.06709507e+00j,
       2.12484232+1.08647420e+00j, 1.39914732-1.28736750e+00j,
       1.53997124-8.95721676e-01j, 1.83679537+3.02917972e+00j,
       1.57431725+1.29827288e+00j, 2.48593824-1.95410398e+00j,
       2.7617898 +1.71102532e+00j, 2.87442578+2.70681804e+00j,
       1.93320703+2.50038456e+00j, 2.47105227+1.69747123e+00j,
       1.41833959-1.62465421e+00j, 2.8414329 +1.15148230e+00j,
       1.53610976-2.41479141e+00j, 2.65090951+2.77471106e-01j,
       2.65730455+2.97029173e+00j, 1.31582018-1.52695426e+00j,
       2.67532568+3.23981714e-01j, 1.16142832-2.62725533e-01j,
       2.24139091-2.81614281e+00j, 1.54979141+1.15290392e+00j,
       1.44701693-8.06628123e-01j, 1.55998966-1.23687476e-01j,
       1.26574116-3.00858573e+00j, 2.89992153-1.93507312e+00j,
       2.96068858-2.25071008e+00j, 2.83417774-2.57719907e+00j,
       2.44308312+4.48060028e-02j, 1.0673165 -2.44392220e+00j,
       1.67474382+1.59420420e+00j, 2.53228837+1.45639097e+00j,
       1.29733146-8.26521036e-01j, 2.8984329 +1.07959820e+00j,
       2.00487348+1.33408319e+00j, 2.95367372-1.25550225e+00j,
       2.99376619-1.58110339e+00j, 2.57160592+1.25114683e+00j,
       2.07115524-2.15718169e+00j, 1.01711654-1.69279745e+00j,
       2.64602484-3.09583381e+00j, 2.08768249-5.39782259e-01j,
       1.19686722-3.10276875e+00j, 2.13285629-1.15905118e+00j,
       1.36053695-1.03899769e+00j, 1.77402846-1.08826952e+00j,
       2.83518205+2.44628155e+00j, 2.44232038+2.56694563e+00j,
       1.02294123+2.33603207e+00j, 1.60925434+2.75578881e+00j,
       2.15478278+2.38177813e+00j, 2.14202393-1.35868145e+00j,
       2.06796019+1.26988100e+00j, 2.20180462-2.62159713e+00j,
       2.97770316-1.07345921e+00j, 1.79445985+2.03828917e-01j,
       1.36919401+5.69800514e-01j, 1.16174729+1.84465369e+00j,
       2.74432049+2.42811681e+00j, 2.96092568+1.20885193e+00j,
       1.96468457-1.53794519e+00j, 2.41621071-3.83080085e-02j,
       2.52129323+2.62410986e+00j, 2.16580422-1.05284253e+00j,
       1.9847096 +2.45499521e+00j, 2.42604328-7.14534920e-01j,
       1.59664641-2.48098650e+00j, 2.98817611-8.06018196e-01j,
       1.21425966-1.10228687e-01j, 1.57725556+2.41879310e+00j,
       2.88718348+4.42692175e-01j, 1.18649187-1.44648473e+00j,
       2.48334479+8.99928271e-01j, 1.13012159-5.95106410e-01j,
       2.24051085-2.56760736e+00j, 1.16097622-2.17404848e+00j,
       1.72348881-2.17936831e+00j, 1.68484495-2.03744286e+00j,
       2.69030839+5.68790753e-02j, 1.75788135-6.99247227e-01j,
       1.57883347+1.25871453e+00j, 1.99937228-1.73390010e+00j,
       2.56284898-8.77010925e-01j, 1.65633217+1.94888332e+00j,
       2.81452022-2.95040903e+00j, 1.59298561-1.07611284e+00j,
       2.82536437+1.10974931e+00j, 2.72572823-1.14327380e+00j,
       1.77385352+2.67860313e+00j, 1.27019304+3.08135138e+00j,
       1.91369834+1.34975705e+00j, 2.74399463+8.64313343e-01j,
       2.39384561-2.70237818e+00j, 1.6280077 +1.67550503e+00j,
       1.78712674-2.48819455e-01j, 1.45860354+1.21822223e+00j,
       1.52202268+7.66895717e-02j, 2.96660286-2.33801901e+00j,
       1.23976083-2.38004611e+00j, 2.96632604+9.09346319e-01j,
       2.58588427-2.39352636e+00j, 1.66074397+2.87335178e+00j,
       1.84731371+1.31518596e+00j, 1.56003689-1.88950423e+00j,
       1.58351808-5.43310751e-01j, 1.57408648-2.25277222e-01j,
       1.11321473-2.02275506e+00j, 2.48319419+1.66911134e+00j,
       1.38482319+5.17035567e-01j, 1.9145233 +2.77030066e+00j,
       1.53338986+1.06147698e-01j, 1.75021709+4.53081249e-01j,
       2.25195796-1.00555149e+00j, 2.11768072+2.75439358e+00j,
       1.42224978-8.45173361e-01j, 1.6428325 +2.29583316e+00j,
       2.73815383+1.76790041e+00j, 1.7892793 +1.30104713e+00j,
       2.68222064+2.52863364e+00j, 1.21775689-2.85632366e+00j,
       2.57665229-2.00515601e+00j, 2.19512294-2.10696924e+00j,
       1.33262817-1.28719297e+00j, 1.73117932+2.19271065e+00j,
       2.54724047-1.58456045e+00j, 2.21936304-1.73290267e+00j,
       2.9770926 -1.81871341e+00j, 2.95397245+2.38223689e+00j,
       1.86190812-5.86713524e-01j, 2.79310101+4.38074298e-01j,
       1.50230856+4.55970869e-01j, 2.9573726 +2.34169046e+00j,
       1.9173025 +5.99702213e-01j, 1.4188899 +2.70318117e+00j,
       1.3458046 +2.42430009e+00j, 2.13211917-1.78185440e+00j,
       2.91215721+2.75814321e-01j, 1.45115397+1.82532745e+00j,
       2.31093078+6.69218911e-01j, 2.96097601-1.90082733e+00j,
       1.22076488-2.95348722e-02j, 1.96939645-1.30320739e-01j,
       2.09344698+3.03327932e+00j, 1.96091736-1.04446306e-01j,
       1.4869815 -8.36609226e-01j, 1.72685217-2.24351625e+00j,
       2.75587205+1.45602357e+00j, 2.35824553-2.54357968e+00j,
       2.46258622-2.56188666e+00j, 2.76935088-2.75068861e-01j,
       1.57236391+2.93750973e+00j, 2.89949774-9.90017875e-02j,
       2.59328585+7.96964486e-01j, 2.96312963+2.16158049e+00j,
       1.39226532-2.23080130e+00j, 2.79950497+1.11221349e+00j,
       2.63103479-1.53941160e+00j, 2.99787711+5.38181580e-01j,
       1.36449006-2.26534623e+00j, 2.91604255-2.74168409e-01j,
       2.98167871-2.26683603e-01j, 2.07953109+2.41785884e+00j,
       1.32505305+1.95039275e+00j, 2.30243595+1.81565255e+00j,
       2.56571276+7.66696171e-01j, 1.74321191-2.14839782e+00j,
       1.948216  +4.01663509e-01j, 1.28363853+1.48938801e+00j,
       2.14079798+7.59383530e-01j, 1.05241321+1.00773470e+00j,
       1.11148602-2.82001036e+00j, 2.22920416-2.07500001e+00j,
       2.7467289 +4.80157193e-01j, 2.58343103-9.90961918e-01j,
       1.75561257+1.07086053e+00j, 1.60128315-1.61130640e+00j,
       2.91074658-7.80956001e-01j, 2.36288341-1.91566652e+00j,
       2.86820791+2.46710069e+00j, 2.65985782-2.10557351e+00j,
       2.89418808+2.11180279e+00j, 2.86394586+3.41138258e-01j,
       1.32833238+1.90825720e+00j, 1.81769494+3.03508068e+00j,
       2.89060959-2.65498885e+00j, 1.64919035-1.16964095e+00j,
       1.24531824+2.29104591e+00j, 2.08285383+1.05864951e+00j,
       2.02029036+2.21612623e+00j, 2.9484115 -9.52197207e-01j,
       1.31214608-2.41487733e-02j, 1.09888365-2.10848459e+00j,
       2.01648795-2.16469814e+00j, 1.73979168-2.47366404e+00j,
       1.77446009+1.92515338e+00j, 1.15487523-1.58161457e+00j,
       2.61955928+1.83510213e+00j, 1.55540201+2.22433301e+00j,
       2.40746382-2.40541420e+00j, 2.45647383-1.45700317e+00j,
       1.23416538-5.22485271e-01j, 1.20591337-1.59872542e+00j,
       2.07578094-1.08133518e-01j, 2.34648387+1.97840094e-01j,
       2.34339673+1.98398924e+00j, 1.15410344-2.50137259e+00j,
       2.93980692-6.06851873e-01j, 2.24065086-2.75082531e+00j,
       1.44246804+1.22246880e+00j, 2.14716976-1.99869880e+00j,
       2.94605864-2.06758947e+00j, 1.67311596+3.06472838e+00j,
       1.8412501 +1.49885368e-01j, 2.53576908+2.93904429e+00j,
       2.22256131+3.54732233e-02j, 2.20041319-2.96823522e+00j,
       2.82976813+1.57161729e+00j, 2.25805496+3.05380781e+00j,
       1.5085415 -1.86968638e+00j, 1.38173549-2.82724481e+00j,
       2.05931315+2.32336061e-01j, 2.91454488+1.46694660e+00j,
       2.75893235-2.07222050e+00j, 2.09771677+2.09406061e+00j,
       2.34676763-2.18333228e+00j, 1.29562189-7.00893328e-01j,
       1.98364091+1.93500993e+00j, 1.60195775-2.60360066e+00j,
       2.67922965+2.78933018e+00j, 2.91543662-2.45631653e+00j,
       1.44854305-2.18704371e-01j, 1.30770151-1.99711477e+00j,
       1.13883863+1.35238831e+00j, 1.53386176-2.80123517e+00j,
       1.59455622-8.12240777e-01j, 1.40818584+2.06871383e+00j,
       2.13638654+6.95919932e-01j, 1.75810397+7.64945771e-01j,
       1.65021153+1.58827197e+00j, 1.25887611-1.38596368e+00j,
       1.63476273+2.41981157e+00j, 2.31209579-1.19366818e+00j,
       2.83853947+6.18193762e-01j, 1.6747012 -1.06753011e-01j,
       2.73809915+2.46987238e+00j, 2.14715159+1.20146074e+00j,
       1.25886046-5.12054476e-01j, 1.43352651+2.56772724e+00j,
       1.09695039-2.81181001e+00j, 2.53790298-7.83954762e-01j,
       2.57167857-5.93298020e-01j, 2.47028254-3.08287135e-01j,
       2.13888865+2.55792884e+00j, 1.14679394-3.13058998e+00j,
       2.5269501 -2.89046175e+00j, 1.15582182+1.51021490e+00j,
       2.95686454+2.32208387e+00j, 1.37458787+9.71619962e-01j,
       1.03616492-1.27623424e+00j, 1.96476594+1.38531788e+00j,
       1.90209502+2.40612870e+00j, 2.25922698-2.89756254e+00j,
       1.40364705+2.04506423e+00j, 2.93714833+2.31301253e+00j,
       1.44447354-3.82133317e-01j, 1.85990261-2.34310537e-01j,
       2.0003344 +8.75592819e-01j, 2.73164623-2.43447202e+00j,
       1.77628044-2.58006601e+00j, 1.18811525-8.98961079e-01j,
       2.73172673-2.27197695e+00j, 2.30268405+1.91870201e+00j,
       2.49652364+1.71455838e+00j, 1.18116595+7.84181276e-02j,
       2.63022255+2.06682695e+00j, 1.63898951+1.83978227e-01j,
       1.50191508-1.64848116e+00j, 1.87742011+7.27893463e-01j,
       1.14632723+1.41071541e+00j, 2.62446267-1.77929448e+00j,
       1.34029077-9.79752984e-01j, 2.32378143+3.07815237e+00j,
       1.65861373-5.44510268e-01j, 2.78185657+1.78151261e+00j,
       2.38296322-2.18228876e+00j, 2.79489367+2.29413030e+00j,
       2.87002378+1.90571864e+00j, 2.98954316+1.51658956e+00j,
       2.77455501+1.89824268e+00j, 1.86753404+2.24289788e-01j,
       2.99896295-1.99533139e+00j, 2.92328687+1.01985265e+00j,
       2.37572633-1.98334489e+00j, 1.11015958-2.56847317e+00j,
       1.35574053-1.07407997e+00j, 2.56000479-2.34179128e+00j,
       1.60854325-2.52220913e+00j, 1.42398621+5.91963130e-01j,
       2.07161872+1.39397869e+00j, 1.19359249-9.80680842e-01j,
       2.63481931-2.36935075e+00j, 1.90407978+1.42755504e+00j,
       2.43644137-1.95269124e+00j, 1.34607323+9.85212062e-01j,
       1.74227585+7.87190803e-01j, 2.73889198-1.95211840e+00j,
       1.10839121+2.76395456e-01j, 2.10124629+2.05152910e+00j,
       2.65337663+6.86534002e-02j, 2.85735614+2.01476727e+00j,
       2.16149765+1.25381461e+00j, 2.03781514-5.90177120e-01j,
       1.72215312+2.66962091e+00j, 2.43667462+2.75496712e+00j,
       1.33399516-7.23412305e-02j, 1.18305573+1.48887490e+00j,
       1.66956923+2.82473536e-01j, 2.29892591+2.67360113e+00j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  ret += -loggamma(p4 + S*p5)
  return ret
fingerprint = frompyfunc(fp,6,1)
