from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9):
  S = array([(2.557491038204092-2.9248359877964347j), (5.734039879653268-2.365973298886631j), (6.108202124314285+1.8252399786991296j), (6.966706781524118+2.3534889122583706j), (1.698610243117424+1.5850264212913023j), (4.577375597273967+2.571113936023239j), (6.480011723318894-0.9016447938408367j), (3.374014875083755+1.1558673901447145j), (4.004843802606397-0.8875662456439559j), (4.755944373666443-0.352102705709727j), (1.3084121988094113+2.413795767794544j), (6.455613576752189+1.6353889927657699j), (3.941474170076278-0.33465107583720455j), (6.409977270157995+1.2066549193264793j), (6.877002835294688-0.1316706240090637j), (5.242988238510523-1.2048222474091599j), (6.864049676678199-0.49495124040206884j), (4.013150685779181-1.4931364366954663j), (1.1174474845957245+0.376894961979644j), (1.2531632673101396+2.369507350502257j), (2.6220882446948592+1.6071302449676939j), (6.938696563865837-0.6232919497527596j), (5.749778042392507+0.7716190074136016j), (5.920561705779466+2.7781511051413377j), (4.105224898768121-2.893518260514249j), (5.412348668053124-2.6005871970381085j), (6.861041074137134+2.762436782876934j), (5.4681544547050365+1.7289860457628086j), (1.8106170191134352-0.05439465541293664j), (3.443597436814377-2.277064673522803j), (3.4323626567097296+3.133199555391748j), (1.9266942752398233+2.6071439463983417j), (1.347494632909521-1.2350257421349482j), (3.087315601659842-2.8715578204735412j), (3.986136202896981+2.275400782030877j), (5.018239357018432-0.027163143199800288j), (2.620199641040754-0.751324730779392j), (6.953222453495315-0.23505984378210787j), (6.741076885933204+1.5512898643345432j), (2.3096967421002663-1.0506929348718006j), (1.7774807495435894+0.7193021382590055j), (5.558193169982411-0.7090114252499693j), (5.0427168331974315+1.1825381848177123j), (1.6236904481520287-2.8345110402984557j), (5.100611568192534-1.3416289116029967j), (2.0512641514302334-0.5411899225397478j), (5.463116572562834-0.6419783970137805j), (6.738035092940477-0.5369930837153247j), (2.4010844802942457+2.625491491026928j), (4.383719452494724-2.7121302652268477j), (6.632385963266597+2.9020765104846973j), (5.93579594804417-1.445236016227109j), (1.1856851497872827-0.8807295684636123j), (3.2234431092954807+1.2251970227621163j), (5.868787132596572+1.275775692639808j), (6.233461372268218-2.0292966819070237j), (2.618937642899008-0.9596813190776889j), (3.0581133241310403+2.9529211300732356j), (3.9862730491783798-0.18894696785281484j), (6.710261900881511-0.5117174716405803j), (1.1057277743067901+0.11806006064130203j), (6.413565525896139+0.3279522550684475j), (5.737761217486623+1.568613886535731j), (3.921827498717617+0.9361686141798611j), (5.820389845377041+0.9428978277736038j), (6.0230141440825715-2.0774691880140264j), (3.5086604304632942+1.057907078385595j), (1.8666358586317084-0.11498062079199167j), (6.775760213713424-2.086758073494427j), (3.2072200325849303+0.7695654291791918j), (5.002776414428554-0.2997993333045055j), (3.3222237909539376+1.000579424792166j), (2.154487850429616-2.4299744012292015j), (2.114936063097133-0.42668156338229934j), (6.86782758421956-1.0650135161409389j), (1.2033632151222837+1.787498962694153j), (4.961983817443151+0.7179784067904595j), (6.196015497211432+2.1986829277054065j), (6.993841685200251+0.7047950211962375j), (6.501201926857264-0.029424006101267164j), (4.781566328466174-0.4074361741872643j), (1.9348837554501064-1.7095591861347221j), (6.12546301544469-2.378744791594906j), (3.477641790312827-2.4602754116446266j), (6.354488324147056-0.5055964069632308j), (1.240789477442694-0.229137747037075j), (4.104354159331578-0.13155790232178743j), (3.3341219620103164+1.513981549114682j), (3.943489297485311-1.6925355089896161j), (1.9776946243383318+0.09907087632301259j), (3.2012322143458736+0.20814357541250095j), (1.7603507934174023-0.0582357925306467j), (6.037034409264605+2.9996821356086727j), (5.317110353048+1.7026005840171754j), (6.568824041502411+1.324982663211654j), (1.4764319592480761-3.0874122023626014j), (5.4122814197320475-0.2650751768649524j), (1.3829353067277665+1.253836440573422j), (4.204058079267264-1.848756078804178j), (2.8579599280526415+1.9943335103056272j), (1.7534115739371496+2.738714955607752j), (4.6449819318453756+0.5890244373891007j), (3.004458749144239+2.209019434098577j), (4.575959727513414-0.11675507672401997j), (1.6348591934998078+2.301895289785657j), (3.11490380171913+2.424661778109539j), (6.60460927145099+1.177442342537045j), (5.306614181416143-2.718072595599324j), (6.742723813195164-3.1123105489134293j), (5.9933096658251195+0.46180374211752273j), (3.9984607366493004-0.5338849324634269j), (5.142355618096105+1.9772116645110236j), (5.86027067448345-2.913332866041341j), (6.22754515234939+3.0205264502602596j), (3.7355230212103594-2.276480821061449j), (2.2631163643482424+1.2557554959076542j), (4.644554188061419-2.9941685012829056j), (1.2439117954937549+1.26567175315982j), (4.815128467139967-2.77765691381171j), (2.597664751438602+0.8345486397570663j), (5.526256496002826-1.244847669268259j), (4.700889510239849-0.412770153740488j), (3.1704543723240026-3.0316390836066587j), (4.633546925401324+0.07758865064103526j), (6.179436090311606-0.9885295818468558j), (5.632930094953769-1.5154858470165387j), (6.777984147944592-2.6831609119171436j), (2.737344785691196+1.248666416357489j), (1.3800098290585305-2.2470089749421853j), (6.769598673008186-1.9156485561924352j), (5.27216711720491+0.5720508367252664j), (6.446863472328365-2.5843589093132135j), (1.173575015487548-0.4275941953479263j), (5.385820419633355+0.3766427827336263j), (4.371326277745911+0.2459026162841642j), (6.292685922970247+0.22935434298985502j), (1.144367081109675+2.4598162659616643j), (4.21145377927594+2.6338625278767056j), (3.7322403189392075+0.2335482710405734j), (3.1184232311990416-1.0283082036405609j), (2.7959117192886938-3.091315978194137j), (6.602009235977618-2.479399746484522j), (2.713748026663729+0.9496058149921085j), (4.398306980487136+1.2588338884803267j), (3.9280641892418773-2.1915067475137695j), (5.269804952462474+2.381999043631569j), (5.49738225778079-3.0547051381862267j), (1.4218813072210508-0.4224316718983654j), (3.948734434711819-1.0217561236307233j), (1.4317205474976165-0.7780868573882915j)])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  ret += loggamma(p4 + S*p5)
  ret += -loggamma(p6 + S*p7)
  ret += -loggamma(p8 + S*p9)
  return ret
fingerprint = frompyfunc(fp,10,1)
