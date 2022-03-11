from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2):
  S = array([array([1.58392194+1.17104671e+00j, 2.83651923+1.76533927e-01j,
       2.57662606-1.93112884e+00j, 2.72396634-1.89814542e+00j,
       2.29067069+9.57537676e-01j, 2.97859077-1.97880574e+00j,
       1.61323957-6.72754613e-01j, 2.92594976+8.38010438e-01j,
       1.44581304-2.67518274e+00j, 2.08633182-1.84907524e+00j,
       2.87889256+7.66691946e-01j, 1.61931387-1.31937195e+00j,
       2.7336936 -1.45173010e+00j, 1.6919216 -8.05111614e-01j,
       2.67275301-1.98214713e+00j, 2.40837191-1.48207233e+00j,
       1.259751  -3.73387584e-01j, 1.91291632-1.74085059e+00j,
       1.07925195-1.33412672e+00j, 1.30173189-1.09978034e-01j,
       2.89201723-2.25492159e+00j, 1.30389298+1.66822441e+00j,
       2.87377224+1.76172379e+00j, 2.47091327+7.16803026e-01j,
       1.62839993-2.91276336e+00j, 1.26602497-3.01668347e-01j,
       1.13164184-1.99665752e+00j, 1.74321488-2.60091431e+00j,
       2.74987397+3.12586438e+00j, 2.2179974 +2.28115425e+00j,
       1.73712051-1.72849718e+00j, 2.96064177-2.00755910e+00j,
       2.09597604+3.08127374e+00j, 1.84734176+2.07819381e+00j,
       1.08965096-1.44148679e+00j, 2.27467873-1.49366322e-01j,
       2.62642419-2.62139850e+00j, 2.34148564+8.00181200e-01j,
       2.76361421+2.96063308e+00j, 1.39288811-1.15511870e+00j,
       2.95706241+1.31237927e+00j, 1.10742347-2.31654489e+00j,
       2.41600802-3.49993060e-01j, 2.97231671-2.15806655e+00j,
       2.54671459+2.70585922e+00j, 1.41832273-2.95695517e+00j,
       1.93140927-3.04333660e-01j, 2.69489133-2.87257987e-01j,
       1.82900545-1.98162105e+00j, 1.5522987 -9.79445921e-01j,
       2.11017485-3.64921658e-01j, 1.77749721+1.31075484e+00j,
       1.68270503-1.29431918e+00j, 2.49772698+5.26017403e-01j,
       2.86448005-1.17666177e+00j, 2.92970617-1.84713477e+00j,
       1.14111589-2.92095268e+00j, 1.72477006-4.67088397e-01j,
       2.69548158+2.40584961e+00j, 1.30984067-2.43665897e+00j,
       1.38926432+1.26524536e+00j, 1.39332114-2.56523643e+00j,
       2.97660836-1.18506504e+00j, 2.31956691+6.47369943e-02j,
       1.89213628+3.03423665e+00j, 1.41019878+2.25558191e+00j,
       2.3855954 -3.83481061e-01j, 1.54206299-2.77294877e+00j,
       1.15990892-2.36090696e+00j, 1.55889352+2.17799684e+00j,
       1.81761756+2.34124065e+00j, 1.94196971+2.84748553e+00j,
       1.51218557-2.25852714e+00j, 2.62987693+8.14544281e-01j,
       2.36224057-2.80161530e+00j, 2.52207539+1.66308704e+00j,
       2.82768074+2.01547693e+00j, 2.37564528+7.43012761e-01j,
       2.69331811-7.82707196e-01j, 1.96218106-3.50489912e-02j,
       1.68714583-2.28025414e+00j, 1.74182512+2.55668632e+00j,
       2.55437276-5.61204808e-01j, 2.85482692+2.07129368e+00j,
       1.81726345+1.50397541e+00j, 2.9515079 -3.02545320e+00j,
       1.49867183-2.94345512e+00j, 1.262815  -2.19544557e+00j,
       1.79329101-2.46316368e+00j, 1.23013331+2.62233068e+00j,
       2.9853524 +2.27499571e+00j, 1.34111121-1.28237170e+00j,
       1.13212211-1.96366189e+00j, 2.8240095 -2.39269128e+00j,
       1.09173889-2.54910071e+00j, 2.04837656+2.07969827e+00j,
       1.96610851+1.05079592e+00j, 1.63190825+1.34903281e-01j,
       1.76014246-3.13056973e+00j, 2.97968969-1.25859110e+00j,
       2.65109906+2.28590151e+00j, 2.79624515-4.85944859e-01j,
       1.04623196+4.20648664e-01j, 2.18471985+1.02030841e+00j,
       2.57101801-1.39420170e+00j, 1.2889668 +9.03126621e-01j,
       2.72364567-2.20548428e+00j, 1.55002765-2.54555524e-01j,
       1.61937789-5.24692620e-02j, 2.29031251-2.90214591e-02j,
       1.07324949-2.82103124e-01j, 1.66894425+2.73055588e+00j,
       1.69483581+2.63600668e+00j, 2.08750893-2.36505442e+00j,
       2.78504507-2.77611716e-01j, 1.25685089+2.98800594e+00j,
       1.79170076+3.70043645e-01j, 2.64057133+2.15550663e+00j,
       2.41266199-1.97386696e+00j, 2.56019366-2.75622289e+00j,
       1.14353409+2.70842634e+00j, 1.70266934-3.04216134e+00j,
       2.65534411-1.93766300e+00j, 1.25142941+1.05283164e+00j,
       2.08811068+2.71186356e+00j, 2.14512918-1.88552337e+00j,
       1.91241474-1.67349261e+00j, 2.68702903+1.80066987e+00j,
       2.6649972 -1.54763395e+00j, 2.81350295-5.67702216e-02j,
       2.74038175+2.44577446e-01j, 1.68202851-2.19448777e+00j,
       2.41401272-1.57001965e+00j, 2.52703873+2.40765556e+00j,
       1.30548624-1.37449891e+00j, 2.60501614-1.03188314e+00j,
       2.48260354-1.40446608e+00j, 1.08705366+2.89344814e+00j,
       1.34464648+2.06344933e+00j, 1.20075328+2.84249502e+00j,
       2.49987671+6.40846511e-01j, 1.35920418-2.92595723e+00j,
       1.24419636+2.88132007e+00j, 1.29819106+1.78620328e+00j,
       2.08744996-3.02504202e+00j, 2.07913234+1.80964396e+00j,
       1.44243875+1.26335749e+00j, 1.8563458 +1.61762538e+00j,
       1.35686809+2.30121552e+00j, 2.66002904-2.19956161e+00j,
       2.36189176+3.38385319e-01j, 2.00831672-2.23511533e+00j,
       1.18918682+7.50341528e-01j, 2.65835768+2.67987290e-01j,
       2.0530164 -2.70150095e+00j, 2.25868856+6.97038900e-01j,
       1.2337485 +1.36529047e+00j, 1.70413673-5.62166368e-01j,
       1.11889847-1.87753265e+00j, 1.09673323-1.74276397e+00j,
       1.52503345-1.49937402e+00j, 1.01079683+1.65835303e+00j,
       2.95585985+3.00525321e+00j, 1.38554344+1.47479692e+00j,
       1.48665396+2.99839315e+00j, 1.02153472-1.88138091e+00j,
       2.88527106+1.12742180e-01j, 2.95307151-2.40387742e-01j,
       1.64807803-2.20359708e+00j, 1.63309012+6.76990686e-01j,
       2.86258973-3.05021380e+00j, 2.79729024-2.51857762e+00j,
       1.45480758+2.03742681e-01j, 1.45943335+2.50104934e+00j,
       2.70674238+4.57223617e-02j, 1.35649091+1.55591073e+00j,
       1.6989958 +2.01366915e-01j, 1.00294083-5.67630129e-01j,
       2.35952883+2.20985585e+00j, 1.99516668+2.91674489e+00j,
       2.1872139 +1.05067513e+00j, 1.72234778+1.29062924e+00j,
       2.14329473-2.45649952e-01j, 1.23949499-2.41688999e+00j,
       2.33381103-2.08460087e+00j, 2.86119277+1.50439406e+00j,
       2.01434277+2.26108045e+00j, 2.53821243-1.35285567e+00j,
       1.60716729-2.85522865e-01j, 2.28862169+1.49867647e+00j,
       2.43321856-8.95607184e-01j, 1.48314787+5.10837552e-02j,
       2.46353587-2.63602564e-01j, 1.05844216+1.90413650e+00j,
       1.06943094-8.57836821e-01j, 1.844229  -3.03268021e+00j,
       1.17311044+2.97115749e+00j, 1.02279123+1.33478463e-01j,
       1.17797525+1.44463397e+00j, 2.45376246-5.49890499e-01j,
       2.37078827+1.10742485e+00j, 2.86847999-4.44696748e-01j,
       2.27937177+2.47043444e+00j, 2.93461404-5.88174133e-01j,
       1.71381095+1.34076844e+00j, 1.12567381+2.03076357e+00j,
       2.912288  -1.30539457e-01j, 2.7790556 -1.28666308e+00j,
       2.63934192+7.03817459e-01j, 1.93744887-1.72249248e+00j,
       1.61007194+1.79388178e+00j, 2.57232634-9.89387370e-01j,
       2.09524756+2.47696267e-01j, 1.52212391+1.91433186e+00j,
       1.48968138-3.10106357e+00j, 2.98591423-1.66953645e+00j,
       1.13155674-7.62314511e-02j, 2.89238756+6.77897966e-01j,
       1.03274579-2.97206915e+00j, 2.08134044-6.68194280e-01j,
       1.47523079-4.16417699e-01j, 1.70846675-2.99198740e+00j,
       1.03443207+1.53476848e+00j, 2.03174229-2.18239260e+00j,
       1.82848017-3.74730232e-01j, 1.05085131-3.82571417e-01j,
       2.5260888 -4.61826977e-01j, 1.17881861-3.09958497e+00j,
       2.74752605-2.30783359e+00j, 1.19034784+8.95913087e-01j,
       2.53898073-3.56921187e-01j, 1.67760246+2.39613888e+00j,
       1.23512782-2.40317242e+00j, 1.94602132+1.93911047e-01j,
       1.95183412-1.98424772e+00j, 2.21895101+1.78834431e+00j,
       1.89539386-1.45453000e+00j, 2.05095988+1.21455904e+00j,
       2.47682807+3.89374230e-01j, 2.30780452+5.59796035e-01j,
       2.35814173+9.25525532e-01j, 1.6208868 -2.71780315e+00j,
       1.37467287-7.26051524e-01j, 1.50484563-3.72755813e-01j,
       2.69339412-2.88653316e+00j, 1.09178967+1.19942482e+00j,
       2.38023131-2.58998920e+00j, 2.25416445+9.03121150e-01j,
       1.23899137+1.28238519e+00j, 2.88260164+2.40160509e+00j,
       1.46631203-2.38990306e+00j, 1.78537775+1.97738843e-01j,
       2.89921936+1.57378684e+00j, 1.4677414 -2.32167604e+00j,
       2.86720627-3.10391872e+00j, 1.13849536-1.84341978e+00j,
       2.40320976-7.05289747e-01j, 1.12102471-4.45125394e-01j,
       1.44698942-1.67869148e+00j, 1.27420416-2.00900113e-03j,
       2.85167128+3.06241667e-01j, 1.18578837-1.59944910e-01j,
       2.61253001-2.89300093e+00j, 1.73626425-9.58882707e-01j,
       1.61605936+2.07627268e-01j, 1.11706438+9.25280511e-02j,
       1.04728588-2.24609104e+00j, 2.96953832-2.01072076e+00j,
       1.06653641-1.42712623e+00j, 2.90453456-5.14761842e-03j,
       2.13564481-1.88297501e+00j, 1.9741795 -3.11582050e+00j,
       1.89676177-8.39862742e-02j, 1.61548301-5.43649072e-01j,
       2.75249245-1.93340935e+00j, 2.68685958-2.75216425e+00j,
       2.73166839+3.47189272e-01j, 2.13214229-3.90058663e-01j,
       2.98562031+2.65329634e+00j, 1.24180012-2.72435960e+00j,
       1.44959657+3.07724983e+00j, 2.82355616-2.38178908e+00j,
       2.14010511+1.09154606e+00j, 2.36374105-2.72849766e+00j,
       1.9171483 -4.26982915e-01j, 2.3083098 +2.48897736e+00j,
       1.49537255+2.79886443e+00j, 2.71079596-2.38697733e+00j,
       2.45938874+2.63836088e+00j, 1.66321738-1.06720166e+00j,
       2.81437345+2.83128209e+00j, 2.90101351-2.79256617e+00j,
       2.8806143 -2.70378322e+00j, 2.98817812-1.61876080e+00j,
       1.88535016+1.90085456e+00j, 1.68872137-5.00069116e-01j,
       1.6160823 +1.73830604e+00j, 1.86733808+4.47313633e-01j,
       2.0831601 -1.06831410e+00j, 1.16417272-4.00632453e-01j,
       1.97886805-3.03009678e+00j, 1.59212581+1.15215985e+00j,
       2.24338034+3.03593443e+00j, 2.699407  +1.31805498e+00j,
       2.93531504+7.98080247e-01j, 2.56444377+3.06601987e+00j,
       1.6480833 -9.96279131e-01j, 1.44143519-2.33467298e+00j,
       2.23071239-9.29369641e-01j, 2.21942066-8.72845539e-01j,
       1.64442119+7.25994677e-01j, 2.15366184+9.98918289e-01j,
       1.82144102+5.83986043e-02j, 2.40325695+8.49532510e-01j,
       1.94346196+3.03744721e+00j, 1.59327287-1.15582192e+00j,
       1.09836101+2.42919395e+00j, 1.45847959-7.46549976e-01j,
       2.1669887 -1.21981449e+00j, 1.91728786-6.54484393e-01j,
       1.23785106-2.88858066e+00j, 1.49387756-2.67856563e+00j,
       2.78614769-2.52547931e+00j, 1.96638446-3.49134869e-01j,
       2.85690211-3.26882311e-01j, 1.11559846+1.78247913e+00j,
       2.86064591+2.63736726e+00j, 2.8501808 -2.77886368e+00j,
       2.13928075+1.87277477e-01j, 1.35848692-1.24755259e+00j,
       2.31918228-3.08395539e-02j, 1.70922023+9.31756526e-01j,
       1.64860979+1.78061568e+00j, 1.63967079+2.28288046e+00j,
       2.85314351+1.24056484e+00j, 1.2018921 +2.54195509e+00j,
       2.64410308-1.04214034e+00j, 2.7485051 -1.82371272e+00j,
       1.43497882-3.46280261e-01j, 2.84743018-4.20871700e-01j,
       2.29603751-7.42314169e-01j, 1.02134818-2.07730145e-04j,
       2.61011127+2.06990611e+00j, 2.20564717+2.34578669e+00j,
       1.34992011-1.06708160e+00j, 2.34135559+8.17524271e-01j,
       1.64508108+3.08050787e+00j, 2.93288069-5.75095864e-01j,
       1.78954733+5.65091651e-01j, 1.76541555+1.89076524e+00j,
       1.45910194+2.17794808e+00j, 1.69745647-2.16874189e+00j,
       1.68355   -1.14808826e+00j, 2.52204936+4.47181551e-01j,
       1.3829964 +1.78824216e+00j, 1.83337103+1.43154514e-01j,
       2.42904087-4.67837693e-01j, 1.83317694-2.95788118e+00j,
       2.95345432-2.58271793e+00j, 1.78547981-1.19426667e+00j,
       2.34062468+1.69079520e+00j, 1.87715033-9.88731972e-01j,
       2.45081288+1.09896559e+00j, 2.23823309+2.52806987e+00j,
       1.05569053-2.63542440e+00j, 1.27369403-7.17387994e-01j,
       1.73554769+7.05929784e-01j, 1.1204678 -7.12996910e-01j,
       1.19129919+1.30933781e+00j, 1.75003683-2.05676274e+00j,
       1.11573672-4.75104332e-01j, 1.03511507+2.23022862e+00j,
       1.21079452+7.91301354e-01j, 1.12553558+3.65495096e-01j,
       2.96011494+1.47545826e+00j, 2.36428128-1.13185206e+00j,
       2.13137075+9.82955707e-01j, 2.5663307 +2.61230458e+00j,
       1.06190452-5.36759586e-01j, 1.13228352+7.17828432e-01j,
       1.6037626 -6.15582819e-01j, 2.98343044-1.55879960e+00j,
       2.07295573-9.52854873e-01j, 1.46325876+7.43640734e-01j,
       1.29879159-1.44465364e+00j, 1.79131905-2.14169313e+00j,
       1.37445187-8.37656460e-01j, 2.11873086-1.07346488e+00j,
       1.35726412-2.30241873e+00j, 2.20217203+2.05121198e+00j,
       2.16480915+2.94550286e+00j, 2.15316439-8.15241006e-01j,
       1.27697325-6.40618437e-01j, 2.65377157+5.21467372e-01j,
       2.80104875+1.23520115e+00j, 2.50666858-2.98794778e+00j,
       1.64828942-7.27280761e-01j, 1.6010827 -2.66934384e+00j,
       1.90308628+2.48448955e-01j, 2.73462579-9.13659430e-01j,
       1.24044213+1.67382002e+00j, 2.28135638+2.26070451e+00j,
       2.23620085-2.82991467e+00j, 2.18745628-2.22879379e+00j,
       2.29151929+2.36517069e+00j, 2.98559262-2.52866481e+00j,
       2.87279816-4.99053740e-01j, 1.45242738-2.76386305e+00j,
       2.70322246-1.48128436e+00j, 1.70357387+8.48539295e-01j,
       2.1821224 +1.08855083e+00j, 1.91883113-2.39939179e+00j,
       2.72514523-4.15359926e-01j, 1.58761053-7.60177901e-01j,
       2.6648613 +1.85257491e+00j, 1.96695392+2.06561497e+00j,
       1.75214237+2.69881562e+00j, 2.55829529+2.19265229e-01j,
       1.99713079-5.52717274e-01j, 1.86458831+1.26189948e+00j,
       2.79926751+3.81990820e-01j, 2.32608659-1.94878884e+00j,
       2.43943421+2.72076321e+00j, 1.06939157-2.00505285e-01j,
       1.39080111-2.48068196e+00j, 2.91694727-1.69741304e+00j,
       2.90291257+1.46282063e+00j, 1.75814633+2.26014318e+00j,
       1.68742208-2.56405199e-01j, 1.37945509+2.85759567e+00j,
       2.01074686+2.55692433e+00j, 1.97491619-1.46654677e+00j,
       2.33809241-1.52352067e+00j, 1.52597729-9.79226167e-01j,
       1.41879588-2.56786071e+00j, 2.34073237+1.90343812e+00j,
       1.99944843-7.44061212e-02j, 1.23284112-2.74915913e+00j,
       2.36506989-2.31784740e+00j, 2.64512135-2.50817192e+00j,
       1.97931698-2.89656648e+00j, 1.52906336-3.50643392e-01j,
       1.44155272+2.88807889e+00j, 1.73744166+3.06679942e+00j,
       1.66404414+3.07334605e+00j, 1.39364614-2.09624593e+00j,
       1.66789594+2.15168352e+00j, 1.03309197+1.35193346e+00j,
       1.22953165+2.85537636e+00j, 2.3880021 +1.76140601e+00j,
       1.95393715+2.77578053e+00j, 1.71328558+2.10567893e+00j,
       1.98220002-1.56978989e+00j, 2.01581409+1.93838015e+00j,
       2.56190717-1.44209837e+00j, 1.5217534 +9.75478814e-01j,
       2.78634658+1.76688445e+00j, 1.1891378 -2.77534298e+00j,
       2.41947922+4.79369533e-01j, 1.1499059 -2.71656874e-01j,
       1.99100179+2.05701095e+00j, 2.15268772-1.44145219e+00j,
       2.25878382-2.12012548e+00j, 1.13668895-2.41242686e+00j,
       2.47171228+8.71093140e-01j, 1.76515273+1.16481940e+00j,
       2.70291106+2.74572093e-01j, 1.24296307-2.78288841e+00j,
       1.83223958-6.33282639e-01j, 2.99276351+2.32790856e+00j,
       2.51354034+2.07235155e+00j, 1.54427781+1.14200133e+00j,
       1.15291972-1.49882187e+00j, 2.83025714-1.85288049e+00j,
       2.89763941-1.03768892e+00j, 1.98153773+2.90620669e-01j,
       2.89363417-2.73832491e-01j, 1.5098896 +2.46625749e+00j,
       2.40636863+2.19200393e+00j, 2.23472375+1.04728887e+00j,
       1.9621416 +1.96912795e+00j, 1.3345802 -2.70552615e+00j,
       1.22575853+2.56142495e+00j, 1.00934687+2.90611049e-01j,
       1.42323408+1.02260103e+00j, 1.08473784-6.69431526e-01j,
       1.57202497-2.88947056e+00j, 2.7871273 +2.73451723e-01j,
       1.98794803+5.89550709e-01j, 1.85786116+1.96845832e+00j,
       1.67644798+7.72366839e-01j, 1.8312928 -2.23740636e+00j,
       2.19095412-1.86848899e+00j, 2.43386519-4.10192238e-01j,
       2.20801129-1.84738420e-01j, 1.4235125 -3.12897320e+00j,
       1.19802199+1.11100896e+00j, 1.74199059+1.79583826e+00j,
       1.89007969+4.16544002e-02j, 2.28854992+1.61243484e+00j,
       2.85789225-1.13424313e+00j, 2.19487552+8.23563332e-01j,
       2.02929875+2.56619930e+00j, 2.59995979-2.37104625e+00j,
       1.4935037 +2.87838149e+00j, 1.21696715+1.09845910e+00j,
       2.61062426+1.63158007e+00j, 1.01976678-2.93774518e-01j,
       1.48653639-3.10076210e+00j, 2.1820898 -6.23640131e-01j,
       2.1154208 +8.50402210e-01j, 1.09145244+2.68068099e+00j,
       1.01546324-2.69987490e+00j, 2.56562741+5.92169250e-01j,
       1.89267556-1.03213717e+00j, 1.7552948 -1.21655885e+00j,
       2.43646539+9.98625210e-01j, 1.94043305-2.26037872e+00j,
       1.41247106-1.64055608e-01j, 1.16543099-2.82937814e+00j,
       1.17577924-5.59159717e-01j, 2.94371212+9.04004445e-01j,
       1.4027665 +1.41547807e+00j, 1.56550754+3.20703871e-01j,
       1.52183307+6.95065703e-01j, 1.67221002-1.77687736e-01j,
       2.88266993+2.05057206e+00j, 1.17885268-2.74692248e+00j,
       2.50579691+2.69768334e+00j, 1.58881584-2.85739889e+00j,
       2.2538495 +3.10814757e+00j, 1.85640536+3.07416279e+00j,
       2.89462135+1.48751456e-01j, 2.6964316 +1.41436874e+00j,
       1.94493571+1.11474360e+00j, 1.15160676-7.78396659e-01j,
       1.74145554+2.19958736e+00j, 2.55167899-7.22805848e-01j,
       2.48967272-4.17083200e-01j, 1.5493122 -2.58971479e+00j,
       1.26581119-1.66134302e+00j, 2.63274531-2.20597405e+00j,
       1.19968758+1.56279756e+00j, 2.31279579-1.63736222e+00j,
       2.86733824+9.20037162e-01j, 2.65121763-1.11400580e+00j,
       1.29669878+1.27659966e+00j, 1.75546347-3.03369986e+00j,
       1.83408764-1.42662218e+00j, 1.96246117+1.31943921e+00j,
       2.15308972+2.59892945e-01j, 1.56707062-1.05527447e+00j,
       1.89551171+7.39538584e-01j, 2.41377856+1.85422940e+00j,
       2.26812809-3.05970378e+00j, 2.19637054-1.39030264e+00j,
       2.22907292+9.62722989e-01j, 2.04749392+1.65424214e+00j,
       1.72437256-2.99722352e+00j, 2.1813966 +9.33618106e-02j,
       1.95445642+1.61853177e+00j, 1.02251318-2.80501317e+00j,
       2.69375134+2.41922052e+00j, 1.11402079-4.19768885e-01j,
       2.51943578-1.29330061e+00j, 2.5213867 -2.33306613e+00j,
       2.93158454+9.20388817e-01j, 2.57168736+2.90985492e+00j,
       2.77687541-1.71971660e+00j, 1.85200937-4.25254182e-02j,
       1.72469819+2.11542471e+00j, 1.46934655+8.88788757e-02j,
       2.7285684 -5.89693136e-01j, 1.77118848-2.34903906e+00j,
       2.49662372-1.22028589e+00j, 1.16919425+2.22319201e+00j,
       1.4711526 +3.58306612e-01j, 1.61892201-1.10641799e+00j,
       1.11368055-1.69813406e+00j, 2.96936201-1.62238758e+00j,
       2.15239739-2.73676374e+00j, 1.70012208+2.68144002e+00j,
       2.58205048-4.85498926e-01j, 1.98175178-1.75741834e+00j,
       1.55993021+2.75732953e+00j, 1.6389468 +5.94282189e-01j,
       1.82975776+2.85254626e+00j, 1.28103716-2.60458167e+00j,
       1.10584361-1.31930874e+00j, 1.91284508+1.29151459e+00j,
       2.39824307+1.06327590e+00j, 1.99963363-2.18519707e+00j,
       2.08502079-1.18431024e-01j, 2.79851488+5.09151050e-01j,
       1.16808138-2.19211240e+00j, 2.56985186-2.34864921e+00j,
       1.71468177+1.81300847e+00j, 1.03308162+2.54636207e-01j,
       2.76333182-8.73400623e-01j, 1.13573092+1.57061762e+00j,
       1.91757772+4.83946611e-01j, 2.54044254-2.41274141e+00j,
       1.87676926-9.28381130e-02j, 2.24310827-1.41312227e+00j,
       1.28612988+3.01593235e+00j, 2.10129301+4.75641005e-01j,
       1.48108696-1.91034552e+00j, 2.48426577-2.52619673e+00j,
       2.61603293+1.47755742e-01j, 1.30133663-1.98339075e+00j,
       1.97040189-1.91226971e+00j, 1.27806107-2.34731330e+00j,
       1.16927541+2.61401534e+00j, 2.49674477-1.32626750e+00j,
       1.27867569-3.02567938e+00j, 1.04593737+2.71100932e+00j,
       1.4510799 -9.16113860e-01j, 1.85014071-2.27505810e-01j,
       1.15960561-2.77959345e+00j, 2.27956327-1.20174648e-01j,
       1.39995132+1.53862927e-01j, 2.93615898+2.07454589e+00j,
       2.08709797-2.50705587e+00j, 2.62412167+2.36039488e+00j,
       2.51230477-1.54625733e+00j, 2.58403304-1.62178865e+00j,
       2.99417933-2.28599406e-02j, 1.91334351+8.41322825e-01j,
       1.81131816-2.00932974e+00j, 2.17104999+7.34000646e-01j,
       2.86127103-2.08130198e+00j, 1.00617806-1.14261101e+00j,
       1.19856922+1.82039945e+00j, 1.52733749+2.80077365e+00j,
       1.74068698+1.03809111e+00j, 1.61540949-4.11847977e-01j,
       1.2354356 -1.86590003e+00j, 1.23027196+3.62210547e-01j,
       2.59955679+2.43309452e+00j, 2.56975996-1.67731969e+00j,
       1.10289802-1.95533906e-01j, 2.8990181 +9.96253498e-02j,
       1.8365828 -1.10287184e+00j, 2.814831  +2.17235118e+00j,
       2.38679002+2.71326367e+00j, 1.04214736-7.11722891e-01j,
       1.74117248-6.71832686e-01j, 2.14295174-1.52776325e+00j,
       1.73069257-4.42779292e-01j, 1.13790658-2.19410245e+00j,
       2.40331409-2.19134396e+00j, 2.49768405-3.79164696e-01j,
       2.83917288-2.68973026e+00j, 2.91277498-4.37718389e-01j,
       2.27718547-2.67678729e+00j, 2.59989858+1.41827227e+00j,
       1.02565206-2.08770597e+00j, 2.29695352+2.20985237e+00j,
       1.14949712-7.38178050e-01j, 2.26360091-8.81018615e-01j,
       1.43211429+1.89688468e+00j, 2.23916274+2.70359140e+00j,
       2.02911445+1.95764332e-01j, 2.1719608 -1.25171663e+00j,
       1.19756137+5.18412330e-01j, 1.75274065+1.62506111e+00j,
       2.50509324+2.20371730e-01j, 2.32472271-2.33273159e+00j,
       2.38131112-7.94137301e-01j, 1.09368196-1.87898048e+00j,
       1.25166108+2.13995511e+00j, 2.89874775+3.05696984e+00j,
       1.6477465 +2.50164920e+00j, 2.39036984-1.86603117e+00j,
       1.12615224-1.51761573e+00j, 1.78893111-2.38685593e+00j,
       2.57434015+1.16780131e-01j, 2.92268152-2.20616087e+00j,
       2.70068928+1.00780820e-01j, 1.1645032 +1.52638386e+00j,
       2.0072934 -1.16398169e+00j, 2.49965473-1.42287324e+00j,
       1.09294022+2.08617870e+00j, 2.08143176+1.14618979e+00j,
       1.96629775-2.43490727e+00j, 2.52683449+1.39559575e+00j,
       2.14283051+1.49938930e+00j, 1.09704668-1.21351724e+00j,
       2.92246612-3.00895430e+00j, 1.87818038+2.93791020e-02j,
       2.28896256+2.26041757e+00j, 2.93984772-7.86834127e-02j,
       2.26432103+2.96953361e+00j, 2.89158231-3.00101166e+00j,
       1.96095121-1.54069881e+00j, 2.64706262-7.29552588e-01j,
       2.2582005 +7.73258979e-01j, 1.078009  -3.11366920e+00j,
       1.00697729-2.01430172e+00j, 2.80395341-3.02700277e+00j,
       1.18501945+7.19900872e-01j, 2.49017893+1.57843752e+00j,
       2.77755088-1.23016448e+00j, 2.62395068-2.33013430e-01j,
       2.63508285-2.65284180e+00j, 1.16917054-1.43459791e+00j,
       2.91033258-6.65679965e-02j, 1.54124216-8.29714203e-02j,
       2.74707745+2.87132673e-02j, 2.3010591 +2.53414717e+00j,
       2.86386338-6.44708366e-01j, 2.16282113+5.89533504e-01j,
       1.81403826-1.97960480e+00j, 2.52903489-1.11422143e+00j,
       2.27899145+1.61609460e+00j, 1.90941182-1.42628088e+00j,
       2.38711711+2.74079629e-01j, 1.92467764+2.83246917e+00j,
       1.92269441-1.89736012e+00j, 2.98074287+2.43110493e+00j,
       1.07511383+1.00372358e+00j, 2.25615973+1.02637951e+00j,
       1.05888754+7.77757330e-01j, 2.30998705+3.81486459e-01j,
       2.1655325 -4.02583167e-01j, 1.63601023-2.83262237e-01j,
       1.94499789-1.70296097e+00j, 1.29872218+8.51188144e-01j,
       2.01532181+1.24134610e+00j, 2.40365633+2.49138268e+00j,
       1.82061073-3.67134719e-01j, 2.06753464-5.41870390e-01j,
       2.12372642+1.91096420e+00j, 1.84205228-3.08561075e+00j,
       2.3397648 +6.48737349e-01j, 2.77706557+2.61656329e-01j,
       2.50111035+1.66462795e+00j, 1.47436382+2.48066642e+00j,
       1.08109768+1.45738431e+00j, 1.1543034 +8.15824598e-01j,
       2.70909555-3.07761690e+00j, 2.89723423-7.41241051e-01j,
       1.8279037 +3.08148820e+00j, 1.23962882-1.61374228e+00j,
       1.8439971 +7.01733384e-01j, 1.03029175+8.89219534e-01j,
       2.70822199-9.16095733e-01j, 2.60877588+1.07212727e-01j,
       1.37953535+1.04833443e+00j, 1.45916829+2.53150760e+00j,
       1.21420768-2.10605375e+00j, 1.22535753+7.48383816e-01j,
       2.39836692+1.49274967e+00j, 1.43710664+3.03372530e+00j,
       2.24388687+2.49824045e+00j, 2.81177221-9.15908055e-01j,
       1.15343849-2.67621750e+00j, 2.81826676+3.00417569e+00j,
       1.36511067-9.31531301e-01j, 2.0182285 -3.21149077e-01j,
       2.35104717-9.66994656e-01j, 2.03174118-2.94446360e+00j,
       2.08075503-4.35410783e-01j, 2.04311551+4.48067322e-01j,
       1.67610336-3.83275549e-01j, 2.23486175+2.10808131e+00j,
       2.9089426 -1.68759386e-02j, 2.38022181+1.42162214e+00j,
       1.50939746+5.82664591e-01j, 2.81057124+2.02219607e+00j,
       2.61787569-4.84022090e-01j, 1.69682065-5.27108142e-01j,
       1.95537938+2.87263900e+00j, 2.80874764+1.02021365e+00j,
       1.12850178+1.73884854e+00j, 1.4156326 -1.60401369e+00j,
       1.74506269-2.32236313e-01j, 2.37566877-1.59572930e+00j,
       2.4944111 -1.57834366e+00j, 2.83064438-2.85034185e+00j,
       1.44279691-1.79996644e+00j, 1.42552659+2.38089726e+00j,
       1.97300624-1.97904329e+00j, 1.54209033+2.65440594e+00j,
       2.54057168+2.78641028e+00j, 2.84503861-1.03518116e+00j,
       1.05290794+2.53377123e+00j, 2.75905415+4.48185515e-01j,
       2.05669848-3.05289670e+00j, 2.44884441+5.57292002e-01j,
       2.50486327+6.12793967e-01j, 2.42932946+2.68342518e+00j,
       2.99550941+2.62131935e+00j, 1.42982736-2.51283879e+00j,
       1.06708023+6.06252929e-01j, 2.32908597+7.87846758e-01j,
       2.01684407+3.01502911e+00j, 1.62326302+1.04982626e+00j,
       1.71612668-1.69197516e+00j, 1.10089543-1.26070465e+00j,
       2.53406981-1.21168155e+00j, 2.65512391-7.00230486e-01j,
       1.24343317-6.30179276e-01j, 2.10590232+7.76255549e-01j,
       1.68459891+5.60786983e-01j, 2.94479073+2.80165846e+00j,
       1.54617674+2.92339943e-01j, 2.74161799-2.58226534e+00j,
       1.99348324-2.32158337e+00j, 1.77338908-4.98381119e-01j,
       1.33208638+3.25574422e-01j, 1.52171763-2.83887614e+00j,
       2.55338571-2.26190174e+00j, 2.60878171-9.29420272e-01j,
       1.18107224+1.62016270e+00j, 1.34523015-3.14401270e-01j,
       2.7771499 +1.11986599e-01j, 2.53404641-3.86111395e-01j,
       1.26333634+1.55862355e+00j, 2.62957593+3.08196131e+00j,
       2.33995361-2.22443799e+00j, 2.59631814+1.70186185e+00j,
       1.64617141+1.38404079e+00j, 1.22269454+7.46125942e-01j,
       1.02639837-2.29572445e+00j, 2.02477565+8.71556724e-02j,
       2.32705696+1.20474752e+00j, 2.54464871-2.86344789e+00j,
       1.45169247+1.67009341e+00j, 1.12836499+1.25670041e+00j,
       2.97595894+3.13179521e+00j, 2.68617743+1.86853951e+00j,
       1.02137014+2.90756071e+00j, 1.79592582+5.24885911e-01j,
       2.02935197+2.86891049e-01j, 2.20464818-1.27112530e+00j,
       2.05426678+7.81762876e-01j, 1.53289079+2.08796838e+00j,
       1.48105189+8.40961319e-01j, 1.55959163+3.00828269e+00j,
       2.39841932-1.82204278e+00j, 2.67679088+1.58518329e-01j,
       2.68558372-5.08908749e-01j, 2.37756565+1.49312306e+00j,
       2.95978779+1.24753283e+00j, 1.98109141+1.39091031e+00j,
       1.57611416-7.73713441e-01j, 1.78434402-2.15963571e+00j,
       1.52538909-1.64657487e+00j, 1.2939378 -2.42860239e+00j,
       2.86336615-2.71759461e+00j, 2.46975988+1.12572892e+00j,
       2.89545486-1.45883932e+00j, 2.74353815-8.73547934e-01j,
       1.13972694-1.08013466e+00j, 1.73773892+1.58283859e+00j,
       1.0879161 +3.11840200e+00j, 1.87351718-2.10387317e+00j,
       1.72544927-7.42448134e-01j, 1.54403637+5.11178260e-01j,
       1.53387856-2.24801980e+00j, 2.43109958-1.18039380e+00j,
       2.30795644+1.53581188e+00j, 1.55794146+2.48566502e+00j,
       2.28849646+1.32456112e+00j, 1.27706841-3.10933563e+00j,
       2.50735823-2.41468788e+00j, 2.31351096+9.47685661e-01j,
       2.43857427-2.81224094e+00j, 1.94379408-2.67661446e+00j,
       2.17430847+2.44359254e+00j, 1.58266346-1.18491837e+00j,
       1.3951971 -2.87947163e+00j, 1.01814163-1.18801492e+00j,
       1.36436519+3.00524961e+00j, 2.82483251+2.10868388e+00j,
       1.92930192+6.21291589e-02j, 1.63973236+6.95444755e-02j,
       2.30767104+8.20826124e-01j, 2.41637443-8.69777867e-01j,
       2.21899326-7.54171765e-01j, 2.88607673-1.66919411e+00j,
       1.78228325-6.64560020e-01j, 1.0852877 -2.19350415e+00j,
       1.87159298+3.42407297e-01j, 2.58858732+9.48295334e-01j,
       1.77624278+2.63768323e+00j, 1.87286659-2.03360586e+00j,
       1.23811014+2.70487383e-02j, 1.89128087+2.02933554e+00j,
       2.31026096+2.33964924e+00j, 1.892338  -5.45167366e-01j,
       1.99602935-1.82822315e+00j, 2.61796177+2.39720455e-01j,
       2.80940197-1.81649342e+00j, 2.0833299 -1.06338766e+00j,
       1.4996351 +2.61288057e+00j, 1.44613208-1.59671121e-01j,
       1.76018011+1.82154300e-02j, 1.16160093+2.38841593e-01j,
       1.38671231+2.54073435e+00j, 2.27534604+2.39822450e+00j,
       1.03801639-3.56838878e-01j, 2.94425912-1.63692411e+00j,
       1.44297836-3.27266161e-02j, 2.80394307-1.70664243e+00j,
       2.52236313+2.51044890e+00j, 2.56133185-2.22676685e+00j,
       1.88344488-2.60915279e+00j, 1.90954271+4.08944054e-01j,
       2.44758653-1.80314973e+00j, 1.75514929-8.20639698e-01j,
       2.81793291-5.37866077e-01j, 2.39712838+2.32253507e+00j,
       2.01691922-2.66748525e+00j, 2.81736482+3.08518490e+00j,
       2.79079092-1.84613198e-01j, 1.01770855-9.42365131e-01j,
       1.64529713+2.75526667e+00j, 2.92845656-9.58451472e-01j,
       2.61592794-1.89684831e+00j, 1.20904669+1.84797516e+00j,
       2.51524705-1.16003486e+00j, 1.42567994-4.74770531e-01j,
       1.34177566-1.61287993e+00j, 1.54432415-2.99656440e+00j,
       1.16923662+1.41446986e+00j, 2.62558186-5.14397164e-01j,
       2.16536183+1.34262420e+00j, 1.32704161-1.67701007e+00j,
       2.85769508+8.83308947e-01j, 1.03715349-6.46982470e-03j,
       2.34762197-1.59674134e+00j, 2.63618827+9.75325575e-01j,
       1.12286263+2.22217602e+00j, 1.96925947-1.00858922e+00j,
       1.28587475-2.37986591e+00j, 1.54687344-2.86858868e+00j,
       2.09891887+2.33791625e+00j, 1.67128524+2.78175512e+00j,
       1.85805871+1.59877450e+00j, 2.00180885-2.91797797e-01j,
       1.99335176-4.31284253e-01j, 2.53540612+9.06900641e-01j,
       2.67990078-2.08705442e+00j, 1.48030372-2.76405728e+00j,
       1.05217869+6.94853712e-01j, 2.99167187-1.77796783e+00j,
       2.34723589-2.62835094e-01j, 2.75585173+1.49112070e+00j,
       1.8167535 +2.49303841e+00j, 1.30562428+1.46227243e+00j,
       1.22100615+2.23094790e+00j, 2.92739683+2.18453518e+00j,
       2.6707167 +2.97681782e+00j, 1.48269024+1.65353110e+00j,
       2.64171434-1.19438098e+00j, 1.67110222+5.67330426e-01j,
       2.48354444-3.33761479e-01j, 2.89848016+1.96331384e+00j,
       1.96602939+2.76053449e+00j, 1.1966801 -7.94705178e-01j,
       2.62108618-1.34390823e+00j, 2.44839738-2.38197552e+00j,
       2.52352664+8.91801176e-02j, 1.97969785-1.03816790e+00j,
       1.60721074-1.09538494e+00j, 2.6583535 -2.83974568e+00j,
       2.64780934+8.14920536e-01j, 2.80321828-1.61427263e+00j,
       2.72517874-2.98779754e+00j, 1.4616163 +2.92504688e+00j,
       1.86928095+2.40465162e+00j, 2.12379196-9.77294268e-01j,
       1.28703779+2.28853334e+00j, 2.86371821+9.99198056e-01j,
       2.96986417-1.92333595e+00j, 2.21065639+1.17046540e+00j,
       2.72086707-2.39051941e-01j, 1.67910631+9.19786417e-01j,
       1.84554062+1.78491887e+00j, 2.12234524+9.02729611e-01j,
       1.40787216-2.18737610e-01j, 2.50644042+2.21351252e+00j,
       1.22365768+2.81518470e+00j, 2.99830634-2.93360225e+00j,
       2.88164786+2.39528831e+00j, 2.5018971 +1.65106659e+00j,
       2.45175445-5.31678422e-01j, 1.36807405-1.91493475e+00j,
       1.55391329-1.75149570e+00j, 2.44584342-1.29065640e+00j,
       1.83966914+1.86288370e+00j, 2.16752325-1.16823064e+00j,
       1.17584536-1.18725536e+00j, 2.4424904 +1.42935958e+00j,
       1.72729781+2.18926549e+00j, 2.79755465-1.81516898e+00j,
       1.47092503-1.93045642e+00j, 2.14916353+1.24792403e+00j])])
  ret = 0
  ret += S*log(p0**2)
  ret += loggamma(p1 + S*p2)
  return ret
fingerprint = frompyfunc(fp,3,1)
