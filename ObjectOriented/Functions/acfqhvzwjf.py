from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([array([4.06327421-2.34620647e+00j, 3.55983986+2.23275967e+00j,
       1.55416205+9.16439511e-01j, 4.06027922+3.03651385e+00j,
       3.08142366-2.54479228e+00j, 1.47739926-1.19616722e+00j,
       3.19345799-9.47863144e-01j, 1.31622645+2.28797642e-01j,
       1.73129296-7.44289623e-01j, 2.18203274-1.03567801e+00j,
       1.97908794+6.95330896e-01j, 1.44831961-7.31225802e-01j,
       3.41923782-3.20455510e-01j, 4.72388748+2.89095681e+00j,
       1.59569408-2.63548847e+00j, 1.30594826-1.01024128e+00j,
       2.03956121-1.93776111e+00j, 1.03407118-2.58685290e+00j,
       3.18997489+8.59330620e-01j, 4.54020027+2.71603666e+00j,
       3.25630178-8.62324991e-01j, 4.7143184 +1.66713867e+00j,
       3.23860801-2.88883385e+00j, 2.36238946-6.95036012e-01j,
       3.25848254+9.27917917e-01j, 2.23151145+1.74942941e+00j,
       2.96871022+4.43640391e-01j, 1.21399244-1.12602319e+00j,
       1.03455881+3.12212064e+00j, 3.1657771 -5.55207459e-01j,
       2.47425216-1.54281217e+00j, 2.43977025+2.31445743e+00j,
       1.84200448-2.56401702e+00j, 3.33371523+3.11568496e-01j,
       1.81907499+8.25758285e-01j, 1.24659618-5.78976080e-01j,
       1.42875443-1.30625355e+00j, 4.5847906 +1.96669491e+00j,
       3.65946463-1.11756901e+00j, 4.28667914-1.36084985e+00j,
       2.5071842 -2.02973590e+00j, 3.06949356+5.23001360e-01j,
       1.91069901+1.18765380e+00j, 1.89201007-2.32470365e+00j,
       3.32197456-2.45787802e+00j, 1.01828548-2.46796857e+00j,
       2.74871615-1.13294610e+00j, 3.48152087-2.73964889e+00j,
       3.98742649-2.91636473e+00j, 1.61292936+1.40762333e+00j,
       1.35256686+1.04047739e+00j, 3.95534301-9.16419886e-01j,
       3.66928562+7.80881593e-01j, 2.17807435+6.63428697e-01j,
       2.883536  -2.86698383e+00j, 1.16998533-5.09249494e-01j,
       2.54786705-8.67420753e-01j, 1.07514284+2.37348414e+00j,
       1.15349875-1.33000963e+00j, 2.31791281-2.56006794e+00j,
       2.66378281+2.76457712e+00j, 3.73190996-1.08588834e+00j,
       4.39843501-6.39386345e-01j, 2.42465591-3.07868343e+00j,
       4.24657499+2.75217841e-01j, 1.50041534-1.49589596e+00j,
       4.95490077-9.63000322e-01j, 1.07218845-4.48023216e-01j,
       1.11715038-2.37347504e+00j, 4.73576994+6.12973133e-01j,
       3.8334993 -1.85249944e-01j, 3.74868303+5.78940156e-01j,
       4.0311214 +5.27213677e-02j, 3.18896308+5.48698944e-01j,
       2.75873458+2.56045722e+00j, 4.39559719-2.11218045e-01j,
       1.50621739-1.05714304e+00j, 1.7547927 -1.04827732e+00j,
       1.76169665-7.45979573e-01j, 2.2463947 +9.10239564e-01j,
       4.87234947-2.89628470e+00j, 4.92145104-2.54901725e+00j,
       3.20554482+2.79969723e+00j, 1.00083998-1.97483782e+00j,
       2.67355068+2.83879329e+00j, 2.31513673-5.20711316e-01j,
       2.8859997 +7.10816611e-01j, 3.54457019-2.98156417e+00j,
       3.20637123+2.49613400e+00j, 2.73131723+1.67786657e+00j,
       4.16797618-2.98120519e+00j, 2.21478694-2.28651886e+00j,
       1.22276802+2.36227785e+00j, 2.55484091-2.93511447e+00j,
       3.28594154-2.11440535e+00j, 3.95372802-1.40499278e+00j,
       2.19871197-3.10514618e+00j, 1.64734823-1.67455374e+00j,
       2.94064903-2.49184543e+00j, 2.99671326+1.17607083e+00j,
       4.96717467+6.31395363e-01j, 4.66197562-1.64974517e+00j,
       1.63171212+1.67286595e+00j, 1.84917573-1.56297007e+00j,
       2.99264774+6.65015530e-01j, 2.77667688+2.54479061e+00j,
       4.29152533+2.21390437e+00j, 2.19754853+1.09476369e+00j,
       3.64820582-8.38944740e-02j, 2.53159825+3.02430981e-03j,
       1.87100521-2.60486922e+00j, 4.31536878+2.83162366e+00j,
       1.16810764+2.47857686e+00j, 3.76809618-1.50958095e+00j,
       1.29847311-1.89710796e+00j, 3.74133813+6.61018583e-02j,
       1.37650985+5.10970391e-01j, 4.54037877+6.86669484e-01j,
       2.65040048+1.22105455e+00j, 2.38774455+8.45479168e-01j,
       2.38718629-1.18400884e+00j, 1.46753925+7.31287471e-01j,
       1.10531545+4.46156266e-01j, 3.7779359 +3.15741843e-01j,
       4.46668039-2.49891300e+00j, 3.17277339-2.90863321e+00j,
       1.64549016+1.04241364e+00j, 1.29975263+1.59461583e+00j,
       2.85344615+1.05957126e+00j, 4.03414116-2.70252647e+00j,
       4.90086257+1.51369731e+00j, 4.52440508-1.73399907e+00j,
       2.62248181-8.07519235e-03j, 1.1608243 -3.08449916e+00j,
       4.84164086-3.84990488e-02j, 3.2219247 -4.82759741e-01j,
       2.89411288+1.47608781e-01j, 2.47481914+2.55735889e+00j,
       1.73148405+2.13202802e+00j, 4.95482776-2.62372210e+00j,
       1.48840887-3.13674124e+00j, 3.52685959+1.13776072e+00j,
       2.12942904-5.58249669e-01j, 2.39062199-6.06084618e-01j,
       2.76068418-1.99821921e+00j, 4.92125771-9.92141649e-01j,
       3.47389956+8.14995651e-02j, 2.48639694+2.17869956e+00j,
       4.04540915-2.94277343e+00j, 2.01096761-2.39016702e+00j,
       4.46151805+1.91777407e+00j, 3.52824242-2.92328670e+00j,
       1.31368548-1.45253975e+00j, 2.13229363-1.84482141e+00j,
       1.67105827-6.75575428e-01j, 3.79067567+9.21256126e-01j,
       3.97072838+2.55217152e+00j, 4.24519848-1.82197776e+00j,
       1.826524  -1.26856058e-01j, 4.2624748 -2.06625431e+00j,
       1.79610313-7.85145936e-01j, 1.40630946+2.93141288e+00j,
       2.41875379-2.70778544e+00j, 1.50841289+1.78798918e+00j,
       2.1222791 -1.52039976e-01j, 4.12278797+1.21261831e+00j,
       2.52139618+3.38937859e-01j, 2.69372659+2.67287365e+00j,
       4.08330889+2.00043908e+00j, 1.55203912-1.86321908e+00j,
       1.97540093-2.17333078e+00j, 1.9929065 -7.94963809e-01j,
       4.2262444 +3.11418768e+00j, 1.6666204 -6.69565923e-02j,
       3.62020418-2.58535675e+00j, 4.18868572+1.95588817e+00j,
       2.06823554+1.47483553e+00j, 3.26507985+1.83384938e+00j,
       1.62456186-2.67508101e+00j, 2.30783706-2.07184740e+00j,
       2.5996652 -8.94163773e-01j, 4.86049117+3.68043423e-01j,
       4.54386104-1.54825432e+00j, 1.08488083+5.96468429e-01j,
       3.11020782-1.63179907e+00j, 3.05426024-2.61829539e+00j,
       2.75578181+2.80630050e+00j, 4.9701032 -2.61408238e+00j,
       4.48679993+2.15016796e+00j, 4.02743041+1.77052527e+00j,
       1.22610476+2.24088971e+00j, 3.41632217-2.09978246e+00j,
       2.90046086+2.51030449e+00j, 4.35467768-8.66912056e-01j,
       1.53316981-6.17278123e-01j, 4.33258876+1.66394346e+00j,
       3.43969981+2.30673265e+00j, 4.4972382 +1.10863385e+00j,
       3.81773541+2.08419180e+00j, 2.89020717-1.78313407e+00j,
       2.08922773+4.49859513e-01j, 2.13427422+7.69786632e-01j,
       1.70038557+2.02513546e+00j, 1.15086406+7.86942526e-01j,
       1.04602318-8.06609033e-01j, 1.90284663+1.68227795e+00j,
       3.6610381 -3.03634716e+00j, 1.32115041-1.59968741e+00j,
       2.71339596+4.60364686e-01j, 3.51333137-2.57334170e+00j,
       4.35163813+2.61007239e+00j, 2.67082377+3.20505530e-01j,
       1.61233238-1.24743265e+00j, 3.75332208+1.40092186e+00j,
       3.1786331 -2.08849726e+00j, 4.02781332+1.76266569e+00j,
       2.42663622-1.45279467e+00j, 4.41916471+2.16966442e+00j,
       1.222321  -2.42555705e+00j, 3.3280264 -1.97774347e+00j,
       2.30070816-2.02034693e+00j, 2.09338241+9.24554486e-01j,
       1.17699325-1.74786998e+00j, 4.35625952+1.46035767e+00j,
       1.88170211+4.40228703e-01j, 1.29927504+1.56129174e+00j,
       1.18048318+2.14396095e+00j, 2.65837842+2.29388547e+00j,
       2.18435262-1.82729496e+00j, 2.93316014-1.79328379e-01j,
       2.18711331+1.52194174e+00j, 2.08870264-1.54862180e+00j,
       3.71350623+1.47801139e-01j, 3.00791756-2.44543529e+00j,
       2.06093497+2.06096680e+00j, 2.08375225+1.63675785e+00j,
       4.16692053-4.76771814e-01j, 2.26545581-8.19307760e-01j,
       3.15096919+6.31748639e-01j, 3.77312367-3.04197111e+00j,
       1.09323579-3.12419234e+00j, 2.72776073-1.15213256e-01j,
       1.07442194-2.55533061e+00j, 4.24779931-2.56158822e+00j,
       3.56445079+3.84837315e-01j, 4.9595572 -1.85489670e+00j,
       1.02638467-1.21650347e+00j, 4.09945067+1.98062322e+00j,
       2.69456954-7.71231015e-01j, 3.58808906-2.21594048e+00j,
       2.87478215-2.16484415e+00j, 3.77853965-2.82253512e+00j,
       1.70835038-1.46314939e+00j, 1.0800647 -2.93518700e+00j,
       4.25573687-1.43969344e+00j, 1.29423538-1.68857047e+00j,
       3.69992782-2.05510807e+00j, 1.24234966+1.61400579e+00j,
       2.94134912+2.03148439e+00j, 4.25196589+3.61223012e-01j,
       3.8180477 +1.85901857e+00j, 4.34458853+2.62495279e+00j,
       1.95540709+3.07359849e-01j, 2.23378898-1.52525310e+00j,
       4.39025384-5.44455584e-01j, 4.19896977+2.95328015e+00j,
       1.59825624+1.85770286e+00j, 2.24835618+2.34207538e+00j,
       2.22482569+3.59208561e-01j, 3.23652443-1.86796563e+00j,
       2.78421541-4.35462225e-01j, 1.88037845+3.08954906e+00j,
       1.37249364+1.25273292e+00j, 3.76826566+6.46494122e-01j,
       4.20642751+5.50437047e-01j, 3.02374586+2.96866088e+00j,
       1.83734139-1.20436064e+00j, 1.639836  -2.53898333e+00j,
       1.87298274+2.79604815e+00j, 4.85651534-1.64634046e+00j,
       4.67275947+5.81585798e-01j, 2.35810069+1.22522538e-01j,
       3.01130978+4.34847680e-01j, 1.78448119-8.66476544e-01j,
       3.9308045 +8.12604705e-01j, 3.48928562+1.50801045e+00j,
       3.4834859 +1.16233986e+00j, 1.11810905-2.33652922e+00j,
       4.00025874+2.81168400e+00j, 1.40369606+2.90372908e-01j,
       4.34207883+3.08983104e+00j, 3.41167461+1.71743869e+00j,
       3.30651727+2.30985287e+00j, 3.40762754+1.55850992e-01j,
       3.19670739+1.71458901e+00j, 3.51796903-1.32271940e-01j,
       1.99707315-1.85014364e+00j, 4.45393114-2.20326337e+00j,
       3.57980957-1.11937309e+00j, 3.01952367-2.63136427e-01j,
       3.2923132 +1.83783688e+00j, 2.01140683-3.00889704e+00j,
       1.11583833-1.76340065e+00j, 4.57721793+1.93423037e+00j,
       4.59163992+2.23107393e+00j, 2.78667422+1.54565416e+00j,
       4.52951856+2.06560062e+00j, 1.27826459+2.09620314e+00j,
       1.48120051-1.83355010e+00j, 3.3317752 +3.05314451e-01j,
       4.85454786+1.87958324e+00j, 4.25609265-1.77912462e+00j,
       3.97589729+2.10872571e-01j, 3.96346068-8.85173772e-01j,
       4.76698752-1.99498900e+00j, 2.29515631-2.41719346e+00j,
       2.58841268+1.51135280e-01j, 2.69439518-1.97452189e+00j,
       2.60932774-4.45685094e-01j, 2.26442133-4.87326185e-01j,
       1.28634517+1.19364134e+00j, 4.1892641 +1.78629834e+00j,
       1.49226142+4.98354386e-01j, 3.12084519+3.79875519e-01j,
       1.50615807-1.58521785e+00j, 2.42389515-1.41352876e+00j,
       2.17712264-7.57811684e-02j, 4.03398033+2.99086850e+00j,
       4.74582411-2.87681882e+00j, 1.19005848-1.33043276e-01j,
       4.10797047-5.02799187e-01j, 4.72501322-2.08460500e+00j,
       2.12346726+1.74409836e+00j, 3.51777463+1.25796413e+00j,
       3.55140452-2.67726675e+00j, 3.56495194-3.42369503e-01j,
       3.87680008-9.36863584e-01j, 2.25732042+1.47206386e+00j,
       1.6259652 -2.01694692e+00j, 3.27545123-6.49684367e-01j,
       2.43759034-1.40114473e+00j, 2.91395877+7.70112709e-01j,
       2.66592325+9.66716028e-01j, 4.35051641-2.55820307e+00j,
       1.71695317-1.21808490e+00j, 1.88903952-7.07846624e-01j,
       2.35397827-1.40999060e+00j, 3.91638757-2.54848033e+00j,
       3.55330044-2.62759757e+00j, 4.47618025+1.21942203e+00j,
       4.86531366+2.49420237e+00j, 2.25806082-7.74544461e-01j,
       3.66983197-1.75884207e+00j, 3.58842955+3.93680326e-01j,
       3.49225775+9.12499661e-01j, 3.89152708+8.16389809e-01j,
       2.06940424-1.29661390e+00j, 4.49840643+2.17245765e+00j,
       3.36713749-2.13512273e+00j, 1.26117788+1.63889301e+00j,
       1.79417339-2.76033876e+00j, 3.5991299 +3.08520494e+00j,
       2.29545044+6.96631026e-01j, 1.65429629+1.94146369e+00j,
       4.83734389-3.01366436e+00j, 3.60913829-2.57622306e+00j,
       3.21049287+2.02093165e+00j, 2.59859689+4.63101941e-01j,
       3.7912988 -2.10900668e+00j, 1.071433  +2.34419206e+00j,
       4.2436344 +2.78867401e+00j, 4.13595255+1.81333647e+00j,
       1.65591149+6.76256493e-01j, 3.39560896+1.51170333e-01j,
       3.689631  -1.15068671e+00j, 2.11579816-1.72104162e-01j,
       1.24389861-2.62053606e+00j, 2.77840892-7.28964587e-01j,
       2.20446436-3.05444600e+00j, 3.14192562+2.97331213e+00j,
       3.03394915+1.20610430e+00j, 3.14846828-1.12921825e+00j,
       4.77180381-2.02881448e+00j, 1.61020764-2.34546289e+00j,
       3.58097463-3.02530060e+00j, 2.22958391-1.15643392e+00j,
       2.63213412+9.51313432e-01j, 4.69841958+2.59800740e+00j,
       4.09816942-7.71207490e-02j, 3.07484043+2.67584147e+00j,
       1.45262586+1.27200187e-01j, 2.04552328+4.37753537e-01j,
       3.8356655 +2.29815610e+00j, 1.1817281 -2.51917557e+00j,
       2.19862128+7.03795416e-01j, 1.46365126+2.29804988e+00j,
       2.33069189-2.63553942e+00j, 3.01470253+1.27812001e-01j,
       3.27919138-2.16777153e+00j, 2.97395034+1.69918884e+00j,
       4.94382129-2.66456689e+00j, 3.19455211+1.31100195e+00j,
       4.43988997-1.32664283e+00j, 4.13088157+4.97899022e-01j,
       3.69815562-9.75466814e-01j, 3.90774805-4.86693536e-01j,
       3.60134092-1.71777644e+00j, 4.69831427-1.44828164e+00j,
       4.4758008 +2.86429377e+00j, 4.80813598+1.92438345e+00j,
       1.94062489-1.42823449e+00j, 4.36940241+1.91009561e+00j,
       3.47623145+1.21816172e-01j, 2.91557193+9.39405332e-01j,
       1.37064051-2.80841161e+00j, 4.13801145+5.88452509e-01j,
       1.63079978+2.10082607e+00j, 2.1006847 +1.33932087e+00j,
       1.61233516+1.14208388e+00j, 4.4613286 -8.91316684e-01j,
       1.07888021+2.61016937e+00j, 1.84048517+2.73963448e+00j,
       4.9630971 -2.95644274e+00j, 1.73706243+2.28748007e+00j,
       3.00593715+2.17841556e+00j, 3.51439761-1.51182445e+00j,
       1.70979082-1.44760433e+00j, 1.12788845+2.22779755e+00j,
       1.76586391-6.27655933e-01j, 2.0970936 -6.07693456e-01j,
       3.05616904+2.91241424e+00j, 3.58988419-6.51685621e-01j,
       2.75416648+1.36181138e+00j, 4.21770919+8.26300775e-01j,
       4.65085204+7.83804505e-01j, 4.13066818-3.09715383e+00j,
       3.44898038-2.25479764e+00j, 1.98926442+2.35752191e+00j,
       4.2359955 +1.99938247e+00j, 2.27239267+1.78083694e+00j,
       2.74196305-2.56293979e+00j, 2.16781112+1.19095236e+00j,
       1.04853098-1.57680858e+00j, 3.66220808+7.26199169e-01j,
       4.28720994-1.25475238e+00j, 2.13001652-2.91580207e+00j,
       4.3059824 -8.68170303e-01j, 2.89569484+1.73312110e-01j,
       2.41197562+1.99216439e+00j, 3.90221897+7.98760319e-01j,
       4.68721033-2.08622590e-02j, 3.84207253-1.81434158e+00j,
       2.35326558+2.13284749e+00j, 2.72476214+2.76311976e+00j,
       3.51015451+2.46534374e+00j, 2.76909936+9.01068976e-01j,
       3.26916683-2.31750318e+00j, 1.26985856+2.10589128e+00j,
       4.36704576-2.40937650e+00j, 4.34004576-2.25928722e+00j,
       2.41418189-7.21147973e-01j, 1.50115647+4.47511941e-01j,
       1.78430644+2.95035793e+00j, 1.45316335-9.71914967e-01j,
       2.47089379-1.09962531e+00j, 4.72893569+5.38299755e-01j,
       4.11782316-1.46405836e+00j, 2.27992114-2.74807705e+00j,
       4.95545987+9.90386100e-01j, 3.52179324+6.14829378e-01j,
       1.99699312-1.14522858e+00j, 1.555458  +1.68053922e+00j,
       1.42489274+2.51242524e-01j, 2.42903436+2.00950931e+00j,
       2.6589038 +2.21790287e+00j, 2.59286583+2.75794448e+00j,
       3.58455735-3.59983239e-01j, 2.17512206+5.62084742e-01j,
       1.05407915+1.21447337e-01j, 2.78741599+2.97333210e+00j,
       2.72297017-1.18586821e+00j, 4.87446718-1.61331912e+00j,
       1.30434469+1.59140853e+00j, 2.98584722+2.42699201e-01j,
       3.82708286-1.34720173e+00j, 4.44468397+1.26810812e+00j,
       3.30746431+5.37828153e-01j, 3.99851454+2.23852027e+00j,
       4.98040602+6.81146675e-01j, 4.49325533+3.53525016e-02j,
       4.67860911+2.40877111e+00j, 1.65046724+2.13743497e-02j,
       3.86453715+1.72431486e+00j, 4.16425087+2.73286005e+00j,
       4.86353537-1.19429136e+00j, 2.45464223-1.31333305e+00j,
       4.17407503-1.59935439e+00j, 2.58165429-7.83908213e-01j,
       2.39249435+2.05414564e+00j, 2.05073627-2.07632588e+00j,
       2.82895795-1.39047254e+00j, 2.17568774-1.55821737e+00j,
       3.88940084-9.24578841e-01j, 2.1727963 +9.54902788e-01j,
       3.94186266-2.12159872e+00j, 4.7516956 -2.77999735e+00j,
       4.03607184+2.34185015e+00j, 2.07164872-2.47575889e+00j,
       2.64407418-2.70012207e+00j, 4.77920946+1.53082314e-01j,
       2.04999809-2.04427233e+00j, 2.97789957-2.87824670e+00j,
       3.21136269+6.19681697e-01j, 4.94836289+2.56734532e+00j,
       2.26543028-2.49331441e+00j, 4.46681947-1.22426158e+00j,
       3.40812651+2.58343012e+00j, 2.81357756-2.69685488e+00j,
       4.51876728-3.56689326e-01j, 1.99218974+8.37052387e-01j,
       1.04845473-9.45939807e-01j, 2.8480025 +1.96426799e+00j,
       2.76895129-2.85267088e+00j, 3.16943631+3.04791237e+00j,
       4.42378869-1.56664327e+00j, 2.18615768-1.49636329e+00j,
       3.7218677 +1.62129221e+00j, 3.75500323+8.31361548e-02j,
       4.95697088-1.87750097e+00j, 4.64632481-7.33930504e-01j,
       2.70954176-2.26264665e+00j, 1.51485885+3.04963264e+00j,
       2.71217903-2.72374485e+00j, 4.98333448+1.14400973e-01j,
       1.90878724+1.23700012e+00j, 4.53658003-7.17777478e-01j,
       3.68058921-3.11986565e+00j, 2.74259699-1.33296689e+00j,
       1.11243636+2.14508990e+00j, 3.44016874-1.50604632e+00j,
       2.79667549+2.52862328e+00j, 4.77031029-1.05605484e-01j,
       2.41310347+2.02498895e+00j, 2.5092716 +8.60711406e-01j,
       2.42544135+1.30349875e-01j, 1.37527632+7.45233868e-01j,
       2.84279232+9.11458305e-01j, 4.10128933-1.93233106e+00j,
       2.93795314+3.55493665e-01j, 4.60368292+2.36133889e+00j,
       3.38158468+1.80757398e+00j, 2.76479184-2.67545409e+00j,
       4.86656111+6.99677663e-01j, 1.67428055+2.00265514e+00j,
       2.09706455+8.07224490e-01j, 2.51950135-1.97343342e+00j,
       3.5511474 +4.15264021e-01j, 3.21352258+4.82906239e-01j,
       1.16042511+2.27773312e+00j, 3.50111642-2.91313154e+00j,
       1.77302224-2.92722554e+00j, 3.53174722+2.83888368e+00j,
       2.40660673+9.39614152e-01j, 4.01425893-2.64578548e+00j,
       1.29183204-1.92298503e+00j, 1.86143733-2.13976347e+00j,
       1.72099344-1.00561209e+00j, 4.67975463-7.82414647e-01j,
       3.14192306+4.25133309e-01j, 4.27007557+2.17009905e+00j,
       1.2484189 +9.94553296e-01j, 1.67311555-2.82096435e+00j,
       3.74645245-4.23125978e-01j, 3.66548934-1.82336142e+00j,
       3.47770426+1.01521005e+00j, 2.00998428-6.75220025e-02j,
       1.090291  +1.27716260e+00j, 3.49997619-1.28602126e+00j,
       4.55278112+1.46145131e+00j, 2.69998259-1.10037942e+00j,
       2.39624798+1.58966876e+00j, 1.69615648-1.64624981e-01j,
       4.47071449-1.89311352e+00j, 3.98078836+2.51405470e+00j,
       4.83953896+2.65129909e+00j, 4.17129015-2.14777002e+00j,
       4.58907759+2.47753915e+00j, 2.60314919-8.93168524e-01j,
       3.07614984-6.31280082e-01j, 3.2914583 +3.01282950e+00j,
       1.281301  -2.62090777e+00j, 1.83152739-2.29005815e+00j,
       3.48549077-1.47430504e+00j, 2.03380267-1.83870945e+00j,
       4.57347334-9.69408831e-02j, 2.08212153+2.12302759e+00j,
       1.47854981+2.18812116e+00j, 4.00842846-6.79963906e-01j,
       2.47703361+2.86090941e-01j, 1.93396345-9.46224655e-01j,
       2.59284319+1.10445076e+00j, 1.21495427-3.13104202e+00j,
       1.03563104-1.99810760e+00j, 1.43464371-1.65540191e+00j,
       3.03193499+1.18091056e+00j, 1.44908715+1.51528386e-01j,
       1.53478698-1.47976470e+00j, 3.66457868-5.30442648e-01j,
       3.04725385-1.60914705e+00j, 2.81554764-1.98833397e+00j,
       1.9741296 +2.37491315e+00j, 1.76001262+2.19005869e+00j,
       1.82061007-1.74755122e+00j, 2.4916317 +1.67254828e+00j,
       2.47525819+2.76887067e+00j, 2.86745955-3.70563767e-01j,
       3.70166431+2.52694583e+00j, 2.8042725 +1.06895442e+00j,
       2.38344495+7.29400971e-01j, 2.60746449+8.11062558e-01j,
       2.03597324-2.80890860e-01j, 2.49179242+3.07779661e-01j,
       1.34935833+4.79790810e-02j, 4.33673552+1.12888650e+00j,
       4.80305229-1.47499400e-02j, 4.57686341+1.50057424e+00j,
       2.46646397+7.81430104e-01j, 4.12739576-1.75199917e+00j,
       4.227794  +1.70300938e+00j, 4.17458244+2.10576365e+00j,
       3.04425396+1.47502752e+00j, 1.06445778-2.09947818e+00j,
       4.44838268-1.49468455e+00j, 2.02041566+5.85806649e-01j,
       2.08995551+4.14317037e-01j, 3.65365353-2.30979459e+00j,
       4.42353493-1.91102278e+00j, 1.49452557-1.27657783e+00j,
       4.39249454-9.96737246e-02j, 4.16790977-2.19539479e+00j,
       2.08071644-8.44963747e-01j, 1.2230783 +7.43272150e-01j,
       4.53122517+9.24799522e-01j, 2.95653111+1.69203725e+00j,
       1.34857935-5.44654597e-01j, 3.54471147-1.44847007e+00j,
       4.33499944+1.57754641e-01j, 1.98428   -4.90627257e-01j,
       1.84066794-2.59774110e+00j, 4.05285088+1.87716233e+00j,
       3.59173243-1.22534989e+00j, 2.33791555+3.15620810e-01j,
       3.98928231+1.55245520e-01j, 3.7533485 +2.81959688e+00j,
       3.32540606-8.23931442e-01j, 1.53823097-1.69681700e+00j,
       2.62289276-1.69654077e+00j, 2.20651622+2.15180828e-01j,
       3.86408827-4.99284430e-02j, 3.07832743+2.12525191e+00j,
       4.66332744-1.52773924e+00j, 1.15607065+1.39606411e+00j,
       1.50482783-5.50412284e-01j, 3.79236637+2.56205711e+00j,
       3.42125401+9.76409874e-01j, 3.32772719+1.51230858e+00j,
       1.54636291-2.12447327e+00j, 1.45239468-2.06971214e+00j,
       1.78917343+1.71260361e+00j, 2.06082839-6.97487867e-01j,
       3.38320966-1.72251841e+00j, 3.50947636+1.04105513e+00j,
       3.9300596 -2.98943422e+00j, 3.78067999-1.16515196e+00j,
       3.82378583+6.46258854e-01j, 1.53813131-3.07277068e+00j,
       3.45413612-1.62639827e+00j, 4.39891942-1.89405895e+00j,
       3.11583198-2.77723200e+00j, 1.8595579 -1.04846473e+00j,
       1.50931295-2.35939660e+00j, 1.78151451+1.46973398e+00j,
       2.87532528-2.43776930e+00j, 3.92916052+6.77070437e-01j,
       3.48123916+1.54891480e+00j, 2.55588731+2.61934853e-01j,
       1.94901226+2.85235454e+00j, 4.93801562-6.48292168e-01j,
       1.46828237+3.02816577e+00j, 4.99177008-2.75009178e+00j,
       2.72752816+6.24428585e-02j, 1.12767417-2.82196410e+00j,
       4.19091003-1.98830574e+00j, 2.50894447-2.42079401e+00j,
       4.59089857+8.69442447e-01j, 4.83347978+2.33085830e+00j,
       2.28401592-8.95192417e-01j, 3.5357972 +2.82277513e+00j,
       2.85218837-8.53979569e-01j, 3.29061809+9.92959013e-01j,
       1.71911062+2.16925999e+00j, 3.31539712-3.09018141e+00j,
       4.12465461-7.19723350e-01j, 3.03825386+2.25583348e+00j,
       1.47097404-4.32645032e-01j, 2.22745891-2.86359062e+00j,
       3.19278068+2.05372335e+00j, 1.06261661-1.57455119e+00j,
       4.44945196-1.87718473e+00j, 4.09219802-2.15061713e-01j,
       2.31018769-1.68993432e+00j, 2.40190441-1.37099185e+00j,
       4.00770691-2.02228880e+00j, 1.57295877+2.99605019e+00j,
       2.79007044+2.76060244e+00j, 4.81535663-7.45243098e-02j,
       3.93452458-2.74288702e+00j, 3.9443326 -2.23048107e+00j,
       3.35742304-7.03320699e-01j, 1.63684479-3.11861838e+00j,
       1.6093847 +1.22358899e+00j, 3.28317535-2.87844183e+00j,
       2.03442644+2.36817782e+00j, 1.72168859-1.05137283e+00j,
       3.0628406 -8.20976623e-01j, 3.10682323+2.45623242e+00j,
       2.37863124+1.04641230e+00j, 1.62728951+2.21528213e+00j,
       1.15327361-2.99891954e+00j, 3.30433432+2.15081530e+00j,
       4.01559788-2.57170231e+00j, 1.93915504-2.04114494e+00j,
       1.61333268-1.91436227e+00j, 4.89536663-1.06505309e+00j,
       3.35068226-1.35022812e+00j, 2.710548  -2.78023131e+00j,
       3.75070719+2.56186039e+00j, 2.47889344+2.22140358e+00j,
       3.69213996+2.11097140e+00j, 1.66031822+1.81545983e+00j,
       4.03464504-1.94412467e+00j, 1.73366454-4.72822010e-01j,
       3.87253716-3.08715950e+00j, 2.33013169+1.79628578e+00j,
       3.03647097-2.48138667e+00j, 2.0734772 -1.87577664e+00j,
       3.05297866+1.50026211e+00j, 3.75685536+1.11612248e+00j,
       2.78727899+2.33377978e+00j, 1.0039376 -2.46170679e+00j,
       2.97053482+9.81789988e-01j, 1.32003829+9.04242731e-01j,
       1.13890172-2.81539795e-01j, 4.50975976-2.74118347e-01j,
       1.78556569+2.70067270e+00j, 1.81820986+1.34877026e+00j,
       4.48408714+1.09589515e+00j, 1.94901752+1.53710441e+00j,
       2.07419601-1.78247583e+00j, 1.02064598-6.68250821e-01j,
       1.93396226-2.76914245e+00j, 2.53149209+2.79884533e+00j,
       3.39201123-1.74900188e+00j, 3.12695767+2.57889011e+00j,
       1.06681109-1.16935990e+00j, 3.07160635-2.57220497e+00j,
       4.75346443+1.40303942e+00j, 2.19325162-3.08987412e+00j,
       1.47142074+1.16824479e+00j, 3.95232437+3.83343191e-01j,
       2.96148495-1.86642137e+00j, 2.20241329-4.91637928e-01j,
       2.50380825+3.17624324e-01j, 1.4727768 +3.40332745e-01j,
       2.29305258+1.24647476e+00j, 4.33183699-2.12611839e+00j,
       3.44912247+2.45561879e+00j, 2.74740141-1.15551679e+00j,
       4.11537165-2.05670569e+00j, 4.04305597-2.83196646e+00j,
       2.06543899-1.53352140e+00j, 2.56702487-1.47077803e+00j,
       3.60679055+3.47151950e-01j, 1.83353017+1.37012555e+00j,
       1.90504573+6.04443254e-01j, 4.8292046 +1.40444443e-01j,
       4.64561019-2.72781095e+00j, 2.98113417+2.85925051e+00j,
       1.90790964+2.29499746e+00j, 4.05324275+1.89155265e+00j,
       1.94167754-1.25853606e+00j, 1.522355  +2.75209419e+00j,
       4.17610281-3.11737823e+00j, 1.01410966+1.80750809e+00j,
       3.47244151+2.16985224e+00j, 3.92927384-2.81169265e+00j,
       3.92081134+1.17915248e+00j, 1.98735298-1.82556897e+00j,
       3.91712236+4.29790126e-01j, 2.39852277+2.73277043e+00j,
       3.0260251 -2.00678190e-01j, 1.49843952+1.91289322e-01j,
       2.0666795 +1.88804093e+00j, 4.94354663+2.35552067e+00j,
       2.00883058-1.30417918e+00j, 2.21773041-8.23701412e-02j,
       4.95978282+9.66959327e-01j, 1.56459789-2.29904876e+00j,
       1.3986759 +1.72593109e+00j, 4.74392819-1.28632243e+00j,
       3.45424776+1.53357475e+00j, 3.54008101-6.78627429e-01j,
       4.36614264+9.23749220e-02j, 4.08174168-2.85533370e+00j,
       3.07733496-9.72172213e-01j, 3.3896883 +5.80687830e-02j,
       3.20030493+2.69768329e+00j, 4.68164895+1.44792483e+00j,
       2.89278657+5.16116917e-02j, 2.0691362 +1.35423644e-01j,
       4.05167193-2.10855115e+00j, 1.32504292-2.95575240e+00j,
       3.72688306+7.59866965e-01j, 4.38551983-1.52306608e+00j,
       1.60750475+2.75277816e+00j, 1.40970628+2.60502500e+00j,
       4.51353915-2.10339071e+00j, 3.06183595+3.06051960e+00j,
       2.73382579+2.28522613e+00j, 4.42437212-1.50533402e+00j,
       4.15947947-5.27865197e-02j, 3.92667722+3.01651555e+00j,
       4.62563927+2.11808763e+00j, 1.83150518-1.01816284e+00j,
       4.93910588+4.50711513e-01j, 4.27271985+2.97950676e-01j,
       1.61765505+3.04666860e-01j, 2.7527523 +2.56923818e+00j,
       4.17543696+2.33102491e+00j, 4.58998852+2.35398776e+00j,
       3.7217064 -2.54148169e+00j, 4.74821037+1.32106751e+00j,
       4.38666319-2.34102531e+00j, 1.70123566+9.83696279e-01j,
       3.54039969-2.45525901e+00j, 4.54619049-1.05074884e+00j,
       4.6634302 -2.17018762e+00j, 4.617     +5.97503739e-01j,
       4.13650346-1.02727640e+00j, 2.04849103+2.00926695e+00j,
       4.0543003 +2.16221596e+00j, 1.95260674+2.90359863e+00j,
       1.01747342-1.64900232e+00j, 4.19305512+3.81158036e-01j,
       1.01940297+4.42965502e-01j, 2.92802521+8.47636078e-01j,
       1.59505401+1.62093014e+00j, 2.49416777-1.00798253e+00j,
       2.24176417-4.85300917e-01j, 3.63203346-4.45084477e-01j,
       4.22849058+3.10193800e+00j, 4.47043778+5.67739537e-01j,
       2.0622375 +2.37154477e+00j, 3.22563514-1.51715426e+00j,
       2.11794588-4.16728060e-01j, 3.73609307-8.22983792e-01j,
       4.30158417-1.45945750e+00j, 4.73719268+2.33124126e+00j,
       2.99561633-1.62349825e+00j, 2.40127562+1.20710502e+00j,
       2.43701554+1.10275454e+00j, 4.91066292+3.03069091e+00j,
       2.6597076 +1.22219697e+00j, 3.25051134-2.63636795e+00j,
       4.93222847+4.13820676e-01j, 2.37719124-2.77114722e+00j,
       1.96484286+1.19448295e+00j, 2.82776669+6.99332377e-01j,
       3.47066721-1.50039374e+00j, 1.29479511+1.11055770e+00j,
       1.82782008+2.42112939e+00j, 1.40584052-2.67252096e+00j,
       4.28099498-2.82070767e+00j, 3.21027815-2.57121652e+00j,
       2.49125096-1.44370478e+00j, 3.23497829-1.69407977e+00j,
       4.33790674-7.09555111e-01j, 1.0371629 +2.61808871e+00j,
       2.73960467-2.27547428e+00j, 3.02894742-1.07881063e-01j,
       3.2633332 -1.22637801e+00j, 3.53750344+2.54105986e+00j,
       2.47484376-1.93626385e+00j, 2.53567367+3.10837166e+00j,
       2.26079558+2.50778449e+00j, 4.03392068+2.05998754e-02j,
       2.80845908+9.32450506e-01j, 1.52639685-1.35607204e+00j,
       3.59344736-5.86677393e-01j, 4.75665479+2.73405638e+00j,
       2.14996551+3.11491227e-01j, 3.61577297+1.53139621e+00j,
       2.0706468 -1.38417377e+00j, 1.34893638+3.12617082e+00j,
       3.6165142 -2.49666817e+00j, 1.10967927-2.17359917e-01j,
       1.15456161-2.23529451e+00j, 3.56458794-8.05900534e-01j,
       2.23544457-1.76561127e+00j, 4.21921661-2.35304723e+00j,
       3.45825876+2.25164622e-01j, 2.46967812-1.36815707e+00j,
       2.4182106 +1.16405350e-01j, 2.0944612 -4.82738641e-01j,
       1.3018302 -1.60134055e+00j, 4.08847102-1.57949836e+00j,
       1.42916232-1.98289408e+00j, 4.85058993+1.05779276e+00j,
       2.4666037 -8.95865346e-01j, 4.55224634-1.13852839e+00j,
       4.84893519+7.77522766e-01j, 3.84449196+2.23287579e+00j,
       4.10338537-1.29023595e+00j, 2.70897616+2.36447606e+00j,
       3.14211757-1.63540151e+00j, 1.55515806+2.82490039e+00j,
       1.81920086+1.58705595e-02j, 1.99183378-1.11484702e+00j,
       1.20185608-6.89953101e-01j, 1.83471707-9.20781966e-01j,
       4.01887569-3.08694308e+00j, 4.12301378-2.82529738e+00j,
       1.87106488-8.19209782e-01j, 4.49639818+2.85067784e+00j,
       4.78205126-2.92049898e+00j, 3.59415022-1.09815413e+00j,
       4.95546254-6.81172865e-01j, 2.01654135+1.51819049e+00j,
       4.38039394-2.97672250e+00j, 3.18708765+1.01783552e+00j,
       2.67074579-1.82911827e-01j, 1.17378273+8.65564572e-01j,
       2.19306532-2.39184332e+00j, 3.23681804+7.04747251e-01j,
       4.94794269+2.46910069e+00j, 4.10821967+6.39258844e-01j,
       4.62013448+1.81445004e-01j, 4.98462812-2.95689805e+00j,
       4.38932895+1.03275043e+00j, 2.88477799-1.28296635e+00j,
       2.98430116+3.04651517e+00j, 4.75165587+5.32640233e-01j,
       3.74507647-2.12688106e+00j, 2.25818353+1.08553123e+00j,
       3.82830823-1.77832387e+00j, 2.46506105-2.93267640e+00j,
       1.00852993-2.84592077e+00j, 2.35050779+7.33348465e-01j,
       1.97917448-1.52233896e+00j, 3.66233133+1.79439983e+00j,
       2.78631208+1.54173527e+00j, 3.15225039-2.27958551e+00j,
       2.87314672-2.33851926e+00j, 3.56888002-2.74652774e+00j,
       4.76018399+2.93884281e+00j, 3.06253389-6.72544449e-02j,
       4.42281951+1.87405139e+00j, 1.79129321-6.92213188e-01j,
       2.05775709+3.41531648e-01j, 1.07408137+9.69020070e-01j,
       3.04027432+2.30468396e+00j, 2.34002201-5.72208092e-01j,
       3.8970957 +2.77371157e+00j, 4.04460653-2.02526831e+00j,
       1.85723806-2.78423016e+00j, 3.34698813+1.50235842e-01j,
       4.61447562-2.39143569e+00j, 3.03912709-2.62589879e+00j,
       1.84460144+5.23020493e-01j, 3.11044238-2.28524042e+00j,
       4.04872194-8.51537411e-01j, 4.51585444+5.98204175e-02j,
       1.35875239-1.23266328e+00j, 3.39657389-1.38210585e+00j,
       3.73692644+7.98699170e-01j, 3.10900581+2.12136900e+00j,
       2.11616413-2.53424342e+00j, 3.21048993+1.24125321e+00j,
       2.94609993-6.30812800e-01j, 3.93695576+7.08415920e-01j,
       4.24774422+1.46851018e+00j, 2.95932135-7.78385538e-01j,
       3.08965695+1.77411066e+00j, 3.2387947 -2.78782066e-01j,
       4.8638202 -2.84073274e+00j, 2.83240194-8.99297759e-01j,
       2.97773139+3.11954644e+00j, 4.15556031-5.98567674e-02j,
       2.49963669+1.17579988e+00j, 2.42600352-3.10418584e+00j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
