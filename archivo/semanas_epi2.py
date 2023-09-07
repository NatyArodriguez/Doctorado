import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from math import *


def orden(v):
    SE=np.zeros(96)
    SE[0]=sum(v[104:108]) #ENE09 
    SE[1]=sum(v[108:112]) #FEB09 
    SE[2]=sum(v[112:117]) #MAR09 
    SE[3]=sum(v[117:121]) #ABR09 
    SE[4]=sum(v[121:126]) #MAY09
    SE[5]=sum(v[126:130]) #JUN09
    SE[6]=sum(v[130:134]) #JUL09
    SE[7]=sum(v[134:139]) #AGO09
    SE[8]=sum(v[139:143]) #SEP09
    SE[9]=sum(v[143:148]) #OCT09
    SE[10]=sum(v[148:152]) #NOV09
    SE[11]=sum(v[152:156]) #DIC09
    SE[12]=sum(v[156:161]) #ENE10
    SE[13]=sum(v[161:165]) #FEB10
    SE[14]=sum(v[165:169]) #MAR10
    SE[15]=sum(v[169:173]) #ABR10
    SE[16]=sum(v[173:178]) #MAY10
    SE[17]=sum(v[178:182]) #JUN10
    SE[18]=sum(v[182:186]) #JUL10
    SE[19]=sum(v[186:191]) #AGO10
    SE[20]=sum(v[191:195]) #SEP10
    SE[21]=sum(v[195:200]) #OCT10
    SE[22]=sum(v[200:204]) #NOV10
    SE[23]=sum(v[204:208]) #DIC10
    SE[24]=sum(v[208:213]) #ENE11
    SE[25]=sum(v[213:217]) #FEB11
    SE[26]=sum(v[217:221]) #MAR11
    SE[27]=sum(v[221:225]) #ABR11
    SE[28]=sum(v[225:230]) #MAY11
    SE[29]=sum(v[230:234]) #JUN11
    SE[30]=sum(v[234:239]) #JUL11
    SE[31]=sum(v[239:243]) #AGO11
    SE[32]=sum(v[243:247]) #SEP11
    SE[33]=sum(v[247:252]) #OCT11
    SE[34]=sum(v[252:256]) #NOV11
    SE[35]=sum(v[256:260]) #DIC11
    SE[36]=sum(v[260:265]) #ENE12
    SE[37]=sum(v[265:269]) #FEB12
    SE[38]=sum(v[269:273]) #MAR12
    SE[39]=sum(v[273:278]) #ABR12
    SE[40]=sum(v[278:282]) #MAY12
    SE[41]=sum(v[282:286]) #JUN12
    SE[42]=sum(v[286:291]) #JUL12
    SE[43]=sum(v[291:295]) #AGO12
    SE[44]=sum(v[295:300]) #SEP12
    SE[45]=sum(v[300:304]) #OCT12
    SE[46]=sum(v[304:308]) #NOV12
    SE[47]=sum(v[308:312]) #DIC12
    SE[48]=sum(v[312:317]) #ENE13
    SE[49]=sum(v[317:321]) #FEB13
    SE[50]=sum(v[321:326]) #MAR13
    SE[51]=sum(v[326:330]) #ABR13
    SE[52]=sum(v[330:334]) #MAY13
    SE[53]=sum(v[334:339]) #JUN13
    SE[54]=sum(v[339:343]) #JUL13
    SE[55]=sum(v[343:347]) #AGO13
    SE[56]=sum(v[347:352]) #SEP13
    SE[57]=sum(v[352:356]) #OCT13
    SE[58]=sum(v[356:360]) #NOV13
    SE[59]=sum(v[360:364]) #DIC13
    SE[60]=sum(v[364:369]) #ENE14
    SE[61]=sum(v[369:373]) #FEB14
    SE[62]=sum(v[373:378]) #MAR14
    SE[63]=sum(v[378:382]) #ABR14
    SE[64]=sum(v[382:386]) #MAY14
    SE[65]=sum(v[386:391]) #JUN14
    SE[66]=sum(v[391:395]) #JUL14
    SE[67]=sum(v[395:400]) #AGO14
    SE[68]=sum(v[400:404]) #SEP14
    SE[69]=sum(v[404:408]) #OCT14
    SE[70]=sum(v[408:413]) #NOV14
    SE[71]=sum(v[413:417]) #DIC14
    SE[72]=sum(v[417:421]) #ENE15
    SE[73]=sum(v[421:425]) #FEB15
    SE[74]=sum(v[425:430]) #MAR15
    SE[75]=sum(v[430:434]) #ABR15
    SE[76]=sum(v[434:439]) #MAY15
    SE[77]=sum(v[439:443]) #JUN15
    SE[78]=sum(v[443:447]) #JUL15
    SE[79]=sum(v[447:452]) #AGO15
    SE[80]=sum(v[452:456]) #SEP15
    SE[81]=sum(v[456:460]) #OCT15
    SE[82]=sum(v[460:465]) #NOV15
    SE[83]=sum(v[465:469]) #DIC15
    SE[84]=sum(v[469:474]) #ENE16
    SE[85]=sum(v[474:478]) #FEB16
    SE[86]=sum(v[478:482]) #MAR16
    SE[87]=sum(v[482:486]) #ABR16
    SE[88]=sum(v[486:491]) #MAY16
    SE[89]=sum(v[491:495]) #JUN16
    SE[90]=sum(v[495:500]) #JUL16
    SE[91]=sum(v[500:504]) #AGO16
    SE[92]=sum(v[504:508]) #SEP16
    SE[93]=sum(v[508:513]) #OCT16
    SE[94]=sum(v[513:517]) #NOV16
    SE[95]=sum(v[517:521]) #DIC16
    return SE


def orden_years(v,año):
    year_2007=v[0:365]
    year_2008=v[365:731]
    year_2009=v[731:1096]
    year_2010=v[1096:1461]
    year_2011=v[1461:1826]
    year_2012=v[1826:2192]
    year_2013=v[2192:2557]
    year_2014=v[2557:2922]
    year_2015=v[2922:3287]
    year_2016=v[3287:3653]
    year_2017=v[3653:4018]
    
    years=dict(año_2007=year_2007, año_2008=year_2008, año_2009=year_2009, año_2010=year_2010, año_2011=year_2011,
     año_2012=year_2012, año_2013=year_2013, año_2014=year_2014, año_2015=year_2015, año_2016=year_2016, año_2017=year_2017 )
    
    a=str(año)
    b='año_nnnn'
    año_elegido=b.replace('nnnn',a)
    c=years[año_elegido]
    
    return (c)

def orden_years2(v,año):
    year_2007=v[0:52]
    year_2008=v[52:105]
    year_2009=v[105:157]
    year_2010=v[157:209]
    year_2011=v[209:261]
    year_2012=v[261:313]
    year_2013=v[313:365]
    year_2014=v[365:418]
    year_2015=v[418:470]
    year_2016=v[470:522]
    year_2017=v[522:574]
    
    years=dict(año_2007=year_2007, año_2008=year_2008, año_2009=year_2009, año_2010=year_2010, año_2011=year_2011,
     año_2012=year_2012, año_2013=year_2013, año_2014=year_2014, año_2015=year_2015, año_2016=year_2016, año_2017=year_2017 )
    
    a=str(año)
    b='año_nnnn'
    año_elegido=b.replace('nnnn',a)
    c=years[año_elegido]
    
    return (c)

def elegir(v,inicio,fin):
    inicios=dict(año_2001=0, año_2002=365, año_2003=730, año_2004=1095, año_2005=1461, año_2006=1826, 
    año_2007=2191, año_2008=2556, año_2009=2922, año_2010=3287, año_2011=3652,
    año_2012=4017, año_2013=4383, año_2014=4748, año_2015=5113, año_2016=5478, año_2017=5844 )

    fines=dict(año_2001=365, año_2002=730, año_2003=1095, año_2004=1461, año_2005=1826, año_2006=2191, 
    año_2007=2556, año_2008=2922, año_2009=3287, año_2010=3652, año_2011=4017,
    año_2012=4383, año_2013=4748, año_2014=5113, año_2015=5478, año_2016=5844, año_2017=6209)

    a=str(inicio)
    b=str(fin)
    c='año_nnnn'
    año_inicio=c.replace('nnnn',a)
    año_fin=c.replace('nnnn',b)
    d=v[inicios[año_inicio]:fines[año_fin]]

    return(d)


def temporada(v,año1,inicio1,fin1,año2,inicio2,fin2):
    #años bisiestos: 2000, 2004, 2008, 2012, 2016
    inicios=dict(ene=0, feb=31, mar=59, abr=90, may=120, jun=151, jul=181, ago=212, sep=243, oct=273, nov=304, dic=334)
    fines=dict(ene=31, feb=59, mar=90, abr=120, may=151, jun=181, jul=212, ago=243, sep=273, oct=304, nov=334, dic=365)

    inicios_b=dict(ene=0, feb=31, mar=60, abr=91, may=121, jun=152, jul=182, ago=213, sep=244, oct=274, nov=305, dic=335)
    fines_b=dict(ene=31,feb=60, mar=91, abr=121, may=152, jun=182, jul=213, ago=244, sep=274, oct=305, nov=335, dic=366)

    a=str(inicio1)
    b=str(fin1)
    aa=str(inicio2)
    bb=str(fin2)
    c=orden_years(v,año1)
    cc=orden_years(v,año2)
    #print(año, np.size(c))
    d='nnn'
    m1_i=d.replace('nnn',a)
    m1_f=d.replace('nnn',b)
    m2_i=d.replace('nnn',aa)
    m2_f=d.replace('nnn',bb)

    if (año1==2008) or (año1==2012) or (año1==2016):
        f=c[inicios_b[m1_i]:fines_b[m1_f]]
    else:
        f=c[inicios[m1_i]:fines[m1_f]]

    if (año2==2008) or (año2==2012) or (año2==2016):
        ff=cc[inicios_b[m2_i]:fines_b[m2_f]]
    else:
        ff=cc[inicios[m2_i]:fines[m2_f]]

    union=[]
    union[:len(f)]=f[:]
    union[len(f):]=ff[:]
    return(union)



def meses(v,año,inicio,fin):
    #años bisiestos: 2000, 2004, 2008, 2012, 2016
    inicios=dict(ene=0, feb=31, mar=59, abr=90, may=120, jun=151, jul=181, ago=212, sep=243, oct=273, nov=304, dic=334)
    fines=dict(ene=31, feb=59, mar=90, abr=120, may=151, jun=181, jul=212, ago=243, sep=273, oct=304, nov=334, dic=365)

    inicios_b=dict(ene=0, feb=31, mar=60, abr=91, may=121, jun=152, jul=182, ago=213, sep=244, oct=274, nov=305, dic=335)
    fines_b=dict(ene=31,feb=60, mar=91, abr=121, may=152, jun=182, jul=213, ago=244, sep=274, oct=305, nov=335, dic=366)

    a=str(inicio)
    b=str(fin)
    c=orden_years(v,año)
    #print(año, np.size(c))
    d='nnn'
    m_i=d.replace('nnn',a)
    m_f=d.replace('nnn',b)

    if (año==2004) or (año==2008) or (año==2012) or (año==2016):
        d=c[inicios_b[m_i]:fines_b[m_f]]
    else:
        d=c[inicios[m_i]:fines[m_f]]
    return(d)

def orden_years_medio(v,año):
    year_2001=v[0:365]
    year_2002=v[365:730]
    year_2003=v[730:1095]
    year_2004=v[1095:1461]
    year_2005=v[1461:1826]
    year_2006=v[1826:2191]
    year_2007=v[2191:2556]
    year_2008=v[2556:2922]
    year_2009=v[2922:3287]
    year_2010=v[3287:3652]
    year_2011=v[3652:4017]
    year_2012=v[4017:4383]
    year_2013=v[4383:4748]
    year_2014=v[4748:5113]
    year_2015=v[5113:5478]
    year_2016=v[5478:5844]
    year_2017=v[5844:6209]
    
    years=dict(año_2001=year_2001, año_2002=year_2002, año_2003=year_2003, año_2004=year_2004, año_2005=year_2005, año_2006=year_2006, 
    año_2007=year_2007, año_2008=year_2008, año_2009=year_2009, año_2010=year_2010, año_2011=year_2011,
    año_2012=year_2012, año_2013=year_2013, año_2014=year_2014, año_2015=year_2015, año_2016=year_2016, año_2017=year_2017 )
    
    a=str(año)
    b='año_nnnn'
    año_elegido=b.replace('nnnn',a)
    c=years[año_elegido]
    
    return (c)

def casos_anual(v,año):
    ME = np.zeros(12)
    if año == 2009:
        ME[0] = sum(v[105:109])#ENE
        ME[1] = sum(v[109:113])#FEB
        ME[2] = sum(v[113:118])#MAR
        ME[3] = sum(v[118:122])#ABR
        ME[4] = sum(v[122:127])#MAY
        ME[5] = sum(v[127:131])#JUN
        ME[6] = sum(v[131:135])#JUL
        ME[7] = sum(v[135:140])#AGO
        ME[8] = sum(v[140:144])#SEP
        ME[9] = sum(v[144:149])#OCT
        ME[10] = sum(v[149:153])#NOV
        ME[11] = sum(v[153:157])#DIC
    
    if año == 2010:
        ME[0] = sum(v[157:161])#ENE
        ME[1] = sum(v[161:165])#FEB
        ME[2] = sum(v[165:170])#MAR
        ME[3] = sum(v[170:174])#ABR
        ME[4] = sum(v[174:178])#MAY
        ME[5] = sum(v[178:183])#JUN
        ME[6] = sum(v[183:187])#JUL
        ME[7] = sum(v[187:192])#AGO
        ME[8] = sum(v[192:196])#SEP
        ME[9] = sum(v[196:200])#OCT
        ME[10] = sum(v[200:204])#NOV
        ME[11] = sum(v[204:209])#DIC

    if año == 2011:
        ME[0] = sum(v[209:214])#ENE
        ME[1] = sum(v[214:218])#FEB
        ME[2] = sum(v[218:222])#MAR
        ME[3] = sum(v[222:226])#ABR
        ME[4] = sum(v[226:231])#MAY
        ME[5] = sum(v[231:235])#JUN
        ME[6] = sum(v[235:240])#JUL
        ME[7] = sum(v[240:244])#AGO
        ME[8] = sum(v[244:248])#SEP
        ME[9] = sum(v[248:253])#OCT
        ME[10] = sum(v[253:257])#NOV
        ME[11] = sum(v[257:261])#DIC
    
    if año == 2012:
        ME[0] = sum(v[261:266])#ENE
        ME[1] = sum(v[266:270])#FEB
        ME[2] = sum(v[270:274])#MAR
        ME[3] = sum(v[274:279])#ABR
        ME[4] = sum(v[279:283])#MAY
        ME[5] = sum(v[283:287])#JUN
        ME[6] = sum(v[287:292])#JUL
        ME[7] = sum(v[292:296])#AGO
        ME[8] = sum(v[296:301])#SEP
        ME[9] = sum(v[301:305])#OCT
        ME[10] = sum(v[305:309])#NOV
        ME[11] = sum(v[309:313])#DIC
    
    if año == 2013:
        ME[0] = sum(v[313:318])#ENE
        ME[1] = sum(v[318:322])#FEB
        ME[2] = sum(v[322:327])#MAR
        ME[3] = sum(v[327:331])#ABR
        ME[4] = sum(v[331:335])#MAY
        ME[5] = sum(v[335:340])#JUN
        ME[6] = sum(v[340:344])#JUL
        ME[7] = sum(v[344:348])#AGO
        ME[8] = sum(v[348:353])#SEP
        ME[9] = sum(v[353:357])#OCT
        ME[10] = sum(v[357:361])#NOV
        ME[11] = sum(v[361:365])#DIC
    
    if año == 2014:
        ME[0] = sum(v[365:370])#ENE
        ME[1] = sum(v[370:374])#FEB
        ME[2] = sum(v[374:379])#MAR
        ME[3] = sum(v[379:383])#ABR
        ME[4] = sum(v[383:387])#MAY
        ME[5] = sum(v[387:392])#JUN
        ME[6] = sum(v[392:396])#JUL
        ME[7] = sum(v[396:401])#AGO
        ME[8] = sum(v[401:405])#SEP
        ME[9] = sum(v[405:409])#OCT
        ME[10] = sum(v[409:414])#NOV
        ME[11] = sum(v[414:418])#DIC
    
    if año == 2015:
        ME[0] = sum(v[418:422])#ENE
        ME[1] = sum(v[422:426])#FEB
        ME[2] = sum(v[426:431])#MAR
        ME[3] = sum(v[431:435])#ABR
        ME[4] = sum(v[435:440])#MAY
        ME[5] = sum(v[440:444])#JUN
        ME[6] = sum(v[444:448])#JUL
        ME[7] = sum(v[448:453])#AGO
        ME[8] = sum(v[453:457])#SEP
        ME[9] = sum(v[457:461])#OCT
        ME[10] = sum(v[461:466])#NOV
        ME[11] = sum(v[466:470])#DIC
    
    if año == 2016:
        ME[0] = sum(v[470:475])#ENE
        ME[1] = sum(v[475:479])#FEB
        ME[2] = sum(v[479:483])#MAR
        ME[3] = sum(v[483:487])#ABR
        ME[4] = sum(v[487:492])#MAY
        ME[5] = sum(v[492:496])#JUN
        ME[6] = sum(v[496:501])#JUL
        ME[7] = sum(v[501:505])#AGO
        ME[8] = sum(v[505:509])#SEP
        ME[9] = sum(v[509:514])#OCT
        ME[10] = sum(v[514:518])#NOV
        ME[11] = sum(v[518:522])#DIC

    return (ME)