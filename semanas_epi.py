import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from math import *


def orden(v):
    SE=np.zeros(84)
    SE[0]=sum(v[156:161]) #ENE10
    SE[1]=sum(v[161:165]) #FEB10
    SE[2]=sum(v[165:169]) #MAR10
    SE[3]=sum(v[169:173]) #ABR10
    SE[4]=sum(v[173:178]) #MAY10
    SE[5]=sum(v[178:182]) #JUN10
    SE[6]=sum(v[182:186]) #JUL10
    SE[7]=sum(v[186:191]) #AGO10
    SE[8]=sum(v[191:195]) #SEP10
    SE[9]=sum(v[195:200]) #OCT10
    SE[10]=sum(v[200:204]) #NOV10
    SE[11]=sum(v[204:208]) #DIC10
    SE[12]=sum(v[208:213]) #ENE11
    SE[13]=sum(v[213:217]) #FEB11
    SE[14]=sum(v[217:221]) #MAR11
    SE[15]=sum(v[221:225]) #ABR11
    SE[16]=sum(v[225:230]) #MAY11
    SE[17]=sum(v[230:234]) #JUN11
    SE[18]=sum(v[234:239]) #JUL11
    SE[19]=sum(v[239:243]) #AGO11
    SE[20]=sum(v[243:247]) #SEP11
    SE[21]=sum(v[247:252]) #OCT11
    SE[22]=sum(v[252:256]) #NOV11
    SE[23]=sum(v[256:260]) #DIC11
    SE[24]=sum(v[260:265]) #ENE12
    SE[25]=sum(v[265:269]) #FEB12
    SE[26]=sum(v[269:273]) #MAR12
    SE[27]=sum(v[273:278]) #ABR12
    SE[28]=sum(v[278:282]) #MAY12
    SE[29]=sum(v[282:286]) #JUN12
    SE[30]=sum(v[286:291]) #JUL12
    SE[31]=sum(v[291:295]) #AGO12
    SE[32]=sum(v[295:300]) #SEP12
    SE[33]=sum(v[300:304]) #OCT12
    SE[34]=sum(v[304:308]) #NOV12
    SE[35]=sum(v[308:312]) #DIC12
    SE[36]=sum(v[312:317]) #ENE13
    SE[37]=sum(v[317:321]) #FEB13
    SE[38]=sum(v[321:326]) #MAR13
    SE[39]=sum(v[326:330]) #ABR13
    SE[40]=sum(v[330:334]) #MAY13
    SE[41]=sum(v[334:339]) #JUN13
    SE[42]=sum(v[339:343]) #JUL13
    SE[43]=sum(v[343:347]) #AGO13
    SE[44]=sum(v[347:352]) #SEP13
    SE[45]=sum(v[352:356]) #OCT13
    SE[46]=sum(v[356:360]) #NOV13
    SE[47]=sum(v[360:364]) #DIC13
    SE[48]=sum(v[364:369]) #ENE14
    SE[49]=sum(v[369:373]) #FEB14
    SE[50]=sum(v[373:378]) #MAR14
    SE[51]=sum(v[378:382]) #ABR14
    SE[52]=sum(v[382:386]) #MAY14
    SE[53]=sum(v[386:391]) #JUN14
    SE[54]=sum(v[391:395]) #JUL14
    SE[55]=sum(v[395:400]) #AGO14
    SE[56]=sum(v[400:404]) #SEP14
    SE[57]=sum(v[404:408]) #OCT14
    SE[58]=sum(v[408:413]) #NOV14
    SE[59]=sum(v[413:417]) #DIC14
    SE[60]=sum(v[417:421]) #ENE15
    SE[61]=sum(v[421:425]) #FEB15
    SE[62]=sum(v[425:430]) #MAR15
    SE[63]=sum(v[430:434]) #ABR15
    SE[64]=sum(v[434:439]) #MAY15
    SE[65]=sum(v[439:443]) #JUN15
    SE[66]=sum(v[443:447]) #JUL15
    SE[67]=sum(v[447:452]) #AGO15
    SE[68]=sum(v[452:456]) #SEP15
    SE[69]=sum(v[456:460]) #OCT15
    SE[70]=sum(v[460:465]) #NOV15
    SE[71]=sum(v[465:469]) #DIC15
    SE[72]=sum(v[469:474]) #ENE16
    SE[73]=sum(v[474:478]) #FEB16
    SE[74]=sum(v[478:482]) #MAR16
    SE[75]=sum(v[482:486]) #ABR16
    SE[76]=sum(v[486:491]) #MAY16
    SE[77]=sum(v[491:495]) #JUN16
    SE[78]=sum(v[495:500]) #JUL16
    SE[79]=sum(v[500:504]) #AGO16
    SE[80]=sum(v[504:508]) #SEP16
    SE[81]=sum(v[508:513]) #OCT16
    SE[82]=sum(v[513:517]) #NOV16
    SE[83]=sum(v[517:521]) #DIC16
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



"""
def elegir(v,year):
    year_1=v[0:3]
    year_2=v[3:6]
    years=dict(año_1=year_1,año_2=year_2)
    a=str(year)
    b='año_n'
    eleccion=b.replace('n',a)
    c=years[eleccion]
    return (c)

vector=[1,2,3,4,5,6]

ver=elegir(vector,2)

print(ver)



v1=[1,2,3]
v2=[4,5,6]

z=1

a=str(z)

b='vector_n'

eleccion=b.replace('n',a)



vectores=dict(vector_1=v1,vector_2=v2)

#vectores=[v1,v2]

#print(vectores[eleccion])
"""