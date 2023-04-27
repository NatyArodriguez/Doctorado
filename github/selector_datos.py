import numpy as np
import matplotlib.pyplot as plt

def readinputdata(filename):
    fichero=open(filename,'r')
    f=[]
    line='0'
    while len(line)>0:
        line=np.array(fichero.readline().split()).astype(float)
        if len(line)>0:
            f.append(line)
    fichero.close()
    return np.array(f)

#valor=readinputdata('Oran_2001_2017.txt')
#ejemplo=valor[:,0]
#fecha_inicio=[1,'dic',2007]
#fecha_fin=[31,'may',2008]

def seleccion(datos,v_i,v_f):
    #### separo por años
    year=dict(
        a_2001=0,
        a_2002=365,
        a_2003=730,
        a_2004=1095,
        a_2005=1461,
        a_2006=1826,
        a_2007=2191,
        a_2008=2556,
        a_2009=2922,
        a_2010=3287,
        a_2011=3652,
        a_2012=4017,
        a_2013=4383,
        a_2014=4748,
        a_2015=5113,
        a_2016=5478,
        a_2017=5844)
    
    #### separo por meses
    m=dict(
        ene=0,
        feb=31,
        mar=59,
        abr=90,
        may=120,
        jun=151,
        jul=181,
        ago=212,
        sep=243,
        oct=273,
        nov=304,
        dic=334)

    mb=dict(
        ene=0,
        feb=31,
        mar=60,
        abr=91,
        may=121,
        jun=152,
        jul=182,
        ago=213,
        sep=244,
        oct=274,
        nov=305,
        dic=335)

    bisiesto=[2004,2008,2012,2016]

    mmi=str(v_i[1])
    aaaai=str(v_i[2])

    mmf=str(v_f[1])
    aaaaf=str(v_f[2])

    c='a_nnnn'
    a_inicio=c.replace('nnnn',aaaai)
    a_final=c.replace('nnnn',aaaaf)

    cc='nnn'
    mes_i=cc.replace('nnn',mmi)

    ccc='nnn'
    mes_f=ccc.replace('nnn',mmf)

    aux=year[a_final]

    if (v_f[2] in bisiesto):
        aux1 = mb[mes_f]
    else:
        aux1=m[mes_f]
    
    corte = aux + aux1 + (v_f[0])
    primero = datos[year[a_inicio]:corte]

    if (v_i[2] in bisiesto):
        segundo = primero[mb[mes_i]:]
        dia = v_i[0] - 1
        tercero = segundo[dia:]
    else:
        segundo = primero[m[mes_i]:]
        dia = v_i[0] - 1
        tercero = segundo[dia:]

    return(tercero)

"""
def seleccion(datos,vector_i,vector_f):
    #### separo por años
    year_i=dict(
        a_2001=0,
        a_2002=365,
        a_2003=730,
        a_2004=1095,
        a_2005=1461,
        a_2006=1826,
        a_2007=2191,
        a_2008=2556,
        a_2009=2922,
        a_2010=3287,
        a_2011=3652,
        a_2012=4017,
        a_2013=4383,
        a_2014=4748,
        a_2015=5113,
        a_2016=5478,
        a_2017=5844)

    year_f=dict(
        a_2001=365,
        a_2002=730,
        a_2003=1095,
        a_2004=1461,
        a_2005=1826,
        a_2006=2191,
        a_2007=2556,
        a_2008=2922,
        a_2009=3287,
        a_2010=3652,
        a_2011=4017,
        a_2012=4383,
        a_2013=4748,
        a_2014=5113,
        a_2015=5478,
        a_2016=5844,
        a_2017=6209)
    
    #### separo por meses
    m_i=dict(
        ene=0,
        feb=31,
        mar=59,
        abr=90,
        may=120,
        jun=151,
        jul=181,
        ago=212,
        sep=243,
        oct=273,
        nov=304,
        dic=334)

    m_f=dict(
        ene=31,
        feb=59,
        mar=90,
        abr=120,
        may=151,
        jun=181,
        jul=212,
        ago=243,
        sep=273,
        oct=304,
        nov=334,
        dic=365)

    mb_i=dict(
        ene=0,
        feb=31,
        mar=60,
        abr=91,
        may=121,
        jun=152,
        jul=182,
        ago=213,
        sep=244,
        oct=274,
        nov=305,
        dic=335)

    mb_f=dict(
        ene=31,
        feb=60,
        mar=91,
        abr=121,
        may=152,
        jun=182,
        jul=213,
        ago=244,
        sep=274,
        oct=305,
        nov=335,
        dic=366)

    ddi=str(vector_i[0])
    mmi=str(vector_i[1])
    aaaai=str(vector_i[2])

    ddf=str(vector_f[0])
    mmf=str(vector_f[1])
    aaaaf=str(vector_f[2])

    c='a_nnnn'
    a_inicio=c.replace('nnnn',aaaai)
    a_final=c.replace('nnnn',aaaaf)

    ccc='nnn'
    mes_f=ccc.replace('nnn',mmf)

    aux=year_i[a_final]
    print(aux)

    if (vector_f[2]==2004) or (vector_f[2]==2008) or (vector_f[2]==2012) or (vector_f[2]==2016):
        #print('algo anda mal')
        aux1=mb_i[mes_f]
        print(aux1)
    else:
        #print('todo bien')
        aux1=m_i[mes_f]

    corte=aux+aux1+(vector_f[0])
    print(corte)

    primero=datos[year_i[a_inicio]:corte]
    
    cc='nnn'
    mes_i=cc.replace('nnn',mmi)

    if (vector_i[2]==2004) or (vector_i[2]==2008) or (vector_i[2]==2012) or (vector_i[2]==2016):
        segundo=primero[mb_i[mes_i]:]
        #print('meti mal')
    else:
        segundo=primero[m_i[mes_i]:]
        #print('todo mal')
    
    dia_i= vector_i[0]-1
    tercero=segundo[dia_i:]

    return(tercero)
"""

#final=seleccion(ejemplo,fecha_inicio,fecha_fin)
#print(final)
#print(np.size(final))