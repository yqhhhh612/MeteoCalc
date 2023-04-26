'''
1.根据P,T,Td,计算假相当位温（Qse）
2.根据P,T,计算位温（Qp）
3.根据P,T,Td,Z计算地面大气总能量（Ttdm）
4.根据P,T,Td,Z,V计算地面大气总能量（Ttgk）
5.根据P,Td计算空气比湿（qs）
6.根据P,Td计算凝结函数（Fc）
7.根据T,Td计算空气相对湿度（rh）
8.根据P,T计算湿绝热温度直减率
9.根据P,Td,V计算湿绝热温度直减率
'''
import numpy as np
import math
import metpy.calc as mi
from metpy.units import units
def td_func(t,rh,a=17.27,b=237.7):
    '''根据温度和相对湿度计算露点'''
    import numpy as np
    gama = (a*t)/(b+t)+np.log(rh/100)
    td = (b*gama)/(a-gama)
    return td

def E_t(t):
    '''温度计算饱和水汽压'''
    t = t+273.15
    if t >= 273.15:
        esa = 6.1078 * np.exp((17.2693882 * (t - 273.16)) / (t - 35.86))  # liquid, Tetens
    elif t < 273.15:
        esa = 6.1078 * np.exp((21.8745884 * (t - 273.16)) / (t - 7.66))  # ice, Tetens
    else:
        esa = np.nan
    # < 0 C, liquid:
    # esa=6.112*np.exp(17.67*(t-273.15)/(t-273.15+243.5))  # liquid < 0 C, Bolton
    return (esa)
def E_water(td):
    '''水面饱和水汽压'''
    E0 = 6.1078
    T0 = 273.16
    Cl = 0.57
    Rw = 0.1101787372
    L0 = 597.4
    T = T0+td
    E = ((L0+Cl*T0)*(T-T0))/(Rw*T0*T)
    E_new = math.exp(E)
    E_end = E_new*E0*np.power(T0/T,Cl/Rw)
    return E_end
def E_ice(td):
    E0 = 6.1078
    T0 = 273.16
    Cf = 0.06
    Rw = 0.1101787372
    L0 = 597.4
    Lf = 79.72
    T = td+T0
    Ls = L0+Lf-Cf*(T-T0)
    E = ((Ls+Cf*T0)*(T-T0))/(Rw*T0*T)
    E =math.exp(E)
    E = E*E0*math.pow(T0/T,Cf/Rw)
    return E

def Tc(P,t,td):
    '''凝结高度的温度'''
    Cpd = 0.2403
    Cpv = 0.445
    Rd = 6.85578*0.01
    Rw = 11.017874*0.01
    T0 = 273.16
    step = 10
    T = t+T0
    Td = td+T0
    Etd = E_water(td)
    w = (Rd/Rw)*Etd/(-Etd)
    m1 = (Cpd*(1+Cpv*w/Cpd))/(Rd*(1+w/(Rd/Rw)))
    Z0 = math.pow(T,m1)/Etd
    out = Td
    Z = math.pow(out,m1)/E_water(out-T0)
    while(abs(Z-Z0)>10):
        if (Z<Z0):
            out = out-step
        else:
            out = out+step
            step = step/5
            out = out-step
        Z = math.pow(out,m1)/E_water(out-T0)
        if step<0.0000001:
            break
    return out-T0

def Qse_cal(P,t,td):
    '''假相当位温'''
    Rd = 6.8557782*0.01
    Cpd = 0.24
    L0 = 597.4
    Cl = 0.57
    T0 = 273.16
    T = T0+t
    Kd = Rd/Cpd
    tc = Tc(P,t,td)+T0
    Lc = L0-Cl*(tc-T0)
    E = E_water(td)
    w = 0.622*E/(P-E)
    qse = T*math.pow((1000/(P-E)),Kd)*math.exp((Lc*w)/(Cpd*tc))
    return qse-T0

def Qp_cal(P,t):
    '''位温'''
    Rd = 6.8557782*0.01
    Cpd = 0.24
    Kd = Rd/Cpd
    T = t+273.16
    out = T*math.pow(1000/P,Kd)
    return out

def Ttdm_cal(P,t,td,z):
    '''地面空气总能量'''
    A = 2.38844*10**(-8)
    g = 980.665
    Cpd = 0.2403
    L0 = 597.4
    Cl = 0.57
    E = E_water(td)
    Tz = 100*A*g*z/Cpd
    L = L0-Cl*td
    q = 0.622*E/(P-0.378*E)
    Tl = L*q/Cpd
    Tt = t+Tz+Tl
    return Tt
def Qgk(p,td):
    E = E_water(td)
    out = 622*E/(p-0.378*E)
    return out


def Ttgk_cal(p,t,td,z,v):
    '''高空大气总能量'''
    A = 2.38844*10**(-8)
    g = 980.665
    Cpd = 0.2403
    L0 = 597.4
    Cl = 0.57
    L = L0 - Cl * td
    q = Qgk(p,td)/1000
    out = t+1000*A*g*z/Cpd+L*q/Cpd+A*v*v/(2*Cpd)
    return out

def Etotd(E):
    '''水汽压模拟反算露点'''
    td = -60
    step = 20
    e = E_water(td)
    while True:
        if e>E:
            step = step/2
            td = td-step
        else:
            td = td+step
        e = E_water(td)
        if abs(e-E)>0.0001:
            break
    return td

def rhgk(t,td):
    '''温度和露点计算高空相对湿度'''
    Et = E_water(t)
    Etd = E_water(td)
    out = (Etd/Et)/100
    return out
def Func_coagulation(p,td):
    T0 = 273.16
    L0 = 597.4
    Cl = 0.57
    Cpd = 0.2403
    Rw = 11.017874*0.01
    Rd = 6.85578*0.01

    Td = T0+td
    Etd = E_water(td)
    qs = 1000*Rd*Etd/(p-0.378*Etd)/Rw
    L = L0-Cl*(Td-T0)
    Fcd = (qs/(p-0.378*Etd))*(Rd*L/(Cpd*Rw*Td)-1)
    out = 100*Fcd
    out = out/(1+(L*L*qs*0.01/(Cpd*Rw*Td*Td))*(p/(p-0.378*Etd)))
    return out
def Vapor_flux(p,td,v):
    g = 980.665*0.01
    q = Qgk(p,td)
    out = v*q/g
    return out

def wet_Lapse_rate(p,t):
    '''计算湿绝热温度直减率'''
    T0 = 273.16
    Rd = 6.85578 * 0.01
    Cpd = 0.2403
    rd = 9.76
    T = t+T0
    L = 597.4-0.57*t
    E = E_water(t)
    w = 0.622*E/(p-E)
    q = 0.622*E/(p-0.378*E)
    R = Rd*(1+0.608*q)
    out = rd*(1+(L*w/(R*T)))
    out = out/(1+(0.622*L*L*w/(Cpd*Rd*T*T)))
    return out

