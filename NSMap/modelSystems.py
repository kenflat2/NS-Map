import numpy as np
import numpy.linalg as la
import numpy.random as rand
from scipy.integrate import odeint

## Models ##

def Logistic(x, t):
    r = 4
    return r * x * (1-x)

def LogisticP(x, t, r, k = lambda t: 1):
    return r(t) * x * (1-x/k(t))

def FoodChain(xi,t):
    (x,y,z)=xi


    a1 = 5
    a2 = 0.1
    b1 = 3
    b2 = 2
    d1 = 0.4
    d2 = 0.01

    dx = x*(1-x)- a1*x*y/(1+b1*x)
    dy = a1*x*y/(1 + b1*x) - d1*y - a2*y*z/(1 + b2*y)
    dz = a2*y*z/(1 + b2*y) - d2*z

    return dx, dy, dz

def FoodChainP(xi, t, b1):
    (x,y,z)=xi


    a1 = 5
    a2 = 0.1
    b1 = b1(t)
    b2 = 2
    d1 = 0.4
    d2 = 0.01

    dx = x*(1-x)- a1*x*y/(1+b1*x)
    dy = a1*x*y/(1 + b1*x) - d1*y - a2*y*z/(1 + b2*y)
    dz = a2*y*z/(1 + b2*y) - d2*z

    return dx, dy, dz

# just one function that should take care of all my integrating needs
def generateTimeSeriesContinuous(f, t0, tlen=256, end=32, reduction=1, settlingTime=0, nsargs=None, process_noise=0):
    F = globals()[f]

    if settlingTime > 0:
        tSettle = np.arange(0,settlingTime, step=end/(reduction*tlen))

        # let the system settle
        if nsargs == None:
            x0 = odeint(F, t0, tSettle)[-1]
        else:
            driver_settle_list = []
            for xxx in nsargs:
                if type(xxx) is int or type(xxx) is float:
                    driver_settle_list.append(xxx)
                else:
                    initial_param_value = xxx(0)
                    driver_settle_list.append(lambda _: initial_param_value)

            driver_settle = tuple(driver_settle_list)
            x0 = odeint(F, t0, tSettle, args=driver_settle)[-1]
    else:
        x0 = t0
    
    t = np.linspace(0,end,num=tlen*reduction)
    ts = np.zeros((tlen,len(x0)))
    ts[0] = x0

    if nsargs == None:
        for i in range(tlen-1):
            ts[i+1] = odeint(F, ts[i], t[i*reduction:(i+1)*reduction])[-1] * np.exp(rand.normal(0,process_noise))
    else:
        for i in range(tlen-1):
            ts[i+1] = odeint(F, ts[i], t[i*reduction:(i+1)*reduction],args=nsargs)[-1] * np.exp(rand.normal(0,process_noise))

    return ts

def generateTimeSeriesDiscrete(f, t0, tlen=256, settlingTime=0, nsargs=None, process_noise=0):
    F = globals()[f]
    
    if type(t0) == float:
        ts = np.zeros((tlen,1))
    else:
        ts = np.zeros((tlen,t0.shape[0]))

    ts[0] = t0

    # allow system to settle
    for i in range(settlingTime):
        if nsargs==None:
            ts[0] = F(ts[0], 0)
        else:
            ts[0] = F(ts[0], 0, *nsargs)

    # now evaluate
    if nsargs==None:
        for i in range(1,tlen):
            ts[i] = F(ts[i-1], i) * np.exp(process_noise*rand.normal(0,1))
    else:
        for i in range(1,tlen):
            ts[i] = F(ts[i-1], i, *nsargs) * np.exp(process_noise*rand.normal(0,1))

    return ts

# The Logistic Map with process noise is generated here because the way process noise is applied
# is unusual and doesn't fit elegently in the functions defined above.
#   Parameters
#       x0 - scalar initial condition
#       tlen - length of the time series
#       r - function mapping [0,1,2,...,tlen-1] to values of r
#       process_noise - standard deviation of the process noise draws
#   Returns
#       The time series with shape (tlen, 1)

def generateLogisticMapProcessNoise(x0 = np.pi / 4, tlen = 200, r = lambda t: 4, process_noise = 0.0):
    ts = np.zeros(tlen)
    ts[0] = x0

    for i in range(tlen-1):
        t = i / (tlen - 1)
        x = r(t) * ts[i] * (1 - ts[i])
        u = np.log(x / (1 - x))
        z = rand.normal(0, process_noise)
        ts[i+1] = 1 / (1 + np.exp(z - u))

    return ts[:,None]