import sys
import inspect # Use this if you want to print line numbers
import operator # Used for dictionary sorting
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Standard Model parameters
theta12 = np.arcsin(np.sqrt(0.306))
theta13 = np.arcsin(np.sqrt(0.0251))
theta23 = np.arcsin(np.sqrt(0.42))

SMdmsq21 = 7.58E-5
SMposdmsq31 = 2.32E-3

# Values from the wikipedia Neutrino Oscillation page.
wikitheta12 = 0.5*np.arcsin(np.sqrt(0.861))
wikitheta13 = 0.5*np.arcsin(np.sqrt(0.1))
wikitheta23 = 0.5*np.arcsin(np.sqrt(0.97))

wikidmsq21 = 7.59E-5
wikiposdmsq31 = 2.32E-3

deltacp = 0
alpha21 = 0
alpha31 = 0


##########################################################################
def PMNSmatrix(th13 = theta13, th12 = theta12, th23 = theta23, dcp = deltacp,
               al21 = alpha21, al31 = alpha31):
    """
    Return PMNS matrix with specified mixing angles and complex phases.
    """

    # Mixing matrices and useful trig. definitions
    c12 = np.cos(th12)
    c13 = np.cos(th13)
    c23 = np.cos(th23)
    s12 = np.sin(th12)
    s13 = np.sin(th13)
    s23 = np.sin(th23)

    #    frame = inspect.currentframe()
    #    info = inspect.getframeinfo(frame)
    #    print "line %s" % info.lineno
    ecp = np.exp(dcp*1j)
    if np.absolute(ecp) != 1.:
        print 'deltaCP = %s an invalid value. ||deltaCP|| not equal to one!' % dcp
        print 'Exiting!\n'
        sys.exit(1)

    U12 = np.identity(3) + 0j # Add 0j to cast as a complex 2D array
    U13 = np.identity(3) + 0j
    U23 = np.identity(3) + 0j
    Maj = np.identity(3) + 0j

    U12[0,0] = c12; U12[0,1] = s12
    U12[1,0] = -s12; U12[1,1] = c12

    U13[0,0] = c13; U13[0,2] = s13*(1./ecp)
    U13[2,0] = -s13*ecp; U13[2,2] = c13

    U23[1,1] = c23; U23[1,2] = s23
    U23[2,1] = -s23; U23[2,2] = c23

    Maj[1,1] = np.exp(al21*1j/2); Maj[2,2] = np.exp(al31*1j/2)

    return np.dot(np.dot(np.dot(U23,U13),U12),Maj)


##########################################################################
# 'anti' bool is for antineutrinos; 'invhier' bool is for inverted hierarchy
def vacuumOscProb(startflavor, endflavor, LoverE, anti = False, 
                  invhier = False, pmns = PMNSmatrix(), dmsq21 = SMdmsq21, 
                  posdmsq31 = SMposdmsq31):
    flavors = {'electron': 0, 'e': 0, 'm': 1, 'mu': 1, 'muon': 1, 'tau': 2,
               't': 2}

    if startflavor.lower() not in flavors:
        print '"%s" not a valid starting flavor. Valid options are: \n' % \
            startflavor
        for key in sorted(flavors.iterkeys()):
            print key,             
        print '\n\nExiting.\n'
        sys.exit(1)

    if endflavor.lower() not in flavors:
        print '"%s" not a valid ending flavor. Valid options are: \n' % endflavor
        for key in sorted(flavors.iterkeys()):
            print key,             
        print '\n\nExiting.\n'
        sys.exit(1)

    f1 = flavors[startflavor.lower()]
    f2 = flavors[endflavor.lower()]
    constpart = LEpart = dmsq = dmsq31 = CPphase = 0

    # Deal with hierarchy
    if invhier: dmsq31 = -posdmsq31
    else: dmsq31 = posdmsq31

    dmsq32 = dmsq31 - dmsq21

    for j in range(3):
        for k in range(3):
            prod = pmns[f2,j]*(pmns[f1,j].conj())*pmns[f1,k]*(pmns[f2,k].conj())

            if j == k:
                constpart += prod
            else:
                if (j == 0 and k == 1): dmsq = -dmsq21
                elif (j == 1 and k == 0): dmsq = dmsq21
                elif (j == 2 and k == 0): dmsq = dmsq31
                elif (j == 0 and k == 2): dmsq = -dmsq31
                elif (j == 2 and k == 1): dmsq = dmsq32
                else: dmsq = -dmsq32

                if anti: CPphase = -np.angle(prod)
                else: CPphase = np.angle(prod)

                LEpart += abs(prod)*np.cos(2*dmsq*(LoverE)*1.267 - CPphase)

    return constpart + LEpart


##########################################################################
# 'anti' bool is for antineutrinos    
def testplot(startflavor, endflavor, anti = False, lowerlim=10, upperlim=9e4):
    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)/10)

    ynorm = vacuumOscProb(startflavor, endflavor, x, anti)    
    normlabel = ''
    if anti: normlabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Normal hierarchy)' % (startflavor, endflavor)
    else: normlabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Normal hierarchy)' % (startflavor, endflavor)

    yinv = vacuumOscProb(startflavor, endflavor, x, anti, True)    
    invlabel = ''
    if anti: invlabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Inverted hierarchy)' % (startflavor, endflavor)
    else: invlabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Inverted hierarchy)' % (startflavor, endflavor)

    pmnsZero = PMNSmatrix(0)
    yzero = vacuumOscProb(startflavor, endflavor, x, anti, False, pmnsZero)    
    zerolabel = ''
    if anti: zerolabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Inverted hierarchy, $\theta_{13} = 0$)' % (startflavor, endflavor)
    else: zerolabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Inverted hierarchy, $\theta_{13} = 0$)' % (startflavor, endflavor)

    fig, ax = plt.subplots()

    ax.plot(x, ynorm, linewidth=2, label=normlabel)    
    ax.plot(x, yinv, linewidth=2, label=invlabel)
    ax.plot(x, yzero, linewidth=2, label=zerolabel)
    ax.set_xscale('log')
    ax.grid()
    ax.legend(loc=6) # Create legend in center-left 
    plt.axis([lowerlim, upperlim, 0, 1]) # x-axis and y-axis ranges
    plt.xlabel(r'$L/E\ \mathrm{[m/MeV]}$')
    plt.show()


##########################################################################   
# 'invhier' bool is for inverted hierarchy
def nuVSantinu_prob(startflavor, endflavor, dcp = deltacp, lowerlim=10, 
                    upperlim=9e4, invhier = False):
    """
    Plot oscillation probabilities for neutrinos and antineutrinos for a 
    specified value of deltaCP. Also plot oscillation probability for 
    neutrinos with deltaCP = 0. Optionally specify mass hierarchy with
    'invhier' boolean (default hierarchy: normal).
    """
    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)/10)
    pmns = PMNSmatrix(theta13, theta12, theta23, dcp)

    # Make neutrino plot and label for CP phase of dcp
    ynu = vacuumOscProb(startflavor, endflavor, x, False, invhier, pmns)
    nulabel = ''
    if invhier: nulabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Inverted hierarchy, $\delta_{CP} = %s$)' % (startflavor, endflavor, dcp)
    else: nulabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Normal hierarchy, $\delta_{CP} = %s$)' % (startflavor, endflavor, dcp)

    # Make antineutrino plot and label for CP phase of dcp
    yantinu = vacuumOscProb(startflavor, endflavor, x, True, invhier, pmns)
    antilabel = ''
    if invhier: antilabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Inverted hierarchy, $\delta_{CP} = %s$)' % (startflavor, endflavor, dcp)
    else: antilabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Normal hierarchy, $\delta_{CP} = %s$)' % (startflavor, endflavor, dcp)

    # Make neutrino plot and label for CP phase of zero
    pmnsZero = PMNSmatrix(theta13, theta12, theta23, 0)
    yzero = vacuumOscProb(startflavor, endflavor, x, False, invhier, pmnsZero)
    zerolabel = ''
    if invhier: zerolabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Inverted hierarchy, $\delta_{CP} = 0$)' % (startflavor, endflavor)
    else: zerolabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Normal hierarchy, $\delta_{CP} = 0$)' % (startflavor, endflavor)

    fig, ax = plt.subplots()

    ax.plot(x, ynu, linewidth=2, label=nulabel)    
    ax.plot(x, yantinu, '--', linewidth=2, label=antilabel)
    ax.plot(x, yzero, '-.', linewidth=2, label=zerolabel)
    ax.set_xscale('log')
    ax.grid()
    ax.legend(loc=6) # Create legend in center-left 
    plt.axis([lowerlim, upperlim, 0, 1]) # x-axis and y-axis ranges
    plt.xlabel(r'$L/E\ \mathrm{[m/MeV]}$')
    plt.show()


##########################################################################   
# 'anti' bool is for antineutrinos
def normVSinvHier_prob(startflavor, endflavor, lowerlim=10, upperlim=9e4, 
                       anti = False):
    """
    Plot oscillation probabilities for neutrinos (or antineutrinos) for 
    normal and inverted hierarchies. If 'anti' is true, antineutrinos are
    generated.
    """
    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)*100.)
    pmns = PMNSmatrix()

    # Make neutrino plot and label for CP phase of dcp
    ynorm = vacuumOscProb(startflavor, endflavor, 1000.*x, anti, False, pmns)
    normlabel = ''
    if anti: normlabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Normal hierarchy)' % (startflavor, endflavor)
    else: normlabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Normal hierarchy)' \
            % (startflavor, endflavor)

    # Make antineutrino plot and label for CP phase of dcp
    yinv = vacuumOscProb(startflavor, endflavor, 1000.*x, anti, True, pmns)
    invlabel = ''
    if anti: invlabel = r'$P_{\overline{\nu}_{%s} \rightarrow \overline{\nu}_{%s}}$ (Inverted hierarchy)' % (startflavor, endflavor)
    else: invlabel = r'$P_{\nu_{%s} \rightarrow \nu_{%s}}$ (Inverted hierarchy)'\
            % (startflavor, endflavor)

    fig, ax = plt.subplots()

    ax.plot(x, ynorm, linewidth=2, label=normlabel)    
    ax.plot(x, yinv, linewidth=2, label=invlabel)
    ax.set_xscale('log')
    ax.grid()
#    ax.legend(loc=6) # Create legend in center-left 
#    ax.legend(loc=9) # Create legend in upper-center
    ax.legend(loc=3) # Create legend in lower-left
    plt.axis([lowerlim, upperlim, 0, 1]) # x-axis and y-axis ranges
    plt.xlabel(r'$L/E\ \mathrm{[km/MeV]}$')
    plt.show()


##########################################################################     
def nueTwoScales(lowerlim=100, upperlim=9e4):
    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)/10)

    ynorm = vacuumOscProb('e','e',x)    
    normlabel = r'$P_{\nu_e \rightarrow \nu_e}$ (Normal hierarchy)'

    def myVOP(var):
        return np.real(vacuumOscProb('e', 'e', var, True))
    oscmax13 = optimize.fmin_bfgs(myVOP, 1000)
    oscmax12 = optimize.fmin_bfgs(myVOP, 16000)

    fig, ax = plt.subplots()

    ax.plot(x, ynorm, linewidth=2, label=normlabel)    
    ax.set_xscale('log')
    ax.grid()
    plt.axis([lowerlim, upperlim, 0, 1]) # x-axis and y-axis ranges
    ax.tick_params(axis='both', which='major', labelsize=14) # Tick number sizes
    plt.xlabel(r'$L/E\ \mathrm{[m/MeV]}$', fontsize=20)
    plt.ylabel(r'$\overline{\nu}_e$ Survival probability', fontsize=20)

    plt.annotate('', xy=(oscmax13,np.real(vacuumOscProb('e','e',oscmax13)) ), 
                 xycoords='data', xytext=(oscmax13,1.0), textcoords='data', 
                 arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\theta_{13} \sim$', xy=(300,0.95), xycoords='data', 
                 xytext=(0,0), textcoords='offset points', fontsize=16)

    plt.annotate('', xy=(840,0.943), xycoords='data', xytext=(1290,0.943),
                 textcoords='data', arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\sim \Delta m_{31}^2$', xy=(620,0.88), xycoords='data', 
                 xytext=(0,0), textcoords='offset points', fontsize=16)

    plt.annotate('', xy=(oscmax12,np.real(vacuumOscProb('e','e',oscmax12))+0.03),
                 xycoords='data', xytext=(oscmax12,1.0), textcoords='data', 
                 arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\theta_{12} \sim$', xy=(9600,0.65), xycoords='data', 
                 xytext=(0,0), textcoords='offset points', fontsize=16)

    plt.annotate('', xy=(20500,0.22), xycoords='data', xytext=(44000,0.22),
                 textcoords='data', arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\sim \Delta m_{21}^2$', xy=(19000,0.15), xycoords='data', 
                 xytext=(0,0), textcoords='offset points', fontsize=16)

    fig.subplots_adjust(bottom=0.127) # Move up x-axis to fit label
    plt.show()


##########################################################################  
# 'anti' bool is for antineutrinos; 'invhier' bool is for inverted hierarchy
def wikiOscPlots(startflavor, lowerlim=-100, upperlim=3.9e4, anti = False, 
                 invhier = False):
    """
    Plot probabilities for a neutrino starting in specified flavor to 
    oscillate to (or stay in) all three flavors, per the wikipedia plots. 
    If 'anti' is true, antineutrinos are used. If 'invhier' is true, the 
    plots are generated using the inverted mass hierarchy.
    """
    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)/10)
    pmns = PMNSmatrix(wikitheta13, wikitheta12, wikitheta23, deltacp)

    # Make probability plot for electron flavor
    ynue = vacuumOscProb(startflavor, 'e', x, anti, invhier, pmns, 
                         wikidmsq21, wikiposdmsq31)
    nuelabel = r'$\nu_e$'

    # Make probability plot for muon flavor
    ynum = vacuumOscProb(startflavor, 'm', x, anti, invhier, pmns, 
                         wikidmsq21, wikiposdmsq31)
    numlabel = r'$\nu_{\mu}$'

    # Make probability plot for tau flavor
    ynut = vacuumOscProb(startflavor, 't', x, anti, invhier, pmns, 
                         wikidmsq21, wikiposdmsq31)
    nutlabel = r'$\nu_{\tau}$'

    fig, ax = plt.subplots()

    ax.plot(x, ynue, 'k', linewidth=2, label=nuelabel)    
    ax.plot(x, ynum, 'b', linewidth=2, label=numlabel)
    ax.plot(x, ynut, 'r', linewidth=2, label=nutlabel)    
    ax.grid()
    ax.legend(loc=9) # Create legend in upper-center
    plt.axis([lowerlim, upperlim, 0, 1]) # x-axis and y-axis ranges
    plt.xlabel(r'$L/E\ \mathrm{[m/MeV]}$')
    plt.show()


##########################################################################     
def DCdisapp(lowerlim=100, upperlim=9e4):
    """
    Plot the region of L/E for which Double Chooz can see 
    anti-electron neutrino disappearance.
    """
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)

    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)/10)
    y = vacuumOscProb('e', 'e', x, True)

    def myVOP(var):
        return np.real(vacuumOscProb('e', 'e', var, True))

    oscmax = optimize.fmin_bfgs(myVOP, 1000)

    ax.plot(x, y, linewidth=2)    
    ax.set_xscale('log') 
    ax.grid()

#    ax.vlines(1050/1.3, vacuumOscProb('e','e',1050/1.3), 1.0, color='grey')
#    ax.vlines(270, vacuumOscProb('e','e',270), 1.0, color='grey')
    ax.vlines(oscmax, vacuumOscProb('e','e',oscmax), 1.0, color='red', 
              linestyle='--', linewidth=4)
    # Fill between y and 1 over the range specified in the "where" statement.
    ax.fill_between(x,y,1,where=(x < 1050/1.3) & (x > 1050/9), facecolor='grey',
                    alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=18) # Tick number sizes

    plt.axis([lowerlim, upperlim, 0.46, 1.]) # x-axis and y-axis ranges
    plt.xlabel(r'$L/E\ \mathrm{[m/MeV]}$',fontsize=24,)
    plt.ylabel(r'$\overline{\nu}_e$ Survival probability', fontsize=22,
               family="serif") # Change font size and style for y label
    
    fig.subplots_adjust(bottom=0.14) # Move up x-axis to fit label    
    plt.show()

##########################################################################     
def DCdisappFull(lowerlim=100, upperlim=9e4):
    x = np.linspace(lowerlim, upperlim, (upperlim-lowerlim)/10)

    y = vacuumOscProb('e','e',x)    
    normlabel = r'$P_{\nu_e \rightarrow \nu_e}$ (Normal hierarchy)'

    def myVOP(var):
        return np.real(vacuumOscProb('e', 'e', var, True))
    oscmax13 = optimize.fmin_bfgs(myVOP, 1000)
    oscmax12 = optimize.fmin_bfgs(myVOP, 16000)

    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2, label=normlabel)    
    ax.set_xscale('log')
    ax.grid()
    plt.axis([lowerlim, upperlim, 0, 1]) # x-axis and y-axis ranges
    ax.tick_params(axis='both', which='major', labelsize=14) # Tick number sizes
    plt.xlabel(r'$L/E\ \mathrm{[m/MeV]}$', fontsize=20)
    plt.ylabel(r'$\nu_e$ Survival probability', fontsize=20)

    # Fill between y and 1 over the range specified in the "where" statement.
    ax.fill_between(x,y,1,where=(x < 1050/1.3) & (x > 1050/9), facecolor='grey',
                    alpha=0.4)
    ax.vlines(270, vacuumOscProb('e','e',270), 1.0, color='red', linestyle=':',
              linewidth=4)

    plt.annotate('', xy=(oscmax13,np.real(vacuumOscProb('e','e',oscmax13)) ), 
                 xycoords='data', xytext=(oscmax13,1.0), textcoords='data', 
                 arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\theta_{13} \sim$', xy=(350,0.95), xycoords='data', 
                 xytext=(0,0), textcoords='offset points')

    plt.annotate('', xy=(840,0.943), xycoords='data', xytext=(1290,0.943),
                 textcoords='data', arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\sim \Delta m_{31}^2$', xy=(750,0.90), xycoords='data', 
                 xytext=(0,0), textcoords='offset points')

    plt.annotate('', xy=(oscmax12,np.real(vacuumOscProb('e','e',oscmax12))+0.03),
                 xycoords='data', xytext=(oscmax12,1.0), textcoords='data', 
                 arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\theta_{12} \sim$', xy=(10500,0.65), xycoords='data', 
                 xytext=(0,0), textcoords='offset points')

    plt.annotate('', xy=(23600,0.41), xycoords='data', xytext=(42000,0.41),
                 textcoords='data', arrowprops = {'arrowstyle':'<->'})
    plt.annotate(r'$\sim \Delta m_{21}^2$', xy=(22000,0.36), xycoords='data', 
                 xytext=(0,0), textcoords='offset points')

    fig.subplots_adjust(bottom=0.127) # Move up x-axis to fit label
    plt.show()

