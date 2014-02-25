import numpy as np
import matplotlib.pyplot as plt

def cubic_int(x, pmin1, p0, p1, p2):
    xsq, xcb = x**2, x**3   
    #    print 'xsq: %s   xcb: %s' % (xsq, xcb)

    xvect = np.array([(-xcb + 2*xsq - x), (3.*xcb - 5.*xsq + 2), (-3.*xcb + 4.*xsq + x), (xcb - xsq)])
    #    print 'xvect: %s' % xvect
    pvect = np.array([pmin1, p0, p1, p2])
    #    print 'pvect: %s' % pvect

    return 0.5*np.dot(np.transpose(xvect), pvect)
    #    return 0.5*np.dot(xvect, pvect)

def cubic_int2(x, pmin1, p0, p1, p2):
    xsq, xcb = x**2, x**3    
    #    print 'xsq: %s   xcb: %s' % (xsq, xcb)
    
    return 0.5*(2*p0 + (-pmin1+p1)*x + (2*pmin1 - 5*p0 + 4*p1 - p2)*xsq\
                    + (-pmin1 + 3*p0 - 3*p1 + p2)*xcb)

# Assumes we know (x = -1, y = pmin1), (0, p0), (1, p1), and (2, p2)
def testplot(pmin1, p0, p1, p2, xmin = -2, xmax = 3, npts = 100):
    x = np.linspace(xmin, xmax, npts)
    y = cubic_int2(x, pmin1, p0, p1, p2)
    y2 = cubic_int(x, pmin1, p0, p1, p2)
    pts = [[-1,pmin1], [0,p0], [1,p1], [2,p2]]

    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2, color='b', ls='--')
    ax.plot(x, y2, linewidth=3, color='y', linestyle='-.')
    # This trick is known as 'unpacking argument lists.'
    ax.plot(*zip(*pts), marker='o', color='r')
    plt.axis([xmin, xmax,-2,10])
    plt.show()

if __name__ == "__main__":
    #    print cubic_int2(0.5, 1, -2, 0, -2)

    testplot(1,4,2,3)
