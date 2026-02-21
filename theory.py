import numpy as np
import matplotlib.pyplot as plt

def correlation(a, b, theta, phi):
    
    '''
    Docstring for correlation
    Correlation function E(a, b) for a quantum state cos(theta)|HH> + exp(i phi) sin(theta)|VV>
    
    :param a: transmission angle of polarizer A (rad)
    :param b: transmission angle of polarizer B (rad)
    :param theta: Half wave plate angle (see equation above) (rad)
    :param phi: Quartz plate (+ BBO) phase shift (see above) (rad)
    '''

    return (np.cos(2*a)*np.cos(2*b)
            + np.sin(2*theta)*np.sin(2*a)*np.sin(2*b)*np.cos(phi))

def compute_S(theta, phi, a = np.deg2rad(-45.0), a_p = np.deg2rad(0.0), b = np.deg2rad(-22.5), b_p=np.deg2rad(22.5)):
    '''
    Docstring for compute_S
    Return the S parameter computed for Bell's inequality 
    
    :param theta: Half wave plate angle (rad)
    :param phi: Quartz Plate (+BBO) phase shift (rad)
    :param a: polarizer A configuration 1 transmission angle (rad)
    :param a_p: polarizer A configuration 2 transmission angle (rad)
    :param b: polarizer B configuration 1 transmission angle (rad)
    :param b_p: polarizer B configuration 2 transmission angle (rad)
    '''

    return correlation(a, b, theta, phi) - correlation(a, b_p, theta, phi) + correlation(a_p, b, theta, phi) + correlation(a_p, b_p, theta, phi)


if __name__ == "__main__":
    print("S(45 deg, 0) =", compute_S(np.deg2rad(45.0), np.deg2rad(0.0)), " expected ~", 2*np.sqrt(2))

    theta_grid = np.linspace(-np.pi, np.pi, 361)         
    phi_grid   = np.linspace(-np.pi, np.pi, 361)     

    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    S_grid = compute_S(TH, PH)  
    absS = np.abs(S_grid)

    plt.figure()
    im = plt.imshow(
        absS,
        origin="lower",
        aspect="auto",
        extent=[phi_grid.min(), phi_grid.max(), theta_grid.min(), theta_grid.max()],
    )

    plt.xlabel(r"$\phi$ (rad)")
    plt.ylabel(r"$\theta$ (deg)")
    plt.title(r"$|S(\theta,\phi)|$")

    plt.colorbar(im, label=r"$|S|$")

    plt.contour(
        PH, TH, absS,
        levels=[2.0],
    )

    plt.show()