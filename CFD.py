import numpy as np
import matplotlib.pyplot as plt

# INPUT SECTION
r = 6  # radius in (m)
dens = 1025  # density of the seawater
H = 20  # height of the chanel in (m)
L = 50  # longitude of the chanel in (m)
tol = 0.4  # tolerance of the mesh
v_in = 2  # inlet velocity (m/s) in x-axis
max_iter = 2000  # maximum number of iterations in the gauss-seidel
max_difFer = 1e-6
time = 10  # sec
delta_t = 100  # number of divisions
multi_cyl = False
time_step = np.linspace(0, time, delta_t)
red_factor = 0.9


# MESH DEFINITION
dx = tol  # differential x
dy = tol  # differential y
M = int(round(H / dy))  # number of volumes along X axis
N = int(round(L / dx))  # number of volumes along Y axis
X = np.linspace(0, L, N + 2)
Y = np.linspace(H, 0, M + 2)[:, np.newaxis]
x, y = np.meshgrid(X, Y)
shape_x = np.linspace(0, ((N+1)), N+2)
shape_y = np.linspace((M+1), 0, M + 2)[:, np.newaxis]
shape_grid_x, shape_grid_y = np.meshgrid(shape_x, shape_y)

M_fluid = np.ones_like(x)
M_dens = np.ones_like(M_fluid) * dens
dPE = dx
dPW = dx
dPS = dy
dPN = dy
dPe = dx / 2
dPw = dx / 2
dPs = dy / 2
dPn = dy / 2
dEe = dx / 2
dWw = dx / 2
dSs = dy / 2
dNn = dy / 2

# NUMBER OF POINTS IN BOTH AXIS
py = int(round((M + 2) / 2))
px = int(round((N + 2) / 2))

# initializing bP coefficient
bP = np.zeros_like(M_fluid)

if multi_cyl:
    Cx = L / 2
    Cy = H / 3

    for i in range(M + 2):
        for j in range(N + 2):
            R = np.sqrt((x[i, j] - Cx) ** 2 + (y[i, j] - Cy) ** 2)
            if R < r:
                M_fluid[i, j] = 0
                bP[i, j] = v_in * H / 2

    Cx = (L / 3) * 2
    Cy = (H / 3) * 2

    for i in range(M + 2):
        for j in range(N + 2):
            R = np.sqrt((x[i, j] - Cx) ** 2 + (y[i, j] - Cy) ** 2)
            if R < r:
                M_fluid[i, j] = 0
                bP[i, j] = v_in * H / 2

    for i in range(int((M + 2)/6), int((M + 2)/6)*6):
        for j in range(int((N + 2)/6), int((N + 2)/6)*2):
            M_fluid[i, j] = 0
            bP[i, j] = v_in * H / 2


else:
    Cx = L / 2
    Cy = H / 2

    for i in range(M + 2):
        for j in range(N + 2):
            R = np.sqrt((x[i, j] - Cx) ** 2 + (y[i, j] - Cy) ** 2)
            if R < r:
                M_fluid[i, j] = 0
                bP[i, j] = v_in * H / 2

# INITIALIZING COEFFICIENTS
ae = np.ones_like(M_fluid)
aw = np.ones_like(M_fluid)
as_ = np.ones_like(M_fluid)
an = np.ones_like(M_fluid)
ap = np.ones_like(M_fluid)

# BOUNDARY CONDITIONS for the inlet and outlet
bP[:, 0] = v_in * Y[:, 0]
an[:, 0] = 0
as_[:, 0] = 0
ae[:, 0] = 0
aw[:, 0] = 0
an[:, -1] = 0
as_[:, -1] = 0
ae[:, -1] = 0
bP[0, :] = H * v_in

# Coefficients for the internal solid cylinder with 'r' radius
for j in range(px - int(r / dx), px + int(r / dx)):
    for i in range(py - int(r / dy), py + int(r / dy)):
        if M_fluid[i, j] == 1:
            if np.sqrt((x[i, j] - x[py, px]) ** 2 + (y[i, j] - y[py, px]) ** 2) < r:
                ap[i, j] = 1
                ae[i, j] = 0
                aw[i, j] = 0
                as_[i, j] = 0
                an[i, j] = 0

max_velocity=[]
for t in range(len(time_step)):
    # INITIALIZING VARIABLES

    v_in = v_in * red_factor
    PSI = (y * v_in) * M_fluid
    M_dens = M_dens * M_fluid
    differ = 1
    Iter = 0
    PSI_old = PSI.copy()
    iter_list = []
    dif_list = []

    # COEFFICIENT COMPUTATION
    while differ > max_difFer and Iter < max_iter:
        PSI_old = PSI.copy()
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if M_fluid[i, j] == 1:
                    # using the harmonic mean, calculate the ae coefficient (density dependent but for now dens is
                    # constant)
                    ae[i, j] = ((dPE / ((dPe / (dens / M_dens[i, j])) + (dEe / (dens / M_dens[i, j + 1])))) * (
                                dy / dPE))
                    ind = np.isinf(ae)
                    ae[ind] = 1
                    ind = np.isnan(ae)
                    ae[ind] = 1

                    # using the harmonic mean, calculate the aw coefficient (density dependent but for now dens is
                    # constant)
                    aw[i, j] = (dPW / ((dPw / (dens / M_dens[i, j])) + (dWw / (dens / M_dens[i, j - 1])))) * (dy / dPW)
                    ind = np.isinf(aw)
                    aw[ind] = 1
                    ind = np.isnan(aw)
                    aw[ind] = 1

                    # using the harmonic mean, calculate the as coefficient (density dependent but for now dens is
                    # constant)
                    as_[i, j] = (dPS / ((dPs / (dens / M_dens[i, j])) + (dSs / (dens / M_dens[i + 1, j])))) * (dx / dPS)
                    ind = np.isinf(as_)
                    as_[ind] = 1
                    ind = np.isnan(as_)
                    as_[ind] = 1

                    # using the harmonic mean, calculate the an coefficient (density dependent but for now dens is
                    # constant)
                    an[i, j] = (dPN / ((dPn / (dens / M_dens[i, j])) + (dNn / (dens / M_dens[i - 1, j])))) * (dx / dPN)
                    ind = np.isinf(an)
                    an[ind] = 1
                    ind = np.isnan(an)
                    an[ind] = 1

                    # Falta ver como se calcula bP en los nodos internos y como se cambia la rotación
                    # bP[i, j] = 0.8
                    ap[i, j] = ae[i, j] + aw[i, j] + as_[i, j] + an[i, j]
                    ind = np.isinf(ap)
                    ap[ind] = 1
                    ind = np.isnan(ap)
                    ap[ind] = 1
                else:
                    ap[i, j] = 1
                    ae[i, j] = 0
                    aw[i, j] = 0
                    as_[i, j] = 0
                    an[i, j] = 0
                    bP[i, j] = v_in * H / 2
                PSI[i, j] = (ae[i, j] * PSI[i, j + 1] + aw[i, j] * PSI[i, j - 1] +
                             an[i, j] * PSI[i - 1, j] + as_[i, j] * PSI[i + 1, j] + bP[i, j]) / ap[i, j]
        Iter += 1
        iter_list.append(Iter)
        differ = np.max(np.max(np.abs(PSI_old - PSI)))
        dif_list.append(differ)
        print("Time-step: ", t, " Iterations: ", Iter, " Error: ", "{:.2f}".format(differ))

    PSI[:, -1] = PSI[:, -2]

    # INITIALIZING VELOCITY AND PRESSURE
    vP = v_in * M_fluid  # initializing the scalar velocity field
    pP = v_in * M_fluid  # initializing the pressure field
    # for velocity
    vxP = np.zeros(M_fluid.shape)  # x component velocity matrix
    vyP = np.zeros(M_fluid.shape)  # y component velocity matrix
    vxn = np.ones(M_fluid.shape)
    vxs = np.ones(M_fluid.shape)
    vye = np.ones(M_fluid.shape)
    vyw = np.ones(M_fluid.shape)

    # for pressure
    pxn = np.ones(M_fluid.shape)
    pxs = np.ones(M_fluid.shape)
    pye = np.ones(M_fluid.shape)
    pyw = np.ones(M_fluid.shape)

    # VELOCITY COMPUTATION
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if M_fluid[i, j] == 1:
                vxn = an[i, j] * ((PSI[i, j] - PSI[i - 1, j]) / dPN)
                vxs = as_[i, j] * ((PSI[i + 1, j] - PSI[i, j]) / dPS)
                vye = ae[i, j] * ((PSI[i, j + 1] - PSI[i, j]) / dPE)
                vyw = aw[i, j] * ((PSI[i, j] - PSI[i, j - 1]) / dPW)
                vxP[i, j] = -(vxn + vxs) / 2
                vyP[i, j] = -(vye + vyw) / 2
                vP[i, j] = np.sqrt(vxP[i, j] ** 2 + vyP[i, j] ** 2)

    PSI = PSI * M_fluid  # seeting the inside values of the stream function to zero

    v_max = np.max(np.max(vP))  # maximum velocity of the current time-step
    max_velocity.append(v_max)  # maximum velocities for each time-step
    max_max_v = np.max(max_velocity)  # maximum velocity of all the time


    # ==========================================================================
    # PLOT SECTION

    # STREAMLINE PLOT
    mask_x = np.flipud(vxP != 0)
    mask_y = np.flipud(vyP != 0)
    plt.style.use("dark_background")

    # VELOCITY FIELD =========================================================================
    # STREAMPLOT
    img0 = plt.streamplot(np.flipud(shape_grid_x),  np.flipud(shape_grid_y), np.flipud(vxP), np.flipud(vyP), density=4,
                          linewidth=0.18, arrowsize=0.2, color=np.flipud(vxP), cmap="inferno")  # color=vxP
    plt.colorbar(label="Velocity (m/s)", shrink=0.6)
    plt.xlabel('N control volumes')
    plt.ylabel('M control volumes, Iter: ' + str(Iter))
    plt.title("VELOCITY FIELD  Vmax=" + str("{:.2f}".format(v_max)) + "m/s " + "  Time-Step: " + str(t))
    plt.autoscale()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('0_' + str(t) + '.png', dpi=600)
    plt.clf()

    # QUIVER 1
    img1 = plt.imshow(vP, aspect='auto', interpolation='gaussian', cmap='inferno', alpha=1, vmax=max_max_v)
    plt.axis(img1.get_extent())
    plt.quiver(shape_grid_x[mask_x], shape_grid_y[mask_y], np.flipud(vxP)[mask_x], np.flipud(vyP)[mask_y], color='w',
               alpha=0.6, scale=200, width=0.0005)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.autoscale()
    plt.colorbar(img1, label="Velocity (m/s)", shrink=0.6)
    plt.xlabel('N control volumes')
    plt.ylabel('M control volumes, Iter: ' + str(Iter))
    plt.title("VELOCITY FIELD  Vmax=" + str("{:.2f}".format(v_max)) + "m/s " + "  Time-Step: " + str(t))
    plt.savefig('1_' + str(t) + '.png', dpi=600)
    plt.clf()

    # QUIVER 2
    img2 = plt.imshow(vP, aspect='auto', interpolation='gaussian', cmap='jet', alpha=1, vmax=max_max_v)
    plt.axis(img1.get_extent())
    plt.quiver(shape_grid_x[mask_x], shape_grid_y[mask_y], np.flipud(vxP)[mask_x], np.flipud(vyP)[mask_y], color='k',
               alpha=0.65, scale=200, width=0.0008)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.autoscale()
    plt.colorbar(img2, label="Velocity (m/s)", shrink=0.6)
    plt.xlabel('N control volumes')
    plt.ylabel('M control volumes, Iter: ' + str(Iter))
    plt.title("VELOCITY FIELD  Vmax=" + str("{:.2f}".format(v_max)) + "m/s " + "  Time-Step: " + str(t))
    plt.savefig('2_' + str(t) + '.png', dpi=600)
    plt.clf()

    # PLOT OF CONVERGENCE
    img3 = plt.stackplot(iter_list, dif_list, colors="blue")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.savefig('3_' + str(t) + '.png', dpi=600)
    plt.clf()

# STREAM FUNCTION PSI
plt.imshow(PSI, aspect='auto', interpolation='gaussian', cmap='inferno')
plt.gca().set_aspect('equal', adjustable='box')
plt.autoscale()
plt.colorbar()
plt.title("STREAM FUNCTION (PSI)" + "  Time-Step: " + str(t))
plt.xlabel('Control Volumes')
plt.show()

# FLUIDITY MATRIX
plt.imshow(M_fluid, aspect='auto', interpolation='none', cmap='autumn')
plt.gca().set_aspect('equal', adjustable='box')
plt.autoscale()
plt.colorbar()
plt.xlabel('Control Volumes')
plt.show()
