import numpy as np
import os
import matplotlib.pyplot as plt
import sfs
import scipy
import tqdm
import argparse
import tensorflow as tf
from data_lib import params_circular
import jax.numpy as jnp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# /nas/home/lcomanducci/soundfield_synthesis_filterbank/MeshRIR/S32-M441_npy
# /nas/home/lcomanducci/soundfield_synthesis_filterbank/MeshRIR/irutilities.py
from skimage.metrics import structural_similarity as ssim
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

def plot_soundfield(cmap, P, n_f, selection, axis_label_size, tick_font_size, save_path, plot_ldspks=True, do_norm=True):
    figure = plt.figure(figsize=(20, 20))
    if do_norm:
        im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params_circular.N_sample, params_circular.N_sample)),
                                  params_circular.grid, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    else:
        im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params_circular.N_sample, params_circular.N_sample)),
                                  params_circular.grid,  cmap=cmap, colorbar=False, vmin=P[:, n_f].min(), vmax=P[:, n_f].max(), xnorm=None)
    if plot_ldspks:
        sfs.plot2d.loudspeakers(params_circular.array.x[selection], params_circular.array.n[selection], a0=1, size=0.18)
    plt.xlabel('$x [m]$', fontsize=axis_label_size), plt.ylabel('$y [m]$', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    cbar = plt.colorbar(im, fraction=0.046)
    cbar.ax.tick_params(labelsize=tick_font_size)
    # cbar.set_label('$NRE~[\mathrm{dB}]$',fontsize=tick_font_size))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def sound_field(d, selection, secondary_source, array, grid, tapering=True):
    if tapering:
        tapering_window = sfs.tapering.tukey(selection, alpha=0.3)
    else:
        tapering_window = sfs.tapering.none(selection)
    p = sfs.fd.synthesize(d, tapering_window, array, secondary_source, grid=grid)
    return p, tapering_window
""""""
def normalize(x):
    min_x = x.min()
    max_x = x.max()
    x_norm = (x - min_x)/(max_x-min_x)
    return x_norm

def nmse(P_hat, P_gt,type='freq'):
    if type=='freq':
        return np.mean((np.power(np.abs(P_hat[ :, :] - P_gt[ :, :]), 2) / np.power(np.abs(P_gt[ :, :]), 2)), axis=0)
    else:
        return np.power(np.abs(P_hat[ :, :] - P_gt[ :, :]), 2) / np.power(np.abs(P_gt[ :, :]), 2)


def ssim_abs(P_hat, P_gt):
    P_hat = normalize(np.abs(P_hat))
    P_gt = normalize(np.abs(P_gt))
    return ssim(P_gt, P_hat, data_range=1)

def ssim_freq(P_hat, P_gt):
    ssim_freq_array = np.zeros(params_circular.N_freqs)
    for n_f in range(params_circular.N_freqs):
        ssim_freq_array[n_f] = ssim_abs(P_hat[:, n_f], P_gt[:, n_f])
    return ssim_freq_array


def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for circular array setup')
    parser.add_argument('--dataset_path', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/test/circular_array')
    parser.add_argument('--model_path', type=str, help='Deep learning models folder', default='/nas/home/lcomanducci/pressure_matching_deep_learning/models/circular_array')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    #parser.add_argument('--wfs', type=bool, help='compute soundfield synthesis', default=True)
    #parser.add_argument('--hoa', type=bool, help='compute higher order ambisonics', default=True)
    parser.add_argument('--pm', type=bool, help='compute pressure matching', default=True)
    parser.add_argument('--pwd_cnn', type=bool, help='compute model-based acoustic rendering + CNN', default=True)
    parser.add_argument('--n_loudspeakers', type=int, help='Numbr of loudspeakers in array', default=64)
    eval_points = False
    PLOT = True
    args = parser.parse_args()
    dataset_path =args.dataset_path

    # Grid of points where we actually compute the soundfield
    point = params_circular.point
    N_pts = len(point)     #params_circular.radius_sources_test

    # Load green function secondary sources --> eval points (it is in train directory since it is the same)
    dataset_path_train = '/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/circular_array'
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(args.n_loudspeakers) + '_r_' + str(params_circular.radius) + '.npy'
    G = np.load(os.path.join(dataset_path_train, green_function_sec_sources_path))

    # Let's precompute what we need in order to apply the selected models
    # Load pwd_cnn deep learning model
    if args.pwd_cnn:
        model_name = 'model_circular_config_nl_'+str(args.n_loudspeakers)
        network_model = tf.keras.models.load_model(os.path.join(args.model_path, model_name))

    if args.pm:
        lambda_ = 1e-2
        G_cp = G[params_circular.idx_lr[params_circular.idx_cp]]  # Green function at control points
        # points_cp = params_circular.point[params_circular.idx_lr[params_circular.idx_cp]]
        C_pm = np.zeros((args.n_loudspeakers, len(params_circular.idx_cp), params_circular.N_freqs ),dtype=complex)
        for n_f in tqdm.tqdm(range(len(params_circular.wc))):
            C_pm[:, :, n_f] = np.matmul(jnp.linalg.pinv(np.matmul(G_cp[:, :, n_f].transpose(), G_cp[:, :, n_f]) + lambda_ * np.eye(args.n_loudspeakers)),G_cp[:, :, n_f].transpose())

    if eval_points:
        N_pts = len(params_circular.idx_lr)
        G = G[params_circular.idx_lr]
        point = params_circular.point_lr

    nmse_pwd_cnn = np.zeros((len(params_circular.radius_sources_train), params_circular.n_sources_radius, params_circular.N_freqs))
    nmse_pwd_pm = np.zeros_like(nmse_pwd_cnn)

    ssim_pwd_cnn = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_pm = np.zeros_like(nmse_pwd_cnn)

    for n_r in tqdm.tqdm(range(len(params_circular.radius_sources_train))):
        P_gt = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd_cnn = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd_pm = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)

        for n_s in range(params_circular.n_sources_radius):
            if PLOT:
                n_s = 41
            xs = params_circular.src_pos_test[n_r, n_s]

            if args.gt_soundfield:
                for n_f in range(params_circular.N_freqs):
                    hankel_arg = (params_circular.wc[n_f] / params_circular.c) * \
                                 np.linalg.norm(point[:, :2] - xs, axis=1)
                    P_gt[ :, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_arg)

            if args.pwd_cnn:
                if eval_points:
                    P_input = P_gt[params_circular.idx_cp]
                else:
                    P_input = P_gt[params_circular.idx_lr[params_circular.idx_cp]]

                d_array_cnn = network_model.predict(
                    np.expand_dims(np.concatenate([np.real(P_input), np.imag(P_input)], axis=0), axis=[0, -1]).astype('float32'))[0, :, :, 0].astype('float64')

                d_array_cnn_complex = d_array_cnn[:int(d_array_cnn.shape[0] / 2)] + (1j * d_array_cnn[int(d_array_cnn.shape[0] / 2):])

                plt.figure(figsize=(10,10))
                plt.imshow(d_array_cnn.T, aspect='auto',cmap='RdBu')
                plt.xlabel('$l$',fontsize=120),plt.ylabel('$k$',fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.show()

                for n_p in range(N_pts):
                    P_pwd_cnn[ n_p, :] = np.sum(G[n_p] * d_array_cnn_complex, axis=0)

            if args.pm:
                d_pm = np.zeros((args.n_loudspeakers, params_circular.N_freqs),dtype=complex)
                for n_f in range(params_circular.N_freqs):
                    if eval_points:
                        d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[params_circular.idx_cp, n_f]) # CHECK idx_cp is USING EVAL POINTSSSSS
                    else:
                        d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[params_circular.idx_lr[params_circular.idx_cp], n_f]) # CHECK idx_cp is USING EVAL POINTSSSSS

                for n_p in range(N_pts):
                    P_pwd_pm[n_p, :] = np.sum(G[n_p] * d_pm, axis=0)

            nmse_pwd_cnn[n_r, n_s], nmse_pwd_pm[n_r, n_s] = nmse(P_pwd_cnn, P_gt), nmse(P_pwd_pm, P_gt)
            ssim_pwd_cnn[n_r, n_s], ssim_pwd_pm[n_r, n_s] =  ssim_freq(P_pwd_cnn, P_gt), ssim_freq(P_pwd_pm, P_gt)
            if PLOT:
                plt.figure(), plt.plot(params_circular.array_pos[:4, 0], params_circular.array_pos[:4, 1],
                                       'r*'), plt.show()
                d = np.linalg.norm(np.array([params_circular.array_pos[1, 0], params_circular.array_pos[1, 1]])-np.array([params_circular.array_pos[2, 0], params_circular.array_pos[2, 1]]))
                aliasing_freq = params_circular.c/(2*d)
                # Plot params
                selection = np.ones_like(params_circular.array_pos[:, 0])
                selection = selection == 1
                n_f = 63   # 63
                print(str(params_circular.f_axis[n_f]))
                cmap = 'RdBu_r'
                tick_font_size = 70
                axis_label_size = 90

                # Ground truth
                plot_paths = os.path.join('plots', 'circular')
                save_path = os.path.join(plot_paths, 'sf_real_source_'
                                         + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '_nl'+str(args.n_loudspeakers)+'.pdf')
                plot_soundfield(cmap, P_gt, n_f, selection, axis_label_size, tick_font_size, save_path, plot_ldspks=False)

                # PWD-CNN
                save_path = os.path.join(plot_paths, 'sf_pwd_cnn_'
                                         + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '_nl'+str(args.n_loudspeakers)+'.pdf')
                plot_soundfield(cmap, P_pwd_cnn, n_f, selection, axis_label_size, tick_font_size, save_path)

                # Error
                nmse_pwd_cnn = 10*np.log10(nmse(P_pwd_cnn, P_gt, type='full'))
                save_path = os.path.join(plot_paths, 'nmse_pwd_cnn_'
                                         + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '_nl'+str(args.n_loudspeakers)+'.pdf')
                plot_soundfield(cmap, nmse_pwd_cnn, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False)

                # PM
                save_path = os.path.join(plot_paths, 'sf_pm_'
                                         + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '_nl'+str(args.n_loudspeakers)+'.pdf')
                plot_soundfield(cmap, P_pwd_pm, n_f, selection, axis_label_size, tick_font_size, save_path)

                # Error
                nmse_pm = 10*np.log10(nmse(P_pwd_pm, P_gt, type='full'))
                save_path = os.path.join(plot_paths, 'nmse_pm_'
                                         + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) +'_nl'+str(args.n_loudspeakers)+ '.pdf')
                plot_soundfield(cmap, nmse_pm, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False)

                plt.figure()
                plt.plot(params_circular.f_axis, 10*np.log10(np.mean(nmse(P_pwd_cnn, P_gt, type='full'), axis=0)),'k-*')
                plt.plot(params_circular.f_axis, 10*np.log10(np.mean(nmse(P_pwd_pm, P_gt, type='full'), axis=0)),'r-*')
                plt.tick_params(axis='both', which='major', labelsize=10)

                plt.legend(['CNN', 'PM'])
                plt.show()

                print('pause')

    # If we are plotting it means we are just computing data for the paper --> no need to save anything
    if not PLOT:
        # Save arrays
        np.savez(os.path.join(dataset_path, 'nmse_nl_' + str(args.n_loudspeakers)+'.npz'),
  nmse_pwd_cnn=nmse_pwd_cnn, nmse_pwd_pm=nmse_pwd_pm)
        np.savez(os.path.join(dataset_path, 'ssim_nl_' + str(args.n_loudspeakers)+'.npz'),
 ssim_pwd_cnn=ssim_pwd_cnn, ssim_pwd_pm=ssim_pwd_pm)


if __name__ == '__main__':
    main()
