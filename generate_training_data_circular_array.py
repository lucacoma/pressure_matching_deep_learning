import numpy as np
import os
import matplotlib.pyplot as plt
import sfs
import scipy
import tqdm
import argparse
from data_lib import params_circular
from data_lib import soundfield_generation as sg

from skimage.metrics import structural_similarity as ssim
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

def plot_soundfield(cmap, P, n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=True, do_norm=True):
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
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for circular array setup')
    parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/soundfield_synthesis/dataset')
    #parser.add_argument('--dataset_name', type=str, help="Base Data Directory", default='data_src_wideband_point_W_23_train')
    #parser.add_argument('--dataset_path', type=str, help='path to dataset', default='/nas/home/lcomanducci/soundfield_synthesis/dataset/data_src_wideband_point_W_23_train.npz' )
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield',
                        default=True)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',
                        default=0)
    args = parser.parse_args()
    eval_points = True
    PLOT_RESULTS = False

    dataset_path = '/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/circular_array'

    # Setup
    # Grid of points where we actually compute the soundfield
    point = params_circular.point

    N_pts = len(point)

    # Secondary Sources Green function
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_circular.N_lspks) + '_r_' + str(params_circular.radius) + '.npy'
    if os.path.exists(os.path.join(dataset_path, green_function_sec_sources_path)):
        G = np.load(os.path.join(dataset_path, green_function_sec_sources_path))
    else:
        G = np.zeros((N_pts, params_circular.N_lspks, params_circular.N_freqs),
                     dtype=complex)
        for n_p in tqdm.tqdm(range(N_pts)):
            hankel_factor_1 = np.tile(params_circular.wc / params_circular.c, (params_circular.N_lspks, 1))
            hankel_factor_2 = np.tile(np.linalg.norm(point[n_p] - params_circular.array_pos, axis=1), reps=(params_circular.N_freqs, 1)).transpose()
            G[n_p, :, :] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
        np.save(os.path.join(dataset_path, green_function_sec_sources_path), G)

    for n_p in range(N_pts):
        if np.sum(np.linalg.norm(point[n_p] - params_circular.array_pos, axis=1) == 0) > 0:
            print(str(n_p))


    if eval_points:
        N_pts = len(params_circular.idx_cp)
        G = G[params_circular.idx_lr[params_circular.idx_cp]]
        point = params_circular.point_cp



    P_gt = np.zeros((len(params_circular.src_pos_train), N_pts, params_circular.N_freqs), dtype=complex)

    for n_s in tqdm.tqdm(range(len(params_circular.src_pos_train))):
        xs = params_circular.src_pos_train[n_s]

        # np.tile(np.expand_dims(Phi,axis=0),reps=(params_circular.N_lspks,1,1))
        # Ground truth source
        if args.gt_soundfield:
            for n_f in range(params_circular.N_freqs):
                P_gt[n_s, :, n_f] = (1j / 4) * \
                               scipy.special.hankel2(0,
                                                     (params_circular.wc[n_f] / params_circular.c) *
                                                     np.linalg.norm(point[:, :2] - xs, axis=1))

        if PLOT_RESULTS:
            # Plot params
            selection = np.ones_like(params_circular.array_pos[:, 0])
            selection = selection == 1
            #n_s = 32
            n_f = 41
            print(str(params_circular.f_axis[n_f]))
            cmap = 'RdBu_r'
            tick_font_size = 70
            axis_label_size = 90
            plot_soundfield(cmap, P_gt[n_s], n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=True,
                            do_norm=True)
            print('bella')

    if args.gt_soundfield:
        np.save(os.path.join(dataset_path, 'gt_soundfield_train.npy'), P_gt)


if __name__ == '__main__':
    main()
