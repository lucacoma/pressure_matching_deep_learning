import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

from data_lib import params_circular #, params_linear

def plot2pgf(temp, filename, folder='./'):
    """
    :param temp: list of equally-long data
    :param filename: filename without extension nor path
    :param folder: folder where to save
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, filename + '.txt'), np.asarray(temp).T, fmt="%f", encoding='ascii')

def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    parser.add_argument('--dataset_path', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/test/')
    parser.add_argument('--array_type', type=str, help="Array Configuration", default='circular')
    parser.add_argument('--n_loudspeakers', type=int, help='number missing loudspeakers',
                        default=64)
    args = parser.parse_args()
    array_type = args.array_type

    nmse = np.load(os.path.join(args.dataset_path, array_type+'_array', 'nmse_nl_' + str(args.n_loudspeakers)+'.npz'))
    ssim = np.load(os.path.join(args.dataset_path, array_type+'_array', 'ssim_nl_' + str(args.n_loudspeakers)+'.npz'))

    nmse_pwd_cnn, nmse_pm =  np.real(nmse['nmse_pwd_cnn']), np.real(nmse['nmse_pwd_pm'])
    ssim_pwd_cnn, ssim_pm = np.real(ssim['ssim_pwd_cnn']), np.real(ssim['ssim_pwd_pm'])

    os.path.join(args.dataset_path, array_type, 'nmse_'+str(args.n_loudspeakers)+'.npz')

    pgf_dataset_path = os.path.join(args.dataset_path, array_type + '_array', 'pgfplot')

    if args.array_type == 'circular':
        # Let's store the data in arrays more suitable for saving them (and then re-using them in pgfplot)
        f_axis_plot = np.round(params_circular.f_axis, 1)
        nmse_pwd_cnn_freq_db = 10*np.log10(np.mean(np.reshape(nmse_pwd_cnn, (-1, params_circular.N_freqs)), axis=0))
        nmse_pm_freq_db = 10*np.log10(np.mean(np.reshape(nmse_pm, (-1,params_circular.N_freqs)), axis=0))
        ssim_pwd_cnn_freq = np.mean(np.reshape(ssim_pwd_cnn, (-1, params_circular.N_freqs)), axis=0)
        ssim_pm_freq =  np.mean(np.reshape(ssim_pm, (-1, params_circular.N_freqs)), axis=0)

        plot2pgf([f_axis_plot, nmse_pwd_cnn_freq_db], 'nmse_pwd_cnn_freq_db_'+str(args.n_loudspeakers), folder=pgf_dataset_path)
        plot2pgf([f_axis_plot, nmse_pm_freq_db], 'nmse_pm_freq_db_'+str(args.n_loudspeakers), folder=pgf_dataset_path)
        plot2pgf([f_axis_plot, ssim_pwd_cnn_freq], 'ssim_pwd_cnn_freq_'+str(args.n_loudspeakers), folder=pgf_dataset_path)
        plot2pgf([f_axis_plot, ssim_pm_freq], 'ssim_pm_freq_'+str(args.n_loudspeakers), folder=pgf_dataset_path)


        plt.figure(figsize=(20, 9))
        plt.plot(f_axis_plot, nmse_pwd_cnn_freq_db, 'rd-')
        plt.plot(f_axis_plot, nmse_pm_freq_db, 'k*-')
        plt.xlabel('$f [Hz]$', fontsize=35), plt.ylabel('$NRE [dB]$', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.legend([ '$\mathrm{PWD_{CNN}}$', '$\mathrm{PM}$'], prop={"size": 30})
        plt.show()

        plt.figure(figsize=(20, 9))
        plt.plot(f_axis_plot, ssim_pwd_cnn_freq, 'rd-')
        plt.plot(f_axis_plot, ssim_pm_freq, 'k*-')
        plt.xlabel('$f [Hz]$', fontsize=35), plt.ylabel('$SSIM$', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.legend([ '$\mathrm{PWD_{CNN}}$', '$\mathrm{PM}$'], prop={"size": 30})
        plt.show()

        # Plot w.r.t radius distance
        n_f = 41
        r_axis_plot = np.round(params_circular.radius_sources_test[:-1], 1)

        f_axis_plot = np.round(params_circular.f_axis, 1)
        nmse_pwd_cnn_radius_db = 10*np.log10(np.mean(nmse_pwd_cnn[:, :, n_f], axis=1))
        nmse_pm_radius_db = 10*np.log10(np.mean(nmse_pm[:, :, n_f], axis=1))
        ssim_pwd_cnn_radius = np.mean(ssim_pwd_cnn[:, :, n_f], axis=1)
        ssim_pm_radius = np.mean(ssim_pm[:, :, n_f], axis=1)

        plot2pgf([r_axis_plot, nmse_pwd_cnn_radius_db], 'nmse_pwd_cnn_radius_db_'+str(args.n_loudspeakers), folder=pgf_dataset_path)
        plot2pgf([r_axis_plot, nmse_pm_radius_db], 'nmse_pm_radius_db_'+str(args.n_loudspeakers), folder=pgf_dataset_path)
        plot2pgf([r_axis_plot, ssim_pwd_cnn_radius], 'ssim_pwd_cnn_radius_'+str(args.n_loudspeakers), folder=pgf_dataset_path)
        plot2pgf([r_axis_plot, ssim_pm_radius], 'ssim_pm_radius_'+str(args.n_loudspeakers), folder=pgf_dataset_path)

        plt.figure(figsize=(20, 9))
        plt.plot(r_axis_plot, nmse_pwd_cnn_radius_db, 'rd-')
        plt.plot(r_axis_plot, nmse_pm_radius_db, 'k*-')
        plt.xlabel('$r [m]$', fontsize=35), plt.ylabel('$NRE [dB]$', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.legend([ '$\mathrm{PWD_{CNN}}$', '$\mathrm{PM}$'], prop={"size": 30})
        plt.show()

        plt.figure(figsize=(20, 9))
        plt.plot(r_axis_plot, ssim_pwd_cnn_radius, 'rd-')
        plt.plot(r_axis_plot, ssim_pm_radius, 'k*-')
        plt.xlabel('$r [m]$', fontsize=35), plt.ylabel('$SSIM$', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.legend(['$\mathrm{PWD_{CNN}}$', '$\mathrm{PM}$'], prop={"size": 30})
        plt.show()

        print('done')

if __name__ == '__main__':
    main()