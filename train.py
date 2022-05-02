import numpy as np
import os
import argparse
import datetime
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_ALLOW_GROWTH']='True'
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import sfs

import random
from train_lib import network_utils
from train_lib import train_utils
from data_lib import params_circular
from sklearn.model_selection import train_test_split
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE

def main():

    parser = argparse.ArgumentParser(
        description='Sounfield reconstruction')

    parser.add_argument('--epochs', type=int, help='Number of epochs', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--log_dir', type=str, help='Tensorboard log directory', default='/nas/home/lcomanducci/pressure_matching_deep_learning/logs/scalars')
    parser.add_argument('--number_missing_ldspk', type=int, help='number of missing loudspeakers',default=32)
    parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset', default='/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/circular_array/gt_soundfield_train.npy' )
    parser.add_argument('--learning_rate', type=float, help='LEarning rate', default=0.001)
    parser.add_argument('--n_loudspeakers', type=float, help='LEarning rate', default=32)
    parser.add_argument('--green_function', type=str, help='LEarning rate', default='/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/circular_array/green_function_sec_sources_nl_64_r_1.5.npy')
    parser.add_argument('--gpu', type=str, help='gpu number', default='1')

    args = parser.parse_args()
    number_missing_loudspeakers = args.number_missing_ldspk
    epochs = args.epochs
    batch_size = args.batch_size
    log_dir = args.log_dir

    gt_soundfield_dataset_path = args.gt_soundfield_dataset_path

    lr = args.learning_rate
    green_function_path = '/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/circular_array/green_function_sec_sources_nl_'+str(args.n_loudspeakers)+'_r_1.5.npy'

    saved_model_path = '/nas/home/lcomanducci/pressure_matching_deep_learning/models/circular_array/model_circular_config_nl_'+str(args.n_loudspeakers)

    # Select gpu
    lambda_abs = 25  # we weight the absolute value loss since it is in a different range w.r.t. phase

    log_name = 'prova__lr_'+str(lr)+'lamba_abs_'+str(lambda_abs)

    # Tensorboard and logging
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Training params

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#,clipnorm=100)

    epoch_to_plot = 25  # Plot evey epoch_to_plot epochs
    val_perc = 0.2

    N_mics = 64
    filter_shape = int(N_mics * 2)
    nfft = int(129)
    early_stop_patience = 100

    # Load configuration
    # Load Green function

    G = np.load(green_function_path)
    P_gt = np.load(gt_soundfield_dataset_path)  # gt soundfield
    G_cp = G[params_circular.idx_lr[params_circular.idx_cp]]
    G = tf.convert_to_tensor(G)
    G_cp = tf.convert_to_tensor(G_cp)
    print('ciao')

    # Load dataset

    # Split train/val
    P_train, P_val, src_train, src_val = train_test_split( P_gt, params_circular.src_pos_train, test_size=val_perc)


    def concat_real_imag(P_, src):
        P_concat = tf.concat([tf.math.real(P_), tf.math.imag(P_)], axis=0)
        return P_concat, P_, src

    def preprocess_dataset(P, src):
        data_ds = tf.data.Dataset.from_tensor_slices((P, src))
        preprocessed_ds = data_ds.map(concat_real_imag)
        return preprocessed_ds

    train_ds = preprocess_dataset(P_train, src_train)
    val_ds = preprocess_dataset(P_val, src_val)

    loss_fn = tf.keras.losses.MeanAbsoluteError()
    metric_fn_train_real = tf.keras.metrics.MeanAbsoluteError()
    metric_fn_train_imag = tf.keras.metrics.MeanAbsoluteError()

    metric_fn_val_real = tf.keras.metrics.MeanAbsoluteError()
    metric_fn_val_imag = tf.keras.metrics.MeanAbsoluteError()

    filter_shape = int(args.n_loudspeakers*2)
    sf_shape =  int(len(params_circular.idx_cp)*2)
    N_freqs = params_circular.N_freqs

    # Network Model


    # Load Network
    network_model_filters = network_utils.pressure_matching_network(sf_shape, N_freqs,filter_shape)
    network_model_filters.summary()

    # Load Data
    train_ds = train_ds.shuffle(buffer_size=int(batch_size*2))
    val_ds = val_ds.shuffle(buffer_size=int(batch_size*2))

    train_ds = train_ds.batch(batch_size=batch_size)
    val_ds = val_ds.batch(batch_size=batch_size)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    @tf.function
    def train_step(P_concat, P_):
        with tf.GradientTape() as tape:
            # Compensate driving signals
            d_hat = network_model_filters(P_concat, training=True)[:, :, :, 0]
            d_complex = tf.cast(d_hat[:, :int(d_hat.shape[1] / 2)], dtype=tf.complex128) + (
                        tf.convert_to_tensor(1j, dtype=tf.complex128) * tf.cast(d_hat[:, int(d_hat.shape[1] / 2):],
                                                                                dtype=tf.complex128))

            p_est = tf.einsum('bij, kij-> bkj', d_complex, G_cp)
            loss_value_P = (lambda_abs*loss_fn(train_utils.normalize_tensor(tf.math.abs(P_)), train_utils.normalize_tensor(tf.math.abs(p_est)))) + \
                           loss_fn(train_utils.normalize_tensor(tf.math.angle(P_)), train_utils.normalize_tensor(tf.math.angle(p_est)))
        network_model_filters_grads = tape.gradient(loss_value_P, network_model_filters.trainable_weights)

        optimizer.apply_gradients(zip(network_model_filters_grads, network_model_filters.trainable_weights))

        metric_fn_train_real.update_state(train_utils.normalize_tensor(tf.math.abs(P_)), train_utils.normalize_tensor(tf.math.abs(p_est)))
        metric_fn_train_imag.update_state(train_utils.normalize_tensor(tf.math.angle(P_)), train_utils.normalize_tensor(tf.math.angle(p_est)))
        return loss_value_P

    @tf.function
    def val_step(P_concat, P_):
            # Compensate driving signals
            d_hat = network_model_filters(P_concat, training=False)[:, :, :, 0]

            d_complex = tf.cast(d_hat[:, :int(d_hat.shape[1] / 2)], dtype=tf.complex128) + (
                    tf.convert_to_tensor(1j, dtype=tf.complex128) * tf.cast(d_hat[:, int(d_hat.shape[1] / 2):],
                                                                            dtype=tf.complex128))
            p_est = tf.einsum('bij, kij-> bkj', d_complex, G_cp)
            loss_value_P = (lambda_abs*loss_fn(train_utils.normalize_tensor(tf.math.abs(P_)), train_utils.normalize_tensor(tf.math.abs(p_est)))) + \
                           loss_fn(train_utils.normalize_tensor(tf.math.angle(P_)), train_utils.normalize_tensor(tf.math.angle(p_est)))

            metric_fn_val_real.update_state(train_utils.normalize_tensor(tf.math.abs(P_)), train_utils.normalize_tensor(tf.math.abs(p_est)))
            metric_fn_val_imag.update_state(train_utils.normalize_tensor(tf.math.angle(P_)), train_utils.normalize_tensor(tf.math.angle(p_est)))
            return loss_value_P, d_hat

    for n_e in tqdm(range(epochs)):
        plot_val = True
        for P_concat, P, _ in train_ds:
            loss_value_P = train_step(P_concat, P)

        train_loss = metric_fn_train_imag.result() + metric_fn_train_real.result()
        # Log to tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('train_loss_P', train_loss, step=n_e)
        metric_fn_train_imag.reset_states()
        metric_fn_val_real.reset_states()

        for P_concat, P, src  in val_ds:
            loss_value_P_val, d_hat = val_step(P_concat, P)

        val_loss = metric_fn_val_imag.result() + metric_fn_val_real.result()

        # Every epoch_to_plot epochs plot an example of validation
        if not n_e % epoch_to_plot and plot_val:
            print('Train loss: ' + str(train_loss.numpy()))
            print('Val loss: ' + str(val_loss.numpy()))


            d_hat_complex = tf.cast(d_hat[:, :int(d_hat.shape[1] / 2)], dtype=tf.complex128) + (
                    tf.convert_to_tensor(1j, dtype=tf.complex128) * tf.cast(d_hat[:, int(d_hat.shape[1] / 2):],
                                                                            dtype=tf.complex128))

            n_s = np.random.randint(0, src.shape[0])
            idx_f = np.random.randint(0, N_freqs)
            idx_f = 41 # FIXEDD to 1000 Hzzz

            P_hat_real =  tf.math.real(tf.einsum('bij, kij-> bkj', d_hat_complex, G))[n_s]
            P_gt = np.zeros(P_hat_real.shape, dtype=complex)
            for n_f in range(N_freqs):
                P_gt[ :, n_f] = (1j / 4) * \
                               scipy.special.hankel2(0,
                                                     (params_circular.wc[n_f] / params_circular.c) *
                                                     np.linalg.norm(params_circular.point[:, :2] - src[n_s], axis=1))

            p_hat_real = np.reshape(P_hat_real[:, idx_f], (params_circular.N_sample, params_circular.N_sample))
            p_gt = np.reshape(np.real(P_gt[ :, idx_f]), (params_circular.N_sample, params_circular.N_sample))
            selection = np.ones_like(params_circular.array_pos[:, 0])
            figure_soundfield = plt.figure(figsize=(10, 20))
            plt.subplot(211)
            sfs.plot2d.amplitude(p_gt, params_circular.grid, xnorm=[0, 0, 0])
            sfs.plot2d.loudspeakers(params_circular.array.x, params_circular.array.n, selection)
            plt.title('GT')

            plt.subplot(212)
            sfs.plot2d.amplitude(p_hat_real, params_circular.grid, xnorm=[0, 0, 0])
            sfs.plot2d.loudspeakers(params_circular.array.x, params_circular.array.n, selection)
            plt.title('est')

            #plt.show()

            #figure_soundfield = train_utils.est_vs_gt_soundfield(tf.expand_dims(P_hat_real[:, :, :, idx_plot], axis=3), tf.expand_dims(P_real[:, :, :, idx_plot], axis=3))

            with summary_writer.as_default():
                tf.summary.image("soundfield second training", train_utils.plot_to_image(figure_soundfield), step=n_e)

            filters_fig = plt.figure()
            plt.plot(d_hat.numpy()[0, :, :])
            with summary_writer.as_default():
                tf.summary.image("Filters true second", train_utils.plot_to_image(filters_fig), step=n_e)

        # Select best model
        if n_e == 0:
            lowest_val_loss = val_loss
            network_model_filters.save(saved_model_path)
            early_stop_counter = 0

        else:
            if val_loss < lowest_val_loss:
                network_model_filters.save(saved_model_path)
                lowest_val_loss = val_loss
                early_stop_counter = 0
            else:
                #network_model_filters.save(saved_model_path)
                early_stop_counter = early_stop_counter + 1

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print('Training finished at epoch '+str(n_e))
            break

        # Log to tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('val_loss_P', val_loss, step=n_e)
            tf.summary.scalar('val_abs', metric_fn_val_real.result(), step=n_e)
            tf.summary.scalar('val_phase', metric_fn_val_imag.result(), step=n_e)

        metric_fn_val_real.reset_states()
        metric_fn_val_imag.reset_states()


if __name__ == '__main__':
    main()


