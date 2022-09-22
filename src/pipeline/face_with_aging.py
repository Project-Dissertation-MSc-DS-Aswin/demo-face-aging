import sys
from context import Constants, Args
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA, KernelPCA
import whylogs
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.drift_synthesis_by_eigen_faces import DriftSynthesisByEigenFacesExperiment
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from tqdm import tqdm
from copy import copy
import pickle
import logging
import sys
import re
import tensorflow as tf
import imageio
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.manifold import Isomap

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.dataset = os.environ.get('dataset', 'agedb')
args.model = os.environ.get('model', 'facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.CACD_DATADIR)
args.grouping_distance_type = os.environ.get('grouping_distance_type', constants.EIGEN_FACES_DISTANCES_GROUPING)
args.grouping_distance_cutoff_range = os.environ.get('grouping_distance_cutoff_range')
args.batch_size = os.environ.get('batch_size', 128)
args.preprocess_prewhiten = os.environ.get('preprocess_prewhiten', 1)
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.CACD_FACENET_INFERENCES)
args.pca_covariates_pkl = os.environ.get('pca_covariates_pkl', constants.CACD_PCA_COVARIATES)
args.metadata = os.environ.get('metadata', constants.CACD_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_with_aging')
args.no_of_samples = os.environ.get('no_of_samples', 2248)
args.no_of_pca_samples = os.environ.get('no_of_pca_samples', 2248)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.classifier = os.environ.get('classifier', constants.CACD_FACE_CLASSIFIER)
args.drift_synthesis_filename = os.environ.get('drift_synthesis_filename', constants.CACD_DRIFT_SYNTHESIS_EDA_CSV_FILENAME)
args.experiment_id = os.environ.get('experiment_id', 0)
args.drift_source_filename = os.environ.get('drift_source_filename', constants.AGEDB_DRIFT_SOURCE_FILENAME)
args.pca_type = os.environ.get('pca_type', 'PCA')
args.bins = os.environ.get('bins', np.ceil(0.85 * 239))
args.noise_error = os.environ.get('noise_error', 0)
args.mode = os.environ.get('mode', 'image_reconstruction')
args.drift_beta = os.environ.get('drift_beta', 1)
args.covariates_beta = os.environ.get('covariates_beta', 1)
args.drift_type = os.environ.get('drift_type', 'incremental')
args.filenames = os.environ.get('filenames', '')

parameters = list(
    map(lambda s: re.sub('$', '"', s),
        map(
            lambda s: s.replace('=', '="'),
            filter(
                lambda s: s.find('=') > -1 and bool(re.match(r'[A-Za-z0-9_]*=[.\/A-Za-z0-9]*', s)),
                sys.argv
            )
    )))

for parameter in parameters:
    logging.warning('Parameter: ' + parameter)
    exec("args." + parameter)
    
args.batch_size = int(args.batch_size)
args.preprocess_prewhiten = int(args.preprocess_prewhiten)
args.no_of_samples = int(args.no_of_samples)
args.no_of_pca_samples = int(args.no_of_pca_samples)
args.bins = int(args.bins)
args.noise_error = int(args.noise_error)
args.drift_beta = float(args.drift_beta)
args.covariates_beta = float(args.covariates_beta)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)

def images_covariance(images_new, no_of_images):
    """
    Images covariance
    @param images_new:
    @param no_of_images:
    @return: np.ndarray
    """
    images_cov = np.cov(images_new.reshape(no_of_images, -1))
    return images_cov

def demean_images(images_bw, no_of_images):
    """
    Demean the images
    @param images_bw:
    @param no_of_images:
    @return: np.ndarray
    """
    images_mean = np.mean(images_bw.reshape(no_of_images, -1), axis=1)
    images_new = (images_bw.reshape(no_of_images, -1) - images_mean.reshape(no_of_images, 1))

    return images_new

def collect_images(train_iterator):
    """
    Collect the images from iterator
    @param train_iterator:
    @return: np.ndarray
    """
    images_bw = []
    # Get input and output tensors
    for ii in tqdm(range(len(train_iterator))):
        (X, y) = train_iterator[ii]
        images_bw.append(X)

    return np.vstack(images_bw)

def pca_covariates(images_cov, pca_type='PCA', covariates_beta=1, seed=1000):
    """
    Apply PCA on Image Covariates
    @param images_cov:
    @param pca_type:
    @param covariates_beta:
    @param seed:
    @return: tuple()
    """
    pca = KernelPCA(n_components=images_cov.shape[0], kernel='poly') if pca_type == 'KernelPCA' else PCA(n_components=images_cov.shape[0])
    np.random.seed(seed)
    X_pca = pca.fit_transform(images_cov * np.random.normal(0, covariates_beta, size=images_cov.shape) if covariates_beta else images_cov)
    return pca.components_.T if pca_type == 'PCA' else pca.eigenvectors_, pca, X_pca

def isomap_images(images_bw):
    """
    Apply Isomap on Images
    @param images_bw:
    @return: tuple()
    """
    isomap = Isomap(n_components=images_bw.shape[0])
    X_transform = isomap.fit(images_bw)
    return isomap.embedding_vectors_, isomap, X_transform

def load_dataset(args, whylogs, image_dim, no_of_samples, colormode):
    """
    Load the dataset
    @param args:
    @param whylogs:
    @param image_dim:
    @param no_of_samples:
    @param colormode:
    @return: tuple()
    """
    dataset = None
    augmentation_generator = None
    if args.dataset == "agedb":
        augmentation_generator = get_augmented_datasets()
        dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, dim=image_dim, 
                               batch_size=args.batch_size)
    elif args.dataset == "cacd":
        augmentation_generator = get_augmented_datasets()
        dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                                  color_mode=colormode, augmentation_generator=augmentation_generator,
                                  data_dir=args.data_dir, dim=image_dim, batch_size=args.batch_size)
    elif args.dataset == "fgnet":
        augmentation_generator = get_augmented_datasets()
        dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None,
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, dim=image_dim, 
                               batch_size=args.batch_size)

    return dataset, augmentation_generator

def get_reduced_metadata(args, dataset, seed=1000):
  """
  Get reduced metadata
  @param args:
  @param dataset:
  @param seed:
  @return: pd.DataFrame()
  """
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(seed)
    if args.mode == 'image_reconstruction':
        filenames = pd.read_csv(args.drift_source_filename)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)
        
        return dataset.metadata.loc[result_idx].reset_index()
    elif args.mode == 'image_perturbation':
        filenames = pd.read_csv(args.drift_source_filename)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)
        
        return dataset.metadata.loc[result_idx].reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    if args.mode == 'image_reconstruction':
        filenames = pd.read_csv(args.drift_source_filename)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)
        
        return dataset.metadata.loc[result_idx].sample(args.no_of_samples).reset_index()
    elif args.mode == 'image_perturbation':
        filenames = pd.read_csv(args.drift_source_filename)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)
    else:
        names = dataset.metadata.groupby('name').count()
        names = names[names['age'] > 120]
        names = names.index.get_level_values(0)
        idx = [dataset.metadata['name'] == name for name in names]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)
    
        return dataset.metadata.loc[result_idx].sample(args.no_of_samples).reset_index()

if __name__ == "__main__":

    # set mlflow tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)

    # choose model
    if args.model == 'FaceNetKeras':
      model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    elif args.model == 'FaceRecognitionBaselineKeras':
      model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)

    # load the model
    model_loader.load_model()

    # load the dataset for PCA experiment
    dataset, augmentation_generator = load_dataset(args, whylogs, (96,96), args.no_of_pca_samples, 'grayscale')
    # load the dataset for experiment
    experiment_dataset, augmentation_generator = load_dataset(args, whylogs, (args.input_shape[1], args.input_shape[2]), args.no_of_samples, 'rgb')
    
    # pca args copy
    pca_args = copy(args)
    pca_args.no_of_samples = pca_args.no_of_pca_samples
    # set metadata to dataset for PCA experiment
    dataset.set_metadata(
        get_reduced_metadata(pca_args, dataset)
    )
    
    # set metadata to experiment dataset
    experiment_dataset.set_metadata(
        get_reduced_metadata(args, experiment_dataset)
    )

    # collect the images from PCA dataset iterator
    images_bw = collect_images(dataset.iterator)
    if args.noise_error:
        np.random.seed(1000)
        print("Adding error in b/w images of " + str(args.noise_error))
        images_bw += np.random.normal(0, args.noise_error, size=(images_bw.shape))
    # demean the images
    images_new = demean_images(images_bw, len(images_bw))
    # Apply PCA on image covariates
    if not os.path.isfile(args.pca_covariates_pkl):
        images_cov = images_covariance(images_new, len(images_new))
        P, pca, X_pca = pca_covariates(images_cov, args.pca_type, args.covariates_beta)
        
        # pickle.dump(pca, open(args.pca_covariates_pkl, "wb"))
    else:
        pca = pickle.load(open(args.pca_covariates_pkl, "rb"))

    print(pca)
    # create the experiment for dataset and experiment dataset
    experiment = DriftSynthesisByEigenFacesExperiment(args, dataset, experiment_dataset, logger=whylogs, model_loader=model_loader, pca=pca,
                                                      init_offset=0)

    # Find the eigen vector to apply for finding the images
    P_pandas = pd.DataFrame(pca.components_.T if args.pca_type == 'PCA' else pca.eigenvectors_,
                            columns=list(range(pca.components_.T.shape[1] if args.pca_type == 'PCA' else pca.eigenvectors_.shape[1])))

    # find the index
    index = experiment.dataset.metadata['age'].reset_index()

    # collect the images from dataset on real experiment
    images = collect_images(experiment_dataset.iterator)
    if args.noise_error:
        np.random.seed(1000)
        print("Adding error of " + str(args.noise_error))
        images += np.random.normal(0, args.noise_error, size=(images.shape))
    # find the eigen vectors
    eigen_vectors = experiment.eigen_vectors()
    # find the eigen vector coefficient
    b_vector = experiment.eigen_vector_coefficient(eigen_vectors, images_new)
    # in image_reconstruction mode
    if args.mode == 'image_reconstruction':
        weights_vector, offset, mean_age, std_age, age = experiment.weights_vector(experiment.dataset.metadata, b_vector)
        
        b_vector_new = tf.constant(np.expand_dims(b_vector, 2), dtype=tf.float64)
        error = tf.reduce_mean(age - offset - (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)))
        offset *= std_age
        offset += mean_age
        weights_vector *= std_age
        real_error = tf.reduce_mean(experiment.dataset.metadata['age'].values - offset - \
            (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)) * std_age)
    
        print("""
            Error: {error}, 
            Real Error: {real_error}
            """.format(error=error, real_error=real_error))
        
        print("Taken image_reconstruction choice")
        
    # in image_perturbation mode
    elif args.mode == 'image_perturbation':
        # weights vector dimensions
        weights_vector = experiment.weights_vector_perturbation(experiment.dataset.metadata, b_vector, init_offset=0)
        offset = 0
        
    offset_range = np.arange(0.01, -0.01, -0.02)
    if args.log_images == 's3':

        experiment.dataset.metadata['identity_grouping_distance'] = 0.0

        # apply mahalanobis distances
        distances = experiment.mahalanobis_distance(b_vector)
        # set mahalanobis distances
        experiment.dataset.metadata['identity_grouping_distance'] = distances
        
        # choose DISTINCT or DISTRIBUTION mode of execution for finding the hash_samples
        if args.grouping_distance_type == 'DISTINCT':
            experiment.dataset.metadata = experiment.set_hash_sample_by_distinct(experiment.dataset.metadata['identity_grouping_distance'])
        elif args.grouping_distance_type == 'DIST':
            experiment.dataset.metadata = experiment.set_hash_sample_by_dist(experiment.dataset.metadata['identity_grouping_distance'])
        
    #     figures, choices_array = experiment.plot_images_with_eigen_faces(
    #         images, images_new, weights_vector, offset, b_vector, offset_range, P_pandas, index
    #     )
        
    #     hash_samples = np.unique(experiment.dataset.metadata['hash_sample'])
        
    #     experiment_name = "FaceNet with Aging Drift (modified)"
    #     mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    #     experiment_id = None
    #     if experiment is not None:
    #         experiment_id = mlflow_experiment.experiment_id
        
    #     if experiment_id is None:
    #         experiment_id = mlflow.create_experiment(experiment_name)
    #     with mlflow.start_run(experiment_id=experiment_id):
    #         # hash_sample
    #         for ii, figure in enumerate(figures):
    #             # offset range
    #             for jj, fig in enumerate(figure):
    #                 # aging function
    #                 fig1, fig2, fig3, imgs, ages = tuple(fig)
    #                 mlflow.log_figure(fig1, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, str(hash_samples[ii]), str(jj), 'aging'))
    #                 mlflow.log_figure(fig2, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, str(hash_samples[ii]), str(jj), 'actual'))
    #                 mlflow.log_figure(fig3, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, str(hash_samples[ii]), str(jj), 'predicted'))
    #                 os.makedirs("""{0}/hash_sample_{1}""".format(args.logger_name, str(hash_samples[ii])), exist_ok=True)
    
    # get the voting classifier array
    voting_classifier_array = pickle.load(open(args.classifier, 'rb'))
    
    filenames = pd.read_csv(args.drift_source_reduced_filename)
    idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
        result_idx = np.logical_or(result_idx, i)
    
    images = images[result_idx]
    images_new = images_new[result_idx]
    
    experiment.dataset.metadata = experiment.dataset.metadata[result_idx].reset_index()
    
    # collect the drift predictions
    predictions_classes_array, _, _ = experiment.collect_drift_predictions(images, images_new,
                                        weights_vector, offset, b_vector, offset_range, P_pandas, index, 
                                        voting_classifier_array, model_loader, drift_beta=args.drift_beta, covariates_beta=args.covariates_beta, result_idx=result_idx)
    
    np.save(open('predictions_classes_array.npy', 'wb'), np.array(predictions_classes_array))
    # set the predictions to a dataframe
    predictions_classes = pd.DataFrame(predictions_classes_array,
                        columns=['hash_sample', 'offset', 'covariates_beta', 'drift_beta', 'true_identity', 'age', 'filename', 
                        'y_pred', 'proba_pred', 'y_drift', 'proba_drift', 'predicted_age', 'euclidean', 'cosine', 'identity_grouping_distance', 
                        'orig_TP', 'orig_FN', 'virtual_TP', 'virtual_FN', 'stat_TP', 'stat_FP', 'stat_undefined'])
    
    # save the predictions to CSV file
    predictions_classes.to_csv(args.drift_synthesis_filename)
    
    experiment.dataset.metadata.to_csv("../data_collection/experiment_dataset_metadata.csv")
    
    # predictions_classes = pd.read_csv(args.drift_synthesis_filename)
    
    recall = predictions_classes['orig_TP'].sum() / (predictions_classes['orig_FN'].sum() + predictions_classes['orig_TP'].sum())
    precision = 1.0
    accuracy = (predictions_classes['orig_TP'].sum()) / (predictions_classes['orig_TP'].sum() + predictions_classes['orig_FN'].sum())
    f1 = 2 * recall * precision / (recall + precision)
    
    recall_virtual = predictions_classes['virtual_TP'].sum() / (predictions_classes['virtual_FN'].sum() + predictions_classes['virtual_TP'].sum())
    precision_virtual = 1.0
    accuracy_virtual = (predictions_classes['virtual_TP'].sum()) / (predictions_classes['virtual_TP'].sum() + predictions_classes['virtual_FN'].sum())
    f1_virtual = 2 * recall_virtual * precision_virtual / (recall_virtual + precision_virtual)
    
    recall_drift = recall_score(predictions_classes['orig_TP'], predictions_classes['stat_FP'])
    precision_drift = 1.0
    accuracy_drift = accuracy_score(predictions_classes['orig_TP'], predictions_classes['stat_FP'])
    f1_drift = 2 * recall_drift * precision_drift / (recall_drift + precision_drift)
    roc_drift = roc_auc_score(predictions_classes['orig_TP'], predictions_classes['stat_FP'])
    
    print("""
        Drift Source - Original Image
        -----------------------------
        Recall of prediction: {recall}, 
        Precision of prediction: {precision}, 
        F1 of prediction: {f1},  
        Accuracy of prediction: {accuracy}, 
        
        Drift Source - Reconstructed Image
        ----------------------------------
        Recall of prediction: {recall_virtual}, 
        Precision of prediction: {precision_virtual}, 
        F1 of prediction: {f1_virtual},  
        Accuracy of prediction: {accuracy_virtual}, 
        
        Statistical Drift Detected
        --------------------------
        Accuracy of Drift: {accuracy_drift}, 
        Recall of Drift: {recall_drift}, 
        Precision of Drift: {precision_drift}, 
        F1 of Drift: {f1_drift}, 
        ROC of Drift: {roc_drift}, 
    """.format(
        accuracy=accuracy, 
        f1=f1, 
        precision=precision, 
        recall=recall, 
        accuracy_virtual=accuracy_virtual, 
        f1_virtual=f1_virtual, 
        precision_virtual=precision_virtual, 
        recall_virtual=recall_virtual, 
        accuracy_drift=accuracy_drift, 
        f1_drift=f1_drift, 
        precision_drift=precision_drift, 
        recall_drift=recall_drift, 
        roc_drift=roc_drift
    ))
    
    # mlflow log metrics
    # mlflow.log_metric("recall", recall)
    # mlflow.log_metric("precision", precision)
    # mlflow.log_metric("f1", f1)
    # mlflow.log_metric("accuracy", accuracy)
    
    # # start mlflow experiment
    # with mlflow.start_run(experiment_id=args.experiment_id):
    #     figure = experiment.plot_histogram_of_face_distances()
    #     mlflow.log_figure(figure, "histogram_of_face_distances.png")
    #     figure = experiment.plot_scatter_of_drift_confusion_matrix()
    #     mlflow.log_figure(figure, "scatter_plot_of_drift_true_positives_false_negatives.png")
        
    