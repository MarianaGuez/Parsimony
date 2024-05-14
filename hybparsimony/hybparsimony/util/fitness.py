# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
import numpy as np
import warnings
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import math
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

def distanciaPromedio(centroides):
  sum = 0
  num_par = 0
  dist = []
  for i in range(centroides.shape[0]):
    for j in range(i):
      distancia = np.sqrt(np.sum(np.square(centroides[i] - centroides[j])))
      dist.append(distancia)
      num_par += 1
  promedio = np.sum(dist)/num_par
  return promedio,dist

def desviacionClusters(mean_distancias,distancias):
  return np.sqrt(np.sum(np.square(distancias - mean_distancias)))

class FitnessFunction:

  warnings.filterwarnings("ignore")
  def __init__(self, X, n_max, n_min):
    self.X = X
    self.n_max = n_max
    self.n_min = n_min

  def error_n_clusters(self, clusters):
    n_emotions = 6
    error_Clusters = abs((clusters-n_emotions)/((self.n_max-self.n_min)/2))
    return error_Clusters

  def desviacionStandard(self, labels):
    std = []
    for i in (np.unique(labels)):
      indices = np.array(np.where(labels == i))
      indices = np.array(indices[0])
      dist_a = []
      for i in range(indices.shape[0]):
        for j in range(i):
          dist = np.sqrt(np.sum(np.square(self.X[indices[i]] - self.X[indices[j]])))
          dist_a.append(dist)
      mean_dist = np.mean(dist_a)
      std.append(np.sqrt(np.sum(np.square(dist_a - mean_dist))))
    return np.mean(std)

  def fitness_function(self):
    range_n_clusters = np.linspace(self.n_min, self.n_max, (self.n_max - self.n_min)+1)
    max_avg = 0
    for n_clusters in range_n_clusters:
      clusterer = KMeans(n_clusters=int(n_clusters), random_state=10)
      cluster_labels = clusterer.fit_predict(self.X)
      silhouette_avg = silhouette_score(self.X, cluster_labels)
      if silhouette_avg > max_avg:
        max_avg = silhouette_avg
        clusters = n_clusters
        centers = clusterer.cluster_centers_

    error_clusters = self.error_n_clusters(clusters) # 0 -> 6
    mean_distancias,distancias = distanciaPromedio(centers) # > mejor
    std_puntos = self.desviacionStandard(cluster_labels)
    std_clusters = desviacionClusters(mean_distancias,distancias)
    mi_formula = error_clusters * mean_distancias

    return mean_distancias, std_clusters, std_puntos, error_clusters

def generic_complexity(model, nFeatures, **kwargs):
  r"""
  Generic complexity function.

  Parameters
  ----------
  model : model
      The model from which the internal complexity is calculated.
  nFeatures : int
      The number of the selected features.
  **kwargs :
      Other arguments.

  Returns
  -------
  int
      nFeatures.

  """
  return nFeatures


def getFitness(features, algorithm, complexity, custom_eval_fun=cross_val_score, ignore_warnings = True):
    r"""
     Fitness function for hybparsimony.

    Parameters
    ----------
    algorithm : object
        The machine learning algorithm to optimize.
    complexity : function
        A function that calculates the complexity of the model. There are some functions available in `hybparsimony.util.complexity`.
    custom_eval_fun : function
        An evaluation function similar to scikit-learns's 'cross_val_score()'

    Returns
    -------
    float
        np.array([model's fitness value (J), model's complexity]), model

    Examples
    --------
    Usage example for a binary classification model

    .. highlight:: python
    .. code-block:: python

        import pandas as pd
        import numpy as np
        from sklearn.datasets import load_breast_cancer
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from hybparsimony import hybparsimony
        from hybparsimony.util import getFitness, svm_complexity, population
        # load 'breast_cancer' dataset
        breast_cancer = load_breast_cancer()
        X, y = breast_cancer.data, breast_cancer.target
        chromosome = population.Chromosome(params = [1.0, 0.2],
                                        name_params = ['C','gamma'],
                                        const = {'kernel':'rbf'},
                                        cols= np.random.uniform(size=X.shape[1])>0.50,
                                        name_cols = breast_cancer.feature_names)
        print(getFitness(SVC,svm_complexity)(chromosome, X=X, y=y))
    """

    if algorithm is None:
        raise Exception("An algorithm function must be provided!!!")
    if complexity is None or not callable(complexity):
        raise Exception("A complexity function must be provided!!!")


    def fitness(cromosoma, **kwargs):

        if "pandas" in str(type(kwargs["X"])):
            kwargs["X"] = kwargs["X"].values
        if "pandas" in str(type(kwargs["y"])):
            kwargs["y"] = kwargs["y"].values

        X_train = kwargs["X"]
        y_train = kwargs["y"]

        try:
            # Extract features from the original DB plus response (last column)
            data_train_model = X_train[: , cromosoma.columns]

            if ignore_warnings:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"
            ##############
            # MODIFICATION
            ##############
            ########################## PAST CODE #################################
            # train the model
            #aux = algorithm(**cromosoma.params)
            #fitness_val = custom_eval_fun(aux, data_train_model, y_train).mean()
            #modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)
            ######################################################################

            diabetes_UMAP = umap.UMAP(n_neighbors=50,n_components=2,min_dist=0.1,metric='euclidean').fit_transform(data_train_model,y_train)
            clase_fitness = FitnessFunction(diabetes_UMAP,10,2)
            mean_distancias, std_clusters, std_puntos, error_clusters = clase_fitness.fitness_function()
            fitness_val = custom_eval_fun(mean_distancias, std_clusters, std_puntos, error_clusters) #+ 1/(np.sum(cromosoma.columns))

            indices = np.array(cromosoma.columns)
            array_caracteristicas = np.array(features)
            sel = array_caracteristicas[indices]

            #print(np.sum(cromosoma.columns))
            #print(fitness_val)
            modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)
            plot = [diabetes_UMAP,y_train]
            # Reset warnings to default values
            warnings.simplefilter("default")
            os.environ["PYTHONWARNINGS"] = "default"

            # El híbrido funciona de forma que cuanto más alto es mejor. Por tanto, con RMSE deberíamos trabajar con su negación.
            return np.array([fitness_val, generic_complexity(modelo, np.sum(cromosoma.columns))]),modelo,plot,sel
        except Exception as e:
            print(e)
            return np.array([np.NINF, np.Inf]), None

    return fitness



def fitness_for_parallel(algorithm, complexity, custom_eval_fun=cross_val_score, cromosoma=None, 
                         X=None, y=None, ignore_warnings = True):
    r"""
     Fitness function for hybparsimony similar to 'getFitness()' without being nested, to allow the pickle and therefore the parallelism.

    Parameters
    ----------
    algorithm : object
        The machine learning algorithm to optimize. 
    complexity : function
        A function that calculates the complexity of the model. There are some functions available in `hybparsimony.util.complexity`.
    custom_eval_fun : function
        An evaluation function similar to scikit-learns's 'cross_val_score()'.
    cromosoma: population.Chromosome class
        Solution's chromosome.
    X : {array-like, dataframe} of shape (n_samples, n_features)
        Input matrix.
    y : {array-like, dataframe} of shape (n_samples,)
        Target values (class labels in classification, real numbers in regression).
    ignore_warnings: True
        If ignore warnings.

    Returns
    -------
    float
        np.array([model's fitness value (J), model's complexity]), model

    Examples
    --------

    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from hybparsimony import hybparsimony
    from hybparsimony.util import svm_complexity, population
    from hybparsimony.util.fitness import fitness_for_parallel
    # load 'breast_cancer' dataset
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    chromosome = population.Chromosome(params = [1.0, 0.2],
                                       name_params = ['C','gamma'],
                                       const = {'kernel':'rbf'},
                                       cols= np.random.uniform(size=X.shape[1])>0.50,
                                       name_cols = breast_cancer.feature_names)
    print(fitness_for_parallel(SVC, svm_complexity, 
                               custom_eval_fun=cross_val_score,
                               cromosoma=chromosome, X=X, y=y))

    """

    if "pandas" in str(type(X)):
        X = X.values
    if "pandas" in str(type(y)):
        y = y.values

    X_train = X
    y_train = y

    try:
        # Extract features from the original DB plus response (last column)
        data_train_model = X_train[:, cromosoma.columns]

        if ignore_warnings:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

        # train the model
        aux = algorithm(**cromosoma.params)
        fitness_val = custom_eval_fun(aux, data_train_model, y_train).mean()
        modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)

        # Reset warnings to default values
        warnings.simplefilter("default")
        os.environ["PYTHONWARNINGS"] = "default"

        # El híbrido funciona de forma que cuanto más alto es mejor. Por tanto, con RMSE deberíamos trabajar con negativos.

        return np.array([fitness_val, complexity(modelo, np.sum(cromosoma.columns))]), modelo
    except Exception as e:
        print(e)
        return np.array([np.NINF, np.Inf]), None
