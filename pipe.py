from transformer import specific, nominal, numeric, scale, onehot
from sklearn.pipeline import make_pipeline, FeatureUnion

from sklearn.decomposition import PCA
from prince import MCA


def plain():
    return make_pipeline(specific(), onehot(), scale())


def pca():
    return make_pipeline(specific(), onehot(), scale(), PCA())


def mca():
    pipe_mca = make_pipeline(specific(), nominal(), MCA(n_components=25))
    pipe_numeric = make_pipeline(specific(), numeric())

    return make_pipeline(FeatureUnion([('mca_on_nominal', pipe_mca),
                                       ('rest_of_data', pipe_numeric)]),
                         scale())
