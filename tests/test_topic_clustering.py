"""Small regression checks for reporting and empty-input behavior."""
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from topic_clustering import article_tables, keyword_table, predict_topics


class TopicClusteringTests(unittest.TestCase):
    def test_keyword_support_and_weight_export(self):
        matrix = csr_matrix([[1., 0, 1], [1., 0, 0], [0, 1., 0], [0, 1., 0]])
        table = keyword_table(matrix, np.array(["court", "korea", "rare"]),
                              np.array([0, 0, 1, 1]), min_document_share=0.75)
        self.assertEqual(table["term"].tolist(), ["court", "korea"])
        self.assertTrue((table["contrast_score"] > 0).all())
        self.assertTrue((table["document_share"] == 1).all())

    def test_representatives_keep_article_identity(self):
        x = np.array([[0., 0.], [0.1, 0.], [3., 3.], [3.1, 3.]])
        model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x)
        frame = pd.DataFrame({"raw_id": [10, 20, 30, 40],
                              "title": ["a", "b", "c", "d"]}, index=[9, 2, 7, 1])
        articles, central, boundary = article_tables(frame, x, model,
                                                    representatives=1, boundary_examples=1)
        self.assertEqual(articles.raw_id.tolist(), [10, 20, 30, 40])
        self.assertEqual(len(central), 2)
        self.assertEqual(len(boundary), 2)
        self.assertTrue((articles.boundary_margin >= -1e-12).all())
        np.testing.assert_array_equal(articles.cluster, model.predict(x))

    def test_unknown_input_is_not_assigned_to_arbitrary_cluster(self):
        texts = ["court judge law", "court federal judge", "law judge case",
                 "korea missile nuclear", "korea nuclear test", "missile nuclear sanction"]
        pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=2, random_state=42),
                                 Normalizer(), KMeans(n_clusters=2, n_init=10, random_state=42))
        # Use the public stage names expected by the prediction helper.
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(list(zip(["tfidf", "lsa", "normalizer", "kmeans"],
                                     [stage for _, stage in pipeline.steps])))
        pipeline.fit(texts)
        actual = predict_topics(pipeline, [texts[0], "", "zzunknown"])
        self.assertEqual(actual[0], pipeline.predict([texts[0]])[0])
        self.assertEqual(actual[1:].tolist(), [-1, -1])


if __name__ == "__main__":
    unittest.main()
